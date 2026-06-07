"""
Image embedding module for generating SigLIP embeddings.
Supports both numpy array batches (fast path from FFmpeg) and PIL images.

SigLIP ViT-B-16-SigLIP-512 normalization: mean=0.5, std=0.5
  → tensor = uint8 / 127.5 - 1.0  (single vectorized op, no PIL needed)
"""

import logging
from typing import List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import open_clip
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ImageEmbedding:
    """Represents an image embedding with metadata."""
    embedding: np.ndarray
    timestamp: float
    frame_number: int
    image_size: Tuple[int, int]


class ImageEmbedder:
    """
    Generates image embeddings using SigLIP.
    Fast path: numpy uint8 arrays → tensor (no PIL).
    """

    def __init__(self, num_instances: int = 1):
        self.model_name = "ViT-B-16-SigLIP-512"
        self.pretrained = "webli"
        self.num_instances = num_instances
        self.device = self._determine_best_device()

        self.models = []
        self.preprocess = None  # Kept for PIL fallback path
        self._embedding_dim = None

        logger.info(f"ImageEmbedder configured on {self.device}")

    def _determine_best_device(self) -> str:
        """Determine the best available device. Respects CUDA_VISIBLE_DEVICES."""
        try:
            if torch.cuda.is_available():
                return "cuda"
            else:
                logger.warning("No CUDA available, using CPU for image models")
                return "cpu"
        except Exception:
            return "cpu"

    def _load_models(self) -> None:
        """Load SigLIP model instances."""
        if self.models:
            return

        logger.info(f"Loading {self.num_instances} SigLIP model instance(s) on {self.device}...")

        for i in range(self.num_instances):
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name, pretrained=self.pretrained, device=self.device
                )
                model.eval()
                self.models.append(model)

                if self.preprocess is None:
                    self.preprocess = preprocess

                if self._embedding_dim is None:
                    with torch.no_grad():
                        dummy = torch.randn(1, 3, 512, 512).to(self.device)
                        self._embedding_dim = model.encode_image(dummy).shape[-1]

            except Exception as e:
                if "cuda" in self.device:
                    logger.warning(f"CUDA failed ({e}), falling back to CPU")
                    self.device = "cpu"
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        self.model_name, pretrained=self.pretrained, device=self.device
                    )
                    model.eval()
                    self.models.append(model)
                    if self.preprocess is None:
                        self.preprocess = preprocess
                    if self._embedding_dim is None:
                        with torch.no_grad():
                            dummy = torch.randn(1, 3, 512, 512).to(self.device)
                            self._embedding_dim = model.encode_image(dummy).shape[-1]
                else:
                    raise

        logger.info(f"SigLIP models loaded ({len(self.models)} instances, dim={self._embedding_dim})")

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._load_models()
        return self._embedding_dim

    def encode_numpy_batch(self, pixels: np.ndarray) -> np.ndarray:
        """
        Fast path: encode a batch of numpy uint8 frames directly.
        Applies SigLIP normalization (mean=0.5, std=0.5) as a single
        vectorized tensor operation — no PIL, no per-image preprocessing.

        Args:
            pixels: (N, H, W, 3) uint8 numpy array

        Returns:
            (N, embedding_dim) float32 normalized embeddings
        """
        self._load_models()
        model = self.models[0]

        # uint8 (N,H,W,3) → float32 (N,3,H,W) normalized to [-1, 1]
        # SigLIP: mean=0.5, std=0.5 → (x/255 - 0.5) / 0.5 = x/127.5 - 1.0
        t = torch.from_numpy(pixels).permute(0, 3, 1, 2).float()  # (N,3,H,W)
        t = t.div_(127.5).sub_(1.0)  # In-place for speed
        t = t.to(self.device)

        with torch.no_grad():
            if "cuda" in self.device:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    emb = model.encode_image(t)
                    emb = F.normalize(emb, dim=-1)
                    return emb.cpu().numpy()
            else:
                emb = model.encode_image(t)
                emb = F.normalize(emb, dim=-1)
                return emb.numpy()

    def encode_images_batch(self, images: list, batch_size: int = 256) -> np.ndarray:
        """
        Encode a list of images. Accepts either:
          - numpy uint8 arrays (fast path, no PIL)
          - PIL Images (legacy path, uses preprocess)

        Args:
            images: List of numpy arrays or PIL Images
            batch_size: Max images per GPU forward pass

        Returns:
            (N, embedding_dim) float32 normalized embeddings
        """
        self._load_models()

        if not images:
            return np.array([])

        # Detect input type from first element
        first = images[0]
        is_numpy = isinstance(first, np.ndarray)

        if is_numpy:
            # Fast path: stack numpy arrays and encode in one shot
            pixels = np.stack(images)  # (N, H, W, 3) uint8
            all_emb = []
            for i in range(0, len(pixels), batch_size):
                all_emb.append(self.encode_numpy_batch(pixels[i:i + batch_size]))
            return np.vstack(all_emb) if all_emb else np.array([])
        else:
            # Legacy PIL path
            from PIL import Image
            model = self.models[0]
            all_emb = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                tensors = []
                for img in batch:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    tensors.append(self.preprocess(img))
                batch_tensor = torch.stack(tensors).to(self.device)
                with torch.no_grad():
                    if "cuda" in self.device:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            emb = model.encode_image(batch_tensor)
                            emb = F.normalize(emb, dim=-1)
                            all_emb.append(emb.cpu().numpy())
                    else:
                        emb = model.encode_image(batch_tensor)
                        emb = F.normalize(emb, dim=-1)
                        all_emb.append(emb.numpy())
            return np.vstack(all_emb) if all_emb else np.array([])

    def encode_video_frames(
        self,
        frames: List,
        batch_size: int = 256
    ) -> List[ImageEmbedding]:
        """Generate embeddings for video frames."""
        images = [frame.image for frame in frames]
        embeddings = self.encode_images_batch(images, batch_size)

        return [
            ImageEmbedding(
                embedding=emb, timestamp=frame.timestamp,
                frame_number=frame.frame_number,
                image_size=(512, 512)
            )
            for frame, emb in zip(frames, embeddings)
        ]
