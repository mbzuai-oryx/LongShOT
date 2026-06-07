from typing import Any
from pydantic import BaseModel


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[dict[str, Any]] = []
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 1.0
    extra_body: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


class PreloadRequest(BaseModel):
    paths: list[str] = []
    media_type: str = "auto"  # "video", "audio", "image", or "auto" (detect by extension)
