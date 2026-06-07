"""
GPU scheduler for parallel model evaluation.

Manages a pool of GPUs and runs multiple eval.py instances concurrently,
each with its own GPU allocation and vLLM port. Shows live progress bars.

Usage:
    python scheduler.py --models "ModelA:2" "ModelB:1" "ModelC:1" \
        --tasks postvalid_v2 --output_dir results_postvalid
"""

import argparse
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta

import psutil
import requests
from tqdm import tqdm


BASE_PORT = 8100
POLL_INTERVAL = 3

# Status constants
STATUS_LOADING = "loading"
STATUS_READY = "ready"
STATUS_GENERATING = "generating"


def parse_args():
    parser = argparse.ArgumentParser(description='Parallel GPU scheduler for model evaluation')
    parser.add_argument('--models', nargs='+', required=True,
                        help='Models in "name:num_gpus" format')
    parser.add_argument('--tasks', type=str, required=True,
                        help='Task name (e.g., postvalid_v2)')
    parser.add_argument('--output_dir', type=str, default='results_postvalid',
                        help='Output directory')
    parser.add_argument('--num_workers', type=int, default=256,
                        help='Number of inference workers per model')
    parser.add_argument('--config-file', type=str, default='config.yaml',
                        help='Path to system config YAML')
    parser.add_argument('--min-gpu-memory', type=float, default=50,
                        help='Minimum free GPU memory in GiB required to assign a GPU (default: 50)')
    parser.add_argument('--no-stagger', action='store_true',
                        help='Launch all models at once instead of waiting for each server to be ready first')
    return parser.parse_args()


def detect_gpu_vendor():
    """Detect whether the system has NVIDIA or AMD GPUs (or neither)."""
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if r.returncode == 0:
            return "nvidia"
    except FileNotFoundError:
        pass
    try:
        r = subprocess.run(["rocm-smi"], capture_output=True, timeout=5)
        if r.returncode == 0:
            return "amd"
    except FileNotFoundError:
        pass
    return "unknown"


GPU_VENDOR = detect_gpu_vendor()


def get_available_gpus():
    if GPU_VENDOR == "amd":
        visible = os.environ.get("HIP_VISIBLE_DEVICES",
                                 os.environ.get("ROCR_VISIBLE_DEVICES",
                                                os.environ.get("CUDA_VISIBLE_DEVICES", "")))
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        return list(range(8))
    return [int(g.strip()) for g in visible.split(",")]


# ---------------------------------------------------------------------------
# NVIDIA helpers
# ---------------------------------------------------------------------------

def _nvidia_get_gpu_free_memory():
    """Return dict mapping physical GPU index to free memory in GiB (NVIDIA)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        gpu_free = {}
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                gpu_idx = int(parts[0].strip())
                free_mib = float(parts[1].strip())
                gpu_free[gpu_idx] = free_mib / 1024
        return gpu_free
    except Exception:
        return {}


def _nvidia_collect_gpu_info():
    """Collect GPU hardware info via nvidia-smi."""
    info = {"vendor": "nvidia", "driver_version": None, "runtime": "cuda", "runtime_version": None, "gpus": {}}
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free,pcie.link.gen.current,pcie.link.width.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            idx = int(parts[0])
            info["gpus"][idx] = {
                "name": parts[1],
                "memory_total_gib": round(float(parts[2]) / 1024, 1),
                "memory_free_gib": round(float(parts[3]) / 1024, 1),
                "pcie_gen": parts[4] if len(parts) > 4 else None,
                "pcie_width": parts[5] if len(parts) > 5 else None,
            }
        ver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        driver = ver.stdout.strip().split("\n")[0].strip()
        if driver:
            info["driver_version"] = driver
        header = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        for hl in header.stdout.split("\n"):
            if "CUDA Version" in hl:
                match = re.search(r"CUDA Version:\s*([\d.]+)", hl)
                if match:
                    info["runtime_version"] = match.group(1)
                break
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# AMD helpers
# ---------------------------------------------------------------------------

def _amd_get_gpu_free_memory():
    """Return dict mapping physical GPU index to free memory in GiB (AMD/ROCm)."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        gpu_free = {}
        for key, val in data.items():
            match = re.match(r"card(\d+)", key)
            if not match:
                continue
            idx = int(match.group(1))
            # rocm-smi JSON uses "VRAM Total Used (B)" / "VRAM Total Memory (B)"
            total_b = int(val.get("VRAM Total Memory (B)", 0))
            used_b = int(val.get("VRAM Total Used (B)", 0))
            gpu_free[idx] = (total_b - used_b) / (1024 ** 3)
        return gpu_free
    except Exception:
        pass
    # Fallback: parse tabular output
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10
        )
        gpu_free = {}
        current_gpu = None
        total_b = used_b = 0
        for line in result.stdout.split("\n"):
            gm = re.match(r"GPU\[(\d+)\]", line)
            if gm:
                if current_gpu is not None:
                    gpu_free[current_gpu] = (total_b - used_b) / (1024 ** 3)
                current_gpu = int(gm.group(1))
                total_b = used_b = 0
            if "Total" in line and "Used" not in line:
                m = re.search(r"(\d+)", line.split(":")[-1])
                if m:
                    total_b = int(m.group(1))
            if "Used" in line:
                m = re.search(r"(\d+)", line.split(":")[-1])
                if m:
                    used_b = int(m.group(1))
        if current_gpu is not None:
            gpu_free[current_gpu] = (total_b - used_b) / (1024 ** 3)
        return gpu_free
    except Exception:
        return {}


def _amd_collect_gpu_info():
    """Collect GPU hardware info via rocm-smi."""
    info = {"vendor": "amd", "driver_version": None, "runtime": "rocm", "runtime_version": None, "gpus": {}}
    # Get ROCm version
    try:
        for rocm_path in ["/opt/rocm/.info/version", "/opt/rocm/include/rocm-core/rocm_version.h"]:
            if os.path.exists(rocm_path):
                with open(rocm_path) as f:
                    content = f.read().strip()
                # version file has plain version string; header has #define
                m = re.search(r"(\d+\.\d+[\d.]*)", content)
                if m:
                    info["runtime_version"] = m.group(1)
                    break
    except Exception:
        pass
    # Get driver version from modinfo or rocm-smi
    try:
        r = subprocess.run(["modinfo", "amdgpu", "-F", "version"],
                           capture_output=True, text=True, timeout=5)
        v = r.stdout.strip().split("\n")[0].strip()
        if v:
            info["driver_version"] = v
    except Exception:
        pass
    # Per-GPU info via rocm-smi --json
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram",
             "--showpciebw", "--json"],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        for key, val in data.items():
            match = re.match(r"card(\d+)", key)
            if not match:
                continue
            idx = int(match.group(1))
            total_b = int(val.get("VRAM Total Memory (B)", 0))
            used_b = int(val.get("VRAM Total Used (B)", 0))
            info["gpus"][idx] = {
                "name": val.get("Card Series", val.get("Card series", "AMD GPU")),
                "memory_total_gib": round(total_b / (1024 ** 3), 1),
                "memory_free_gib": round((total_b - used_b) / (1024 ** 3), 1),
                "pcie_gen": None,
                "pcie_width": None,
            }
        return info
    except Exception:
        pass
    # Fallback: at least enumerate GPUs from free memory
    gpu_free = _amd_get_gpu_free_memory()
    for idx, free_gib in gpu_free.items():
        info["gpus"][idx] = {
            "name": "AMD GPU",
            "memory_total_gib": None,
            "memory_free_gib": round(free_gib, 1),
            "pcie_gen": None,
            "pcie_width": None,
        }
    return info


# ---------------------------------------------------------------------------
# Vendor-agnostic wrappers
# ---------------------------------------------------------------------------

def get_gpu_free_memory():
    """Return dict mapping physical GPU index to free memory in GiB."""
    if GPU_VENDOR == "amd":
        return _amd_get_gpu_free_memory()
    return _nvidia_get_gpu_free_memory()


def collect_gpu_info():
    """Query GPU hardware info, auto-detecting NVIDIA or AMD."""
    if GPU_VENDOR == "amd":
        return _amd_collect_gpu_info()
    return _nvidia_collect_gpu_info()


def write_run_info(output_dir, model_name, allocated_gpus, port, gpu_info, args, run_id):
    """Write run_info.json into the model's output directory."""
    model_dir = os.path.join(output_dir, to_underscored(model_name))
    os.makedirs(model_dir, exist_ok=True)
    run_info_path = os.path.join(model_dir, f"{to_underscored(model_name)}_run_info.json")

    # Build per-allocated-GPU details
    gpu_details = []
    for g in allocated_gpus:
        detail = gpu_info["gpus"].get(g, {})
        gpu_details.append({"gpu_index": g, **detail})

    entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_vendor": gpu_info.get("vendor", GPU_VENDOR),
        "driver_version": gpu_info.get("driver_version"),
        "runtime": gpu_info.get("runtime"),
        "runtime_version": gpu_info.get("runtime_version"),
        "gpu_count": len(allocated_gpus),
        "gpu_indices": allocated_gpus,
        "gpus": gpu_details,
        "tensor_parallel_size": len(allocated_gpus),
        "port": int(port),
        "num_workers": args.num_workers,
        "task": args.tasks,
        "output_dir": args.output_dir,
    }

    # Append to existing runs list (supports multiple runs on different machines)
    existing = []
    if os.path.exists(run_info_path):
        try:
            with open(run_info_path, 'r') as f:
                data = json.load(f)
                existing = data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    existing.append(entry)

    with open(run_info_path, 'w') as f:
        json.dump(existing, f, indent=2)

    return run_info_path


def filter_gpus_by_memory(gpu_ids, min_free_gib=80):
    """Filter GPU list to only include GPUs with enough free memory."""
    gpu_free = get_gpu_free_memory()
    if not gpu_free:
        return gpu_ids  # Can't check, return all

    available = []
    blocked = []
    for g in gpu_ids:
        free = gpu_free.get(g, 0)
        if free >= min_free_gib:
            available.append(g)
        else:
            blocked.append((g, free))

    for g, free in blocked:
        log(f"GPU {g} has only {free:.1f} GiB free (need {min_free_gib} GiB) — skipping")

    return available


def short_name(model_name):
    return model_name.split("/")[-1]


def to_underscored(model_name):
    part = model_name.split("/")[-1] if "/" in model_name else model_name
    return re.sub(r'[^a-zA-Z0-9]', '_', part)


MOUNT_BASE = "/tmp/hf-mount-models"


def hf_mount(model_name):
    """Mount a HuggingFace model repo via hf-mount. Returns the mount path."""
    safe = model_name.replace("/", "--")
    mount_path = os.path.join(MOUNT_BASE, safe)
    os.makedirs(mount_path, exist_ok=True)

    # Check if already mounted
    if os.path.exists(os.path.join(mount_path, "config.json")):
        log(f"Already mounted: {model_name} → {mount_path}")
        return mount_path

    cmd = ["hf-mount", "start", "--fuse", "repo", model_name, mount_path]
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        cmd.insert(2, f"--hf-token={hf_token}")

    log(f"Mounting {model_name} → {mount_path}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"hf-mount failed for {model_name}: {r.stderr.strip()}")

    # Wait for config.json to appear (mount is async)
    import time as _time
    for _ in range(30):
        if os.path.exists(os.path.join(mount_path, "config.json")):
            log(f"Mounted {model_name} → {mount_path}")
            return mount_path
        _time.sleep(1)

    raise RuntimeError(f"hf-mount timed out waiting for {mount_path}/config.json")


def hf_unmount(mount_path):
    """Unmount an hf-mount mount point."""
    try:
        subprocess.run(["hf-mount", "stop", mount_path],
                       capture_output=True, text=True, timeout=30)
        log(f"Unmounted {mount_path}")
    except Exception as e:
        log(f"Warning: unmount failed for {mount_path}: {e}")


def count_lines(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'rb') as f:
        return sum(1 for _ in f)


def fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def kill_all_active(active, progress_bars):
    """Wait for active eval.py processes to finish cleanup, then force-kill stragglers.

    The terminal already sent SIGINT to the whole process group, so eval.py
    children are already handling KeyboardInterrupt (saving timing, shutting
    down vLLM).  We just wait for them instead of sending a redundant SIGINT
    that would interrupt their finally blocks.
    """
    # Close progress bars first so output is clean
    for bar in progress_bars.values():
        bar.close()

    if not active:
        return

    # Collect the direct eval.py processes (not their children — eval.py
    # handles its own vLLM shutdown in its finally block)
    eval_procs = []
    names = []
    for proc, (model_name, gpu_ids, port, log_file, log_path) in active.items():
        log_file.close()
        names.append(short_name(model_name))
        try:
            eval_procs.append(psutil.Process(proc.pid))
        except psutil.NoSuchProcess:
            pass

    # Wait for eval.py to save timing and shut down vLLM (up to 30s)
    log(f"Waiting for {len(eval_procs)} process(es) to finish cleanup...")
    _, alive = psutil.wait_procs(eval_procs, timeout=30)

    # SIGKILL stragglers and their entire process trees
    for p in alive:
        try:
            for child in p.children(recursive=True):
                child.kill()
            p.kill()
        except (psutil.NoSuchProcess, OSError):
            pass
    psutil.wait_procs(alive, timeout=5)

    log(f"Stopped {len(names)} model(s): {', '.join(names)}")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    tqdm.write(f"  [{ts}]  {msg}")


# ANSI helpers
BOLD = "\033[1m"
DIM = "\033[2m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"
RESET = "\033[0m"

STATUS_LABELS = {
    STATUS_LOADING:    f"{YELLOW}loading{RESET}",
    STATUS_READY:      f"{GREEN}ready{RESET}  ",
    STATUS_GENERATING: f"{CYAN}running{RESET}",
}


def make_bar_desc(name, gpu_label, status):
    label = STATUS_LABELS.get(status, status)
    return f"{BOLD}{name:<32}{RESET} {DIM}{gpu_label:<10}{RESET} {label}"


def print_header(all_gpus, queue, total_samples, gpu_info=None):
    print()
    print("  " + "=" * 62)
    print("   LongShOT Parallel Scheduler")
    print("  " + "=" * 62)
    # Show hostname and GPU hardware
    print(f"   Host:     {socket.gethostname()}")
    if gpu_info and gpu_info.get("driver_version"):
        runtime = gpu_info.get("runtime", "cuda").upper()
        runtime_ver = gpu_info.get("runtime_version", "?")
        print(f"   Driver:   {gpu_info['driver_version']}  {runtime} {runtime_ver}")
    # Show GPU model name (assume homogeneous, show first)
    if gpu_info and gpu_info["gpus"]:
        first_gpu = next(iter(gpu_info["gpus"].values()))
        mem = first_gpu.get("memory_total_gib", "?")
        print(f"   GPU type: {first_gpu.get('name', '?')} ({mem} GiB)")
    gpu_free = get_gpu_free_memory()
    gpu_labels = []
    for g in all_gpus:
        free = gpu_free.get(g)
        gpu_labels.append(f"{g}({free:.0f}G)" if free is not None else str(g))
    print(f"   GPUs:     {', '.join(gpu_labels)} ({len(all_gpus)} available)")
    print(f"   Dataset:  {total_samples} samples")
    print(f"   Queue:    {len(queue)} model(s)")
    print("  " + "-" * 62)
    for name, ngpus, omni, backend, max_frames, audio_only, alias, mount in queue:
        tags = "".join(f" [{t}]" for t in (["omni"] if omni else []) + (["audio"] if audio_only else []) + ([backend] if backend != "vllm" else []) + ([f"f{max_frames}"] if max_frames else []) + (["mount"] if mount else []) + ([f"@{alias}"] if alias else []))
        print(f"   {short_name(name):<40} {ngpus} GPU(s){tags}")
    print("  " + "=" * 62)
    print()


def print_summary(completed, failed, start_time):
    elapsed = fmt_duration(time.time() - start_time)
    print()
    print("  " + "=" * 62)
    print(f"   Scheduler Complete  ({elapsed} elapsed)")
    print("  " + "=" * 62)
    if completed:
        print(f"   Completed ({len(completed)}):")
        for m in completed:
            print(f"     {m}")
    if failed:
        print(f"   Failed ({len(failed)}):")
        for m in failed:
            name = m[0] if isinstance(m, tuple) else m
            print(f"     {name}")
    if not completed and not failed:
        print("   No models were processed.")
    print("  " + "=" * 62)
    print()


def run_scheduler(args):
    scheduler_start = time.time()
    all_gpus = get_available_gpus()
    gpu_info = collect_gpu_info()

    # Parse model configs: "name:gpus[:omni][:hf][:api][:audio][:mount][:f{N}][@alias]"
    # Flags (omni, hf, api, audio, mount, f{N}) can appear in any order after gpus.
    # The :api flag marks cloud API models (OpenRouter etc.) — no GPU allocation needed.
    # Optional @alias at the end overrides the output directory name.
    queue = []  # list of (model_name, num_gpus, omni, backend, max_frames, audio_only, alias, mount)
    for model_spec in args.models:
        # Extract @alias suffix if present
        alias = None
        if "@" in model_spec:
            model_spec, alias = model_spec.rsplit("@", 1)
        parts = model_spec.split(":")
        flags = set()
        max_frames = 0
        while len(parts) > 2 and (parts[-1] in ("omni", "hf", "api", "audio", "mount") or parts[-1].startswith("f")):
            flag = parts.pop()
            if flag.startswith("f") and flag[1:].isdigit():
                max_frames = int(flag[1:])
            else:
                flags.add(flag)
        if len(parts) >= 2:
            try:
                backend = "api" if "api" in flags else ("hf" if "hf" in flags else "vllm")
                num_gpus = int(parts[1])
                if backend == "api":
                    num_gpus = 0
                queue.append((parts[0], num_gpus, "omni" in flags, backend, max_frames, "audio" in flags, alias, "mount" in flags))
            except ValueError:
                log(f"Skipping invalid spec: '{model_spec}'")
        else:
            log(f"Skipping invalid spec: '{model_spec}'")

    # Get total samples
    total_samples = 0
    cache_file = f".dataset_cache/{args.tasks}_test.jsonl"
    if os.path.exists(cache_file):
        total_samples = count_lines(cache_file)

    print_header(all_gpus, queue, total_samples, gpu_info)

    # Filter out GPUs that don't have enough free memory
    min_mem = args.min_gpu_memory
    free_gpus = filter_gpus_by_memory(list(all_gpus), min_free_gib=min_mem)
    if len(free_gpus) < len(all_gpus):
        log(f"Using {len(free_gpus)}/{len(all_gpus)} GPUs after memory check: [{','.join(str(g) for g in free_gpus)}]")

    active = {}          # proc -> (model_name, gpu_ids, port, log_file, log_path)
    mount_paths = {}     # proc -> mount_path (for hf-mount cleanup)
    progress_bars = {}   # proc -> tqdm bar
    output_files = {}    # proc -> jsonl path
    server_status = {}   # proc -> (status, initial_count)
    next_port = BASE_PORT
    pending = list(queue)
    completed = []
    failed = []
    bar_position = 0
    stagger = not args.no_stagger

    if stagger and len(queue) > 1:
        log("Staggered launch enabled — each model waits for the previous server to be ready")

    try:
        while pending or active:
            # When staggering, only launch if no model is still loading
            any_loading = stagger and any(
                s[0] == STATUS_LOADING for s in server_status.values()
            )

            # Launch models that fit
            launched = []
            for i, (model_name, ngpus, omni, backend, max_frames, audio_only, alias, mount) in enumerate(pending):
                if any_loading and ngpus > 0:
                    break  # wait for current model to finish loading first (API models skip)
                # Re-check memory for GPUs that just became free (returned from finished models)
                if ngpus > 0:
                    free_gpus = filter_gpus_by_memory(free_gpus, min_free_gib=min_mem)
                if len(free_gpus) >= ngpus:
                    allocated = free_gpus[:ngpus]
                    free_gpus = free_gpus[ngpus:]
                    port = str(next_port)
                    next_port += 1

                    gpu_str = ",".join(str(g) for g in allocated)
                    is_api = (backend == "api")
                    tags = "".join(f" [{t}]" for t in (["omni"] if omni else []) + (["audio"] if audio_only else []) + ([backend] if backend != "vllm" else []) + ([f"f{max_frames}"] if max_frames else []) + (["mount"] if mount else []))
                    if is_api:
                        log(f"Starting {short_name(model_name)}{tags}  (cloud API, no GPUs)")
                    else:
                        log(f"Starting {short_name(model_name)}{tags}  GPUs=[{gpu_str}]  port={port}")

                    # Mount model via hf-mount if requested
                    mount_path = None
                    if mount:
                        try:
                            mount_path = hf_mount(model_name)
                        except Exception as e:
                            log(f"Mount failed for {model_name}: {e}")
                            free_gpus.extend(allocated)
                            failed.append((model_name, ngpus))
                            launched.append(i)
                            continue

                    run_id = uuid.uuid4().hex[:12]

                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = gpu_str
                    env["LONGSHOT_RUN_ID"] = run_id
                    env.setdefault("VLLM_MEDIA_LOADING_THREAD_COUNT", "32")

                    # Divide workers across concurrent models (including this one about to launch)
                    concurrent_models = len(active) + 1
                    if is_api:
                        api_cfg = {}
                        try:
                            import yaml as _yaml
                            with open(args.config_file) as _f:
                                _cfg = _yaml.safe_load(_f)
                                api_cfg = _cfg.get("openrouter", {})
                        except Exception:
                            pass
                        num_workers = api_cfg.get("max_connections", 8)
                    else:
                        num_workers = min(128, max(4, args.num_workers // concurrent_models))

                    cmd = [
                        sys.executable, "eval.py",
                        "--model", model_name,
                        "--tasks", args.tasks,
                        "--num_workers", str(num_workers),
                        "--generate", "true",
                        "--output_dir", args.output_dir,
                        "--tensor_parallel_size", str(ngpus),
                        "--config-file", args.config_file,
                        "--port", port,
                    ]
                    if is_api:
                        cmd.append("--external-server")
                    if omni:
                        cmd.append("--omni")
                    if audio_only:
                        cmd.append("--audio-only")
                    if backend != "vllm":
                        cmd.extend(["--backend", backend])
                    if max_frames:
                        cmd.extend(["--max-frames", str(max_frames)])
                    if alias:
                        cmd.extend(["--alias", alias])
                    if mount_path:
                        cmd.extend(["--model-path", mount_path])

                    # Use alias for output paths if provided
                    output_name = alias if alias else to_underscored(model_name)

                    # Write run_info.json with GPU hardware and config metadata
                    write_run_info(args.output_dir, model_name, allocated, port, gpu_info, args, run_id)

                    os.makedirs("logs", exist_ok=True)
                    model_safe = output_name.replace("/", "_").replace("-", "_")
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_path = os.path.join("logs", f"scheduler_{model_safe}_{ts_str}.log")
                    log_file = open(log_path, "w")

                    proc = subprocess.Popen(
                        cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT
                    )
                    active[proc] = (model_name, allocated, port, log_file, log_path)
                    if mount_path:
                        mount_paths[proc] = mount_path
                    launched.append(i)

                    # Progress bar
                    out_jsonl = os.path.join(
                        args.output_dir, output_name,
                        f"{output_name}.jsonl"
                    )
                    output_files[proc] = out_jsonl
                    initial = count_lines(out_jsonl)

                    sname = short_name(model_name)
                    desc_base = f"  {sname}"
                    gpu_label = "API" if is_api else f"GPU {gpu_str}"
                    initial_status = STATUS_GENERATING if is_api else STATUS_LOADING
                    bar = tqdm(
                        total=total_samples or None,
                        initial=initial,
                        desc=make_bar_desc(desc_base, gpu_label, initial_status),
                        unit=" samples",
                        position=bar_position,
                        leave=True,
                        bar_format="{desc}  {percentage:3.0f}% {bar:30} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        mininterval=1,
                    )
                    progress_bars[proc] = bar
                    server_status[proc] = (initial_status, initial, desc_base, gpu_label, int(port) if not is_api else 0)
                    bar_position += 1

                    # In stagger mode, launch one at a time
                    if stagger:
                        break

            for i in sorted(launched, reverse=True):
                pending.pop(i)

            if not active:
                if pending:
                    log(f"Cannot fit {short_name(pending[0][0])} ({pending[0][1]} GPUs). {len(free_gpus)} free. Skipping.")
                    failed.append(pending.pop(0))
                continue

            time.sleep(POLL_INTERVAL)

            # Update bars and check completions
            finished = []
            for proc in list(active.keys()):
                model_name, gpu_ids, port, log_file, log_path = active[proc]

                # Update progress and server status
                if proc in progress_bars and proc in output_files:
                    current = count_lines(output_files[proc])
                    bar = progress_bars[proc]
                    status, initial, desc_base, gpu_label, srv_port = server_status[proc]

                    # Check server readiness transition
                    if status == STATUS_LOADING:
                        try:
                            r = requests.get(f"http://localhost:{srv_port}/v1/models", timeout=1)
                            if r.status_code == 200:
                                status = STATUS_READY
                                bar.desc = make_bar_desc(desc_base, gpu_label, STATUS_READY)
                                log(f"Server ready: {short_name(model_name)} (port {srv_port})")
                        except (requests.ConnectionError, requests.Timeout):
                            pass

                    if status == STATUS_READY and current > initial:
                        status = STATUS_GENERATING
                        bar.desc = make_bar_desc(desc_base, gpu_label, STATUS_GENERATING)

                    server_status[proc] = (status, initial, desc_base, gpu_label, srv_port)

                    if current != bar.n:
                        bar.n = current
                        bar.refresh()

                # Check exit
                ret = proc.poll()
                if ret is not None:
                    finished.append(proc)
                    log_file.close()
                    gpu_str = ",".join(str(g) for g in gpu_ids)

                    # Final bar update
                    if proc in progress_bars:
                        bar = progress_bars[proc]
                        bar.n = count_lines(output_files.get(proc, ""))
                        bar.refresh()
                        bar.close()
                        del progress_bars[proc]
                    output_files.pop(proc, None)
                    server_status.pop(proc, None)

                    # Unmount hf-mount if this model was mounted
                    if proc in mount_paths:
                        hf_unmount(mount_paths.pop(proc))

                    if ret == 0:
                        log(f"Completed {short_name(model_name)}  GPUs=[{gpu_str}] freed")
                        completed.append(model_name)
                    else:
                        log(f"Failed {short_name(model_name)}  exit={ret}  log={log_path}")
                        failed.append((model_name, len(gpu_ids)))

                    free_gpus.extend(gpu_ids)

            for proc in finished:
                del active[proc]

    except KeyboardInterrupt:
        print()
        log("Interrupted - stopping all models...")
        kill_all_active(active, progress_bars)
        for mp in mount_paths.values():
            hf_unmount(mp)
        mount_paths.clear()
        active.clear()
        progress_bars.clear()
        print_summary(completed, failed, scheduler_start)
        sys.exit(1)

    # Clean close
    for bar in progress_bars.values():
        bar.close()

    print_summary(completed, failed, scheduler_start)


def main():
    args = parse_args()
    run_scheduler(args)


if __name__ == "__main__":
    main()
