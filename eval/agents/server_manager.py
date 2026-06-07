"""Multi-server lifecycle management for agentic systems."""

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


@dataclass
class ServerSpec:
    """Specification for a single inference server."""
    name: str
    model_path: str
    gpu_id: int
    port: int
    tensor_parallel: int = 1
    max_model_len: Optional[int] = None
    extra_args: Dict[str, Any] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return f"http://localhost:{self.port}"


class AgentServerManager:
    """Manages multiple inference servers for an agent."""

    def __init__(self, specs: List[ServerSpec], startup_timeout: int = 300):
        self.specs = specs
        self.startup_timeout = startup_timeout
        self.processes: Dict[str, subprocess.Popen] = {}
        self.endpoints: Dict[str, str] = {}

    def _kill_process_on_port(self, port: int) -> None:
        """Kill any process listening on the specified port."""
        try:
            result = subprocess.run(
                ["lsof", "-t", "-i", f":{port}", "-s", "TCP:LISTEN"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split('\n'):
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(0.5)
                        os.kill(int(pid), signal.SIGKILL)
                    except (ProcessLookupError, ValueError):
                        pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def start_all(self) -> Dict[str, str]:
        """Start servers one at a time, waiting for each to be healthy before
        launching the next.

        Rationale: spawning several vLLM processes back-to-back causes their
        CUDA init probes (cudaGetDeviceCount / pynvml queries / cudaIpc setup)
        to overlap, which on some driver versions poisons the in-RAM driver
        state for unrelated GPUs and other live CUDA processes on the host.
        Staggering ensures each server has fully finished its CUDA setup
        before the next one begins.
        """
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

        for spec in self.specs:
            self._kill_process_on_port(spec.port)
            self._start_server(spec)

            if not self._wait_for_server(spec):
                log_path = os.path.join(log_dir, f"vllm_{spec.name}_{spec.port}.log")
                print(f"[ServerManager] Server {spec.name} failed. Check log: {log_path}")
                self.stop_all()
                raise RuntimeError(f"Server {spec.name} failed to start within {self.startup_timeout}s")

            print(f"[ServerManager] {spec.name} healthy on {spec.endpoint}")

        return self.endpoints

    def _start_server(self, spec: ServerSpec) -> None:
        """Start a single vLLM server."""
        env = os.environ.copy()

        # Map relative GPU index to physical GPU based on parent's CUDA_VISIBLE_DEVICES
        parent_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if parent_gpus:
            gpu_list = parent_gpus.split(",")
            if spec.gpu_id < len(gpu_list):
                physical_gpu = gpu_list[spec.gpu_id]
            else:
                physical_gpu = str(spec.gpu_id)
        else:
            physical_gpu = str(spec.gpu_id)

        env["CUDA_VISIBLE_DEVICES"] = physical_gpu
        env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        print(f"[ServerManager] Starting {spec.name} on physical GPU {physical_gpu} (index {spec.gpu_id}), port {spec.port}")

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", spec.model_path,
            "--port", str(spec.port),
            "--tensor-parallel-size", str(spec.tensor_parallel),
            "--trust-remote-code",
        ]

        if spec.max_model_len:
            cmd.extend(["--max-model-len", str(spec.max_model_len)])

        for key, value in spec.extra_args.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"vllm_{spec.name}_{spec.port}.log")
        log_file = open(log_path, "w")

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        self.processes[spec.name] = process
        self.endpoints[spec.name] = spec.endpoint

    def _wait_for_server(self, spec: ServerSpec, poll_interval: float = 2.0) -> bool:
        """Wait for a server to become healthy."""
        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:
            process = self.processes.get(spec.name)
            if process and process.poll() is not None:
                return False

            try:
                response = requests.get(f"{spec.endpoint}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass

            time.sleep(poll_interval)

        return False

    def stop_all(self) -> None:
        """Terminate all server processes."""
        for name, process in self.processes.items():
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

        self.processes.clear()
        self.endpoints.clear()

    def health_check(self) -> Dict[str, bool]:
        """Check health of each server."""
        status = {}
        for spec in self.specs:
            try:
                response = requests.get(f"{spec.endpoint}/health", timeout=5)
                status[spec.name] = response.status_code == 200
            except requests.RequestException:
                status[spec.name] = False
        return status

    def __enter__(self):
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()
        return False
