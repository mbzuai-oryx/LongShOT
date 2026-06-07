"""Base agent interface for video understanding agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentResponse:
    """Structured response from an agent."""
    answer: str
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for video understanding agents."""

    name: str = "base"
    repo_url: Optional[str] = None

    @abstractmethod
    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize agent: clone repo if needed, load models, etc."""
        pass

    @abstractmethod
    def start_servers(self) -> None:
        """Start required inference servers."""
        pass

    @abstractmethod
    def stop_servers(self) -> None:
        """Clean shutdown of all servers."""
        pass

    @abstractmethod
    def process_sample(
        self,
        video_path: str,
        question: str,
        subtitles: Optional[str] = None,
    ) -> AgentResponse:
        """Process a single video QA sample."""
        pass

    def adapt_response_for_eval(self, response: AgentResponse) -> str:
        """Convert agent response to string for LLM-based evaluation."""
        return response.answer

    def get_server_endpoints(self) -> Dict[str, str]:
        """Return mapping of server name to endpoint URL."""
        return {}

    def needs_preprocessing(self) -> bool:
        """Return True if agent requires video preprocessing."""
        return False

    @property
    def recommended_concurrency(self) -> Optional[int]:
        """How many samples this agent can usefully process in parallel.

        Override in subclasses that pool inference resources (e.g. multi-GPU
        VLM copies). Returning None defers to the eval-harness default.
        """
        return None

    def preprocess(self, video_dir: str, output_dir: str, **kwargs) -> None:
        """Preprocess videos for this agent. Override in subclass if needed."""
        pass
