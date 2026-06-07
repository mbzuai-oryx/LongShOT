"""Agent registration and discovery."""

from typing import Dict, List, Type, Any
from agents.base import BaseAgent

AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {}


def register_agent(name: str):
    """Decorator to register an agent class."""
    def decorator(cls: Type[BaseAgent]):
        AGENT_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def get_agent(name: str, config: Dict[str, Any]) -> BaseAgent:
    """Instantiate and setup an agent by name."""
    if name not in AGENT_REGISTRY:
        available = ", ".join(AGENT_REGISTRY.keys()) or "none"
        raise ValueError(f"Unknown agent: {name}. Available: {available}")

    agent_cls = AGENT_REGISTRY[name]
    agent = agent_cls()
    agent.setup(config)
    return agent


def list_agents() -> List[str]:
    """List all registered agent names."""
    return list(AGENT_REGISTRY.keys())


def _auto_discover_agents():
    """Import agent modules to trigger registration."""
    import importlib
    import pkgutil
    from pathlib import Path

    agents_dir = Path(__file__).parent
    for _, module_name, is_pkg in pkgutil.iter_modules([str(agents_dir)]):
        if is_pkg and module_name not in ("repos",):
            try:
                importlib.import_module(f"agents.{module_name}")
            except ImportError:
                pass


_auto_discover_agents()
