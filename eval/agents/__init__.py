"""
Agents framework for benchmarking agentic video understanding systems.

Usage:
    from agents import get_agent, list_agents

    agent = get_agent("video_explorer", config)
    agent.start_servers()
    response = agent.process_sample(video_path, question)
    agent.stop_servers()
"""

from agents.base import BaseAgent, AgentResponse
from agents.registry import get_agent, list_agents, register_agent
from agents.config import load_agent_config
from agents.runner import run_agent_inference

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "get_agent",
    "list_agents",
    "register_agent",
    "load_agent_config",
    "run_agent_inference",
]
