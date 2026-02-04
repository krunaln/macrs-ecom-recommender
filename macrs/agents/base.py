from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from macrs.models import AgentOutput
from macrs.state import ConversationState


class BaseAgent(ABC):
    name: str

    @abstractmethod
    def run(self, user_message: str, state: ConversationState) -> AgentOutput:
        raise NotImplementedError

    def _meta(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return kwargs
