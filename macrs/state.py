from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


DEFAULT_STRATEGY_WEIGHTS = {
    "ask": 0.34,
    "recommend": 0.33,
    "chitchat": 0.33,
}


class ConversationState(BaseModel):
    session_id: str
    turn_id: int = 0
    preferences: Dict[str, Any] = Field(default_factory=dict)
    negative_constraints: Dict[str, Any] = Field(default_factory=dict)
    act_history: List[str] = Field(default_factory=list)
    strategy_weights: Dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_STRATEGY_WEIGHTS))
    last_user_message: Optional[str] = None

    def apply_weight_deltas(self, deltas: Dict[str, float]) -> None:
        for key, delta in deltas.items():
            current = self.strategy_weights.get(key, 0.0)
            self.strategy_weights[key] = max(0.0, current + delta)
        total = sum(self.strategy_weights.values()) or 1.0
        for key in list(self.strategy_weights.keys()):
            self.strategy_weights[key] = self.strategy_weights[key] / total

    def record_act(self, act: str) -> None:
        self.act_history.append(act)
        if len(self.act_history) > 50:
            self.act_history = self.act_history[-50:]
