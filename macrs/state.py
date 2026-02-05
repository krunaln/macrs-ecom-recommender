from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConversationState(BaseModel):
    session_id: str
    turn_id: int = 0
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    browsing_history: List[str] = Field(default_factory=list)
    negative_constraints: Dict[str, Any] = Field(default_factory=dict)
    act_history: List[str] = Field(default_factory=list)
    last_user_message: Optional[str] = None
    last_system_response: Optional[str] = None
    dialogue_history: List[Dict[str, Any]] = Field(default_factory=list)
    agent_suggestions: Dict[str, List[str]] = Field(default_factory=dict)
    corrective_experiences: List[str] = Field(default_factory=list)
    last_recommendation_failure_turn: Optional[int] = None
    last_recommendation_turn: Optional[int] = None

    def record_act(self, act: str) -> None:
        self.act_history.append(act)
        if len(self.act_history) > 50:
            self.act_history = self.act_history[-50:]

    def append_dialogue(self, user_message: str, system_response: str, act: Optional[str] = None) -> None:
        entry = {"user": user_message, "system": system_response}
        if act:
            entry["act"] = act
        self.dialogue_history.append(entry)
        if len(self.dialogue_history) > 50:
            self.dialogue_history = self.dialogue_history[-50:]
