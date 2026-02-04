from __future__ import annotations

from typing import Dict

from macrs.models import ReflectionUpdate
from macrs.state import ConversationState


class ReflectionEngine:
    def reflect(self, state: ConversationState, user_message: str) -> ReflectionUpdate:
        text = (user_message or "").lower()
        feedback: Dict[str, float] = {}
        notes = []

        if any(word in text for word in ["thanks", "great", "love", "perfect"]):
            feedback["recommend"] = 0.05
            notes.append("Positive signal detected.")
        if any(word in text for word in ["not", "no", "don't", "hate"]):
            feedback["ask"] = 0.03
            feedback["recommend"] = -0.04
            notes.append("Negative or rejection signal detected.")

        return ReflectionUpdate(
            inferred_feedback={"text": user_message},
            weight_deltas=feedback,
            preference_updates={},
            notes=" ".join(notes) if notes else None,
        )
