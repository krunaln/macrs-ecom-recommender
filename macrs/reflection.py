from __future__ import annotations

from typing import Dict, List, Optional

from macrs.llm import generate_structured_output
from macrs.models import FailureDetectionOutput, InfoReflectionOutput, ReflectionUpdate, StrategyReflectionOutput
from macrs.state import ConversationState


class ReflectionEngine:
    def reflect(self, state: ConversationState, user_feedback: str) -> ReflectionUpdate:
        info = self._info_reflect(state, user_feedback)
        strategy = None
        if self._should_strategy_reflect(state, user_feedback):
            strategy = self._strategy_reflect(state, user_feedback)
        else:
            strategy = None

        if info:
            if info.current_demand:
                state.user_profile.update(info.current_demand)
            if info.browsing_history:
                self._merge_history(state, info.browsing_history)

        if strategy:
            if strategy.suggestions:
                state.agent_suggestions = self._normalize_suggestions(strategy.suggestions)
            if strategy.corrective_experiences:
                state.corrective_experiences.extend(strategy.corrective_experiences)
                state.corrective_experiences = state.corrective_experiences[-20:]
            state.last_recommendation_failure_turn = state.turn_id

        return ReflectionUpdate(
            inferred_feedback={"text": user_feedback},
            weight_deltas={},
            preference_updates=info.current_demand if info else {},
            notes=strategy.error_summary if strategy else None,
        )

    def _info_reflect(self, state: ConversationState, user_feedback: str) -> Optional[InfoReflectionOutput]:
        prompt = (
            "Please infer user preferences based on the conversation and combine them with past preferences. "
            "Return current_demand as key-value pairs (e.g., category, brand, price_max, color, size). "
            "Also return browsing_history as a list of items or attributes mentioned by the user.\n\n"
            "Only use information that the user explicitly stated or confirmed. "
            "Do NOT introduce preferences from the assistant's own responses unless the user confirmed them.\n\n"
            f"Past preferences: {state.user_profile}\n"
            f"Dialogue history: {state.dialogue_history[-5:]}\n"
            f"User feedback: {user_feedback}\n"
        )
        return generate_structured_output(prompt, InfoReflectionOutput)

    def _strategy_reflect(self, state: ConversationState, user_feedback: str) -> Optional[StrategyReflectionOutput]:
        trajectory = self._build_trajectory(state)
        prompt = (
            "Based on your past action trajectory, explain why the recommendation failed as indicated by the user. "
            "Then generate suggestions for Asking Agent, Recommending Agent, and Chit-chatting Agent. "
            "Return suggestions as a dictionary with keys: ask, recommend, chitchat (each value is a list of short suggestions). "
            "Finally, summarize suggestions into corrective experiences for the Planning Agent as a list of short sentences.\n\n"
            f"Trajectory: {trajectory}\n"
            f"User feedback: {user_feedback}\n"
        )
        return generate_structured_output(prompt, StrategyReflectionOutput)

    def _build_trajectory(self, state: ConversationState) -> List[Dict[str, str]]:
        start_index = 0
        if state.last_recommendation_failure_turn is not None:
            start_index = max(0, state.last_recommendation_failure_turn)
        history = state.dialogue_history[start_index:]
        return [{"user": item["user"], "system": item["system"], "act": item.get("act")} for item in history]

    def _should_strategy_reflect(self, state: ConversationState, user_feedback: str) -> bool:
        if not state.act_history:
            return False
        if state.act_history[-1] != "recommend":
            return False
        verdict = self._detect_failure(state, user_feedback)
        return verdict.failed

    def _detect_failure(self, state: ConversationState, user_feedback: str) -> FailureDetectionOutput:
        prompt = (
            "Decide whether the user's feedback indicates the recommendation FAILED.\n"
            "Return failed=true if the user rejects the recommendation, asks for something different, "
            "or expresses dissatisfaction. Otherwise failed=false.\n\n"
            f"Last system response: {state.last_system_response}\n"
            f"User feedback: {user_feedback}\n"
        )
        return generate_structured_output(prompt, FailureDetectionOutput) or FailureDetectionOutput(failed=False)

    def _merge_history(self, state: ConversationState, items: List[str]) -> None:
        existing = set(state.browsing_history)
        for item in items:
            if item not in existing:
                state.browsing_history.append(item)
                existing.add(item)

    def _normalize_suggestions(self, suggestions: Dict[str, List[str]]) -> Dict[str, List[str]]:
        cleaned: Dict[str, List[str]] = {}
        for key in ["ask", "recommend", "chitchat"]:
            values = suggestions.get(key, [])
            cleaned[key] = [v.strip() for v in values if v and v.strip()]
        return cleaned
