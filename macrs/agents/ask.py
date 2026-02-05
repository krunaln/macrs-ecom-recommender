from __future__ import annotations

from typing import List

from macrs.agents.base import BaseAgent
from macrs.llm import generate_structured_output
from macrs.models import AgentCandidate, AgentLLMOutput, AgentOutput
from macrs.state import ConversationState


class AskingAgent(BaseAgent):
    name = "asking"

    def run(self, user_message: str, state: ConversationState) -> AgentOutput:
        llm_output = self._llm_generate(user_message, state)
        if llm_output:
            return AgentOutput(
                agent_name=self.name,
                act="ask",
                confidence=llm_output.confidence,
                candidates=llm_output.candidates,
                metadata={"source": "llm"},
            )

        prompts: List[AgentCandidate] = []
        preferences = state.user_profile

        if not preferences.get("category"):
            prompts.append(
                AgentCandidate(
                    candidate_id="ask_category",
                    response="What kind of product or category are you looking for?",
                    score=0.6,
                    rationale="Category is missing; increases retrieval precision.",
                    slots={"missing": "category"},
                )
            )
        if not preferences.get("price_max"):
            prompts.append(
                AgentCandidate(
                    candidate_id="ask_budget",
                    response="Do you have a budget range in mind?",
                    score=0.55,
                    rationale="Budget helps filter candidates.",
                    slots={"missing": "price_max"},
                )
            )
        if not preferences.get("brand"):
            prompts.append(
                AgentCandidate(
                    candidate_id="ask_brand",
                    response="Any preferred brand?",
                    score=0.5,
                    rationale="Brand preference can raise relevance.",
                    slots={"missing": "brand"},
                )
            )

        if not prompts:
            prompts.append(
                AgentCandidate(
                    candidate_id="ask_refine",
                    response="Anything specific you want to prioritize (price, brand, or features)?",
                    score=0.4,
                    rationale="No obvious missing slots; offer refinement.",
                )
            )

        return AgentOutput(
            agent_name=self.name,
            act="ask",
            confidence=min(1.0, 0.4 + 0.1 * len(prompts)),
            candidates=prompts,
            metadata={"missing_slots": [c.slots.get("missing") for c in prompts if c.slots.get("missing")]},
        )

    def _llm_generate(self, user_message: str, state: ConversationState) -> AgentLLMOutput | None:
        prompt = (
            "You are the Asking Agent in an e-commerce assistant.\n"
            "Goal: elicit missing preferences by asking concise questions.\n"
            "Constraints: do NOT recommend items. Avoid repeating questions already asked.\n\n"
            f"Dialogue history: {state.dialogue_history[-5:]}\n"
            f"User message: {user_message}\n"
            f"Known preferences: {state.user_profile}\n"
            f"Browsing history: {state.browsing_history}\n"
            f"Strategy suggestions: {state.agent_suggestions.get('ask', [])}\n"
            "Return 1-3 candidates."
        )
        return generate_structured_output(prompt, AgentLLMOutput)
