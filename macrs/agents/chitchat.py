from __future__ import annotations

from macrs.agents.base import BaseAgent
from macrs.llm import generate_structured_output
from macrs.models import AgentCandidate, AgentLLMOutput, AgentOutput
from macrs.state import ConversationState


class ChitChatAgent(BaseAgent):
    name = "chitchat"

    def run(self, user_message: str, state: ConversationState) -> AgentOutput:
        llm_output = self._llm_generate(user_message, state)
        if llm_output:
            return AgentOutput(
                agent_name=self.name,
                act="chitchat",
                confidence=llm_output.confidence,
                candidates=llm_output.candidates,
                metadata={"source": "llm"},
            )

        response = (
            "Happy to help. If you tell me a bit more about what you like, "
            "I can narrow it down quickly."
        )
        candidate = AgentCandidate(
            candidate_id="chitchat_default",
            response=response,
            score=0.3,
            rationale="Maintains engagement and invites preference signals.",
        )
        return AgentOutput(
            agent_name=self.name,
            act="chitchat",
            confidence=0.4,
            candidates=[candidate],
            metadata={},
        )

    def _llm_generate(self, user_message: str, state: ConversationState) -> AgentLLMOutput | None:
        prompt = (
            "You are the Chit-Chat Agent in an e-commerce assistant.\n"
            "Goal: keep the conversation light and engaging to elicit preferences.\n"
            "Constraints:\n"
            "- Do NOT recommend products.\n"
            "- Do NOT list items or mention specific products.\n"
            "- Do NOT ask direct clarification questions (leave that to the Asking Agent).\n"
            "- You may express admiration for certain item attributes to guide preferences.\n"
            "Use dialogue history and user profile to avoid repetition.\n\n"
            f"Dialogue history: {state.dialogue_history[-5:]}\n"
            f"User message: {user_message}\n"
            f"Known preferences: {state.user_profile}\n"
            f"Browsing history: {state.browsing_history}\n"
            f"Strategy suggestions: {state.agent_suggestions.get('chitchat', [])}\n"
            "Return 1 candidate."
        )
        return generate_structured_output(prompt, AgentLLMOutput)
