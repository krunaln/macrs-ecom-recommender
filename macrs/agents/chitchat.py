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
            "You are the Chit-Chat Agent in an e-commerce assistant. "
            "Keep the tone warm and concise while encouraging preference signals. "
            f"User message: {user_message}\n"
            f"Known preferences: {state.preferences}\n"
            "Return 1 candidate."
        )
        return generate_structured_output(prompt, AgentLLMOutput)
