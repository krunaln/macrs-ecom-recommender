from __future__ import annotations

from typing import Dict, List

from macrs.llm import generate_structured_output

from macrs.models import AgentOutput, PlannerDecision, PlannerLLMOutput, StrategyUpdate
from macrs.state import ConversationState


class Planner:
    def select(self, outputs: List[AgentOutput], state: ConversationState) -> PlannerDecision:
        if not outputs:
            raise RuntimeError("Planner received no candidates")

        candidates = []
        for output in outputs:
            for cand in output.candidates:
                candidates.append(
                    {
                        "candidate_id": cand.candidate_id,
                        "act": output.act,
                        "agent_name": output.agent_name,
                        "confidence": output.confidence,
                        "score": cand.score,
                        "rationale": cand.rationale,
                    }
                )

        prompt = (
            "You are the Planner Agent. Choose exactly one candidate based on confidence, "
            "conversation state, and strategy weights. "
            "Do not rewrite any response. "
            f"Strategy weights: {state.strategy_weights}\n"
            f"Act history: {state.act_history}\n"
            f"Candidates: {candidates}\n"
            "Return the selected_act, selected_candidate_id, and optional weight_updates."
        )
        llm_output = generate_structured_output(prompt, PlannerLLMOutput)
        if not llm_output:
            raise RuntimeError("Planner LLM failed to return valid output")

        selected = None
        selected_output = None
        for output in outputs:
            for cand in output.candidates:
                if cand.candidate_id == llm_output.selected_candidate_id:
                    selected = cand
                    selected_output = output
                    break
            if selected:
                break
        if not selected or not selected_output:
            raise RuntimeError("Planner selected unknown candidate_id")

        update = StrategyUpdate(weight_updates=llm_output.weight_updates, notes=llm_output.notes)
        decision = PlannerDecision(
            selected_act=selected_output.act,
            selected_candidate_id=selected.candidate_id,
            selected_response=selected.response,
            strategy_update=update,
            metadata={
                "agent_name": selected_output.agent_name,
                "candidate_score": selected.score,
            },
        )
        return decision
