from __future__ import annotations

from typing import Any, Dict, List

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
                products = []
                for product in cand.products[:5]:
                    products.append(
                        {
                            "id": product.id,
                            "title": product.title,
                            "brand": product.brand,
                            "price": product.price,
                            "currency": product.currency,
                            "categories": product.categories,
                            "description": (product.description or "")[:300],
                        }
                    )
                candidates.append(
                    {
                        "candidate_id": cand.candidate_id,
                        "act": output.act,
                        "agent_name": output.agent_name,
                        "response": cand.response,
                        "slots": cand.slots,
                        "products": products,
                    }
                )

        sufficient = self._has_sufficient_preferences(state.user_profile)
        prompt = (
            "You are the Planner Agent for a multi-agent conversational recommender.\n"
            "Goal: select exactly ONE candidate response (ask / recommend / chitchat) that best advances the dialogue.\n"
            "Do NOT rewrite any response. Choose by candidate_id only.\n\n"
            "Multi-step reasoning:\n"
            "1) Review act history and avoid repeating the same act across multiple turns.\n"
            "2) Assess preference sufficiency from user_profile + dialogue_history.\n"
            "   - If sufficient, recommendation is appropriate and should be selected if a recommendation candidate exists.\n"
            "   - If insufficient, prefer responses that increase information gain.\n"
            "3) Consider engagement: choose a response that keeps the conversation natural and forward-moving.\n"
            "4) Use corrective_experiences to avoid prior mistakes.\n\n"
            f"User profile: {state.user_profile}\n"
            f"Browsing history: {state.browsing_history}\n"
            f"Dialogue history: {state.dialogue_history[-5:]}\n"
            f"Act history: {state.act_history}\n"
            f"Corrective experiences: {state.corrective_experiences}\n"
            f"Preference sufficiency: {sufficient}\n"
            f"Candidates: {candidates}\n"
            "Return selected_act and selected_candidate_id."
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

        if sufficient and selected_output.act != "recommend":
            fallback = self._first_recommend_candidate(outputs)
            if fallback:
                selected_output, selected = fallback

        update = StrategyUpdate(weight_updates={}, notes=llm_output.notes)
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

    def _has_sufficient_preferences(self, profile: Dict[str, Any]) -> bool:
        if not profile:
            return False
        key_match = any(k in profile for k in ["category", "type", "product", "item"])
        return key_match and len(profile.keys()) >= 2

    def _first_recommend_candidate(self, outputs: List[AgentOutput]):
        for output in outputs:
            if output.act != "recommend":
                continue
            for cand in output.candidates:
                if cand.products:
                    return output, cand
        return None
