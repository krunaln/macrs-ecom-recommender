from __future__ import annotations

import json
import logging
import time

from typing import Any, Dict, List

from macrs.agents.base import BaseAgent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from macrs.llm import generate_structured_output, get_llm
from macrs.models import AgentCandidate, AgentLLMOutput, AgentOutput, ProductCandidate
from macrs.state import ConversationState
from scripts.retrieve import search_products


@tool
def product_search(
    query: str,
    k: int = 5,
    price_min: float | None = None,
    price_max: float | None = None,
    currency: str | None = None,
    brand: str | None = None,
    category: str | None = None,
) -> list[dict]:
    """Search products using the external retrieval system."""
    return search_products(
        query=query,
        k=k,
        price_min=price_min,
        price_max=price_max,
        currency=currency,
        brand=brand,
        category=category,
    )


class RecommendingAgent(BaseAgent):
    name = "recommending"

    def run(self, user_message: str, state: ConversationState) -> AgentOutput:
        if not hasattr(self, "_last_products"):
            self._last_products = []
        preferences = state.user_profile
        query = self._build_query(user_message, preferences)

        results = self._retrieve_products(user_message, preferences, query, state)
        if not results and self._last_products:
            results = self._last_products
        elif not results and self._is_low_signal(user_message):
            results = self._last_products
        if results:
            self._last_products = results

        products: List[ProductCandidate] = []
        for item in results:
            categories = item.get("categories")
            if isinstance(categories, str):
                try:
                    categories = json.loads(categories)
                except json.JSONDecodeError:
                    categories = [c.strip() for c in categories.split(",") if c.strip()]
            products.append(
                ProductCandidate(
                    id=str(item["id"]),
                    title=item["title"],
                    brand=item.get("brand"),
                    description=item.get("description"),
                    categories=categories,
                    price=item.get("price"),
                    currency=item.get("currency"),
                    score=float(item.get("final_score", 0.0)),
                )
            )

        llm_output = self._llm_generate(user_message, preferences, products, state)
        if llm_output and llm_output.candidates:
            candidates = llm_output.candidates
            for idx, candidate in enumerate(candidates, start=1):
                if not candidate.candidate_id:
                    candidate.candidate_id = f"rec_{idx}"
                candidate.products = products
            confidence = llm_output.confidence
        else:
            response = self._format_response(products)
            candidates = [
                AgentCandidate(
                    candidate_id="rec_primary",
                    response=response,
                    score=max([p.score for p in products], default=0.0),
                    products=products,
                )
            ]
            confidence = min(1.0, 0.3 + 0.1 * len(products))

        return AgentOutput(
            agent_name=self.name,
            act="recommend",
            confidence=confidence,
            candidates=candidates,
            metadata={"query": query, "result_count": len(products)},
        )

    def _build_query(self, user_message: str, preferences: Dict[str, Any]) -> str:
        parts = [user_message]
        for key in ["category", "brand"]:
            value = preferences.get(key)
            if value:
                parts.append(str(value))
        return " ".join(parts).strip()

    def _is_low_signal(self, user_message: str) -> bool:
        text = (user_message or "").strip().lower()
        if not text or len(text) < 3:
            return True
        low = {
            "no",
            "nope",
            "not sure",
            "i dont know",
            "i don't know",
            "maybe",
            "ok",
            "okay",
            "yes",
            "sure",
        }
        return text in low

    def _retrieve_products(self, user_message: str, preferences: Dict[str, Any], query: str, state: ConversationState) -> list[dict]:
        llm = get_llm()
        tools = [product_search]
        llm_with_tools = llm.bind_tools(tools)

        system = SystemMessage(
            content=(
                "You are the Recommending Agent. Decide how to call the product_search tool. "
                "Use the user's message and known preferences. Return a tool call only."
            )
        )
        human = HumanMessage(
            content=(
                "You are the Recommending Agent. Use the product_search tool only.\n"
                f"User message: {user_message}\n"
                f"Known preferences: {preferences}\n"
                f"Browsing history: {state.browsing_history}\n"
                f"Strategy suggestions: {state.agent_suggestions.get('recommend', [])}\n"
                f"Suggested query: {query}\n"
                "Call product_search with the best parameters."
            )
        )
        try:
            response = llm_with_tools.invoke([system, human])
        except Exception as exc:
            logging.error("Tool selection LLM failed: %s", exc)
            raise RuntimeError("Recommending agent failed to call product_search tool") from exc

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            logging.error("No tool calls returned from LLM")
            raise RuntimeError("Recommending agent returned no tool calls")

        call = tool_calls[0]
        args = call.get("args", {})
        logger = logging.getLogger("macrs.tool.product_search")
        logger.info("call args=%s", args)
        start = time.perf_counter()
        results = product_search.invoke(args)
        elapsed = time.perf_counter() - start
        logger.info("returned %d results in %.2fs", len(results), elapsed)
        return results

    def _format_response(self, products: List[ProductCandidate]) -> str:
        if not products:
            return "I could not find good matches yet. Want to refine by brand or price range?"
        lines = ["Here are a few options you might like:"]
        for idx, product in enumerate(products, start=1):
            price = f"{product.currency or ''} {product.price:.2f}" if product.price is not None else "Price N/A"
            brand = f"{product.brand} - " if product.brand else ""
            lines.append(f"{idx}. {brand}{product.title} ({price})")
        return "\n".join(lines)

    def _llm_generate(
        self,
        user_message: str,
        preferences: Dict[str, Any],
        products: List[ProductCandidate],
        state: ConversationState,
    ) -> AgentLLMOutput | None:
        if not products:
            return None
        product_brief = [
            {
                "id": p.id,
                "title": p.title,
                "brand": p.brand,
                "price": p.price,
                "currency": p.currency,
                "score": p.score,
            }
            for p in products
        ]
        prompt = (
            "You are the Recommending Agent in an e-commerce assistant. "
            "Use the provided products (already ranked externally). "
            "Do not invent products or reorder them. "
            "Write a helpful response summarizing the top items. "
            f"Dialogue history: {state.dialogue_history[-5:]}\n"
            f"User message: {user_message}\n"
            f"Known preferences: {preferences}\n"
            f"Browsing history: {state.browsing_history}\n"
            f"Strategy suggestions: {state.agent_suggestions.get('recommend', [])}\n"
            f"Products: {product_brief}\n"
            "Return 1-2 candidates."
        )
        return generate_structured_output(prompt, AgentLLMOutput)
