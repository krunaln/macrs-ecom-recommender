from __future__ import annotations

import logging
import time
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from macrs.agents.ask import AskingAgent
from macrs.agents.chitchat import ChitChatAgent
from macrs.agents.recommend import RecommendingAgent
from macrs.models import AgentOutput, PlannerDecision, ReflectionUpdate
from macrs.planner import Planner
from macrs.reflection import ReflectionEngine
from macrs.state import ConversationState


class GraphState(TypedDict, total=False):
    user_message: str
    conversation_state: ConversationState
    ask_output: AgentOutput
    recommend_output: AgentOutput
    chitchat_output: AgentOutput
    planner_decision: PlannerDecision
    reflection_update: ReflectionUpdate


def _coerce_state(value: ConversationState | dict) -> ConversationState:
    if isinstance(value, ConversationState):
        return value
    return ConversationState.model_validate(value)


class Orchestrator:
    def __init__(self) -> None:
        self.ask_agent = AskingAgent()
        self.rec_agent = RecommendingAgent()
        self.chat_agent = ChitChatAgent()
        self.planner = Planner()
        self.reflection = ReflectionEngine()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("start", self._start)
        graph.add_node("ask_agent", self._ask_agent)
        graph.add_node("recommend_agent", self._recommend_agent)
        graph.add_node("chitchat_agent", self._chitchat_agent)
        graph.add_node("planner", self._planner)
        graph.add_node("reflection", self._reflection)

        graph.set_entry_point("start")
        graph.add_edge("start", "ask_agent")
        graph.add_edge("start", "recommend_agent")
        graph.add_edge("start", "chitchat_agent")
        graph.add_edge("ask_agent", "planner")
        graph.add_edge("recommend_agent", "planner")
        graph.add_edge("chitchat_agent", "planner")
        graph.add_edge("planner", "reflection")
        graph.add_edge("reflection", END)
        return graph.compile()

    def _start(self, state: GraphState) -> GraphState:
        return {}

    def run_turn(self, state: ConversationState, user_message: str) -> GraphState:
        input_state: GraphState = {
            "user_message": user_message,
            "conversation_state": state,
        }
        return self.graph.invoke(input_state)

    def stream_turn(self, state: ConversationState, user_message: str):
        input_state: GraphState = {
            "user_message": user_message,
            "conversation_state": state,
        }
        yield from self.graph.stream(input_state, stream_mode="updates")

    def _ask_agent(self, state: GraphState) -> GraphState:
        user_message = state["user_message"]
        conv_state = _coerce_state(state["conversation_state"])
        start = time.perf_counter()
        logging.info("Agent %s: start", self.ask_agent.name)
        output = self.ask_agent.run(user_message, conv_state)
        elapsed = time.perf_counter() - start
        logging.info(
            "Agent %s: done in %.2fs (confidence=%.2f, candidates=%d)",
            self.ask_agent.name,
            elapsed,
            output.confidence,
            len(output.candidates),
        )
        return {"ask_output": output}

    def _recommend_agent(self, state: GraphState) -> GraphState:
        user_message = state["user_message"]
        conv_state = _coerce_state(state["conversation_state"])
        start = time.perf_counter()
        logging.info("Agent %s: start", self.rec_agent.name)
        output = self.rec_agent.run(user_message, conv_state)
        elapsed = time.perf_counter() - start
        logging.info(
            "Agent %s: done in %.2fs (confidence=%.2f, candidates=%d)",
            self.rec_agent.name,
            elapsed,
            output.confidence,
            len(output.candidates),
        )
        return {"recommend_output": output}

    def _chitchat_agent(self, state: GraphState) -> GraphState:
        user_message = state["user_message"]
        conv_state = _coerce_state(state["conversation_state"])
        start = time.perf_counter()
        logging.info("Agent %s: start", self.chat_agent.name)
        output = self.chat_agent.run(user_message, conv_state)
        elapsed = time.perf_counter() - start
        logging.info(
            "Agent %s: done in %.2fs (confidence=%.2f, candidates=%d)",
            self.chat_agent.name,
            elapsed,
            output.confidence,
            len(output.candidates),
        )
        return {"chitchat_output": output}

    def _planner(self, state: GraphState) -> GraphState:
        state["conversation_state"] = _coerce_state(state["conversation_state"])
        start = time.perf_counter()
        outputs = [
            state["ask_output"],
            state["recommend_output"],
            state["chitchat_output"],
        ]
        decision = self.planner.select(outputs, state["conversation_state"])
        elapsed = time.perf_counter() - start
        logging.info(
            "Planner: selected act=%s candidate=%s in %.2fs",
            decision.selected_act,
            decision.selected_candidate_id,
            elapsed,
        )
        return {"planner_decision": decision}

    def _reflection(self, state: GraphState) -> GraphState:
        conv_state = _coerce_state(state["conversation_state"])
        user_message = state["user_message"]
        start = time.perf_counter()
        reflection = self.reflection.reflect(conv_state, user_message)
        elapsed = time.perf_counter() - start
        logging.info("Reflection: updated weights in %.2fs", elapsed)
        conv_state.apply_weight_deltas(reflection.weight_deltas)
        conv_state.record_act(state["planner_decision"].selected_act)
        conv_state.turn_id += 1
        conv_state.last_user_message = user_message
        return {"reflection_update": reflection, "conversation_state": conv_state}
