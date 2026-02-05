from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActType = Literal["ask", "recommend", "chitchat"]


class ProductCandidate(BaseModel):
    id: str
    title: str
    brand: Optional[str] = None
    description: Optional[str] = None
    categories: Optional[List[str]] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    score: float = 0.0


class AgentCandidate(BaseModel):
    candidate_id: str
    response: str
    score: float = 0.0
    rationale: Optional[str] = None
    slots: Dict[str, Any] = Field(default_factory=dict)
    products: List[ProductCandidate] = Field(default_factory=list)


class AgentOutput(BaseModel):
    agent_name: str
    act: ActType
    confidence: float
    candidates: List[AgentCandidate]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentLLMOutput(BaseModel):
    confidence: float
    candidates: List[AgentCandidate]


class StrategyUpdate(BaseModel):
    weight_updates: Dict[str, float] = Field(default_factory=dict)
    notes: Optional[str] = None


class PlannerDecision(BaseModel):
    selected_act: ActType
    selected_candidate_id: str
    selected_response: str
    strategy_update: StrategyUpdate
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlannerLLMOutput(BaseModel):
    selected_act: ActType
    selected_candidate_id: str
    notes: Optional[str] = None


class InfoReflectionOutput(BaseModel):
    current_demand: Dict[str, Any] = Field(default_factory=dict)
    browsing_history: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class StrategyReflectionOutput(BaseModel):
    suggestions: Dict[str, List[str]] = Field(default_factory=dict)
    corrective_experiences: List[str] = Field(default_factory=list)
    error_summary: Optional[str] = None


class FailureDetectionOutput(BaseModel):
    failed: bool
    reason: Optional[str] = None


class ReflectionUpdate(BaseModel):
    inferred_feedback: Dict[str, Any] = Field(default_factory=dict)
    weight_deltas: Dict[str, float] = Field(default_factory=dict)
    preference_updates: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None
