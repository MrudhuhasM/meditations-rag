from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from meditations_rag.config import get_logger
from meditations_rag.core.llm.base import LLMBase
from meditations_rag.services.guardrail import GuardrailService
from meditations_rag.services.retrieval import RetrievalResult, RetrievalService

logger = get_logger(__name__)

# --- State Definition ---


class AgentState(TypedDict):
    """State for the Agentic RAG pipeline."""

    query: str
    messages: List[
        Dict[str, str]
    ]  # Simple message history: {"role": "user"|"ai", "content": "..."}
    retrieved_docs: List[RetrievalResult]
    search_queries: List[str]  # Track what we've searched for
    iteration: int
    decision: Optional[str]  # "search", "answer", "clarify", "switch_model"
    evaluation_status: Optional[str]  # "pass", "fail"
    evaluation_feedback: Optional[str]
    active_model: str  # "fast" or "strong"
    current_search_query: Optional[str]
    answer: Optional[str]
    guardrail_blocked: bool
    guardrail_reason: Optional[str]


# --- Decision Models ---


class ControllerDecision(BaseModel):
    """Decision made by the controller agent."""

    action: Literal["search", "answer", "clarify", "switch_model"] = Field(
        description="The next action to take. 'search' to find information, 'answer' if you have enough info, 'clarify' if the user query is ambiguous, 'switch_model' if the query is too complex for the current approach."
    )
    search_query: Optional[str] = Field(
        default=None,
        description="The search query to use if action is 'search'. Be specific.",
    )
    reasoning: str = Field(
        description="Brief explanation of why this action was chosen."
    )


class EvaluationResult(BaseModel):
    """Result of the answer evaluation."""

    grounded: bool = Field(
        description="Is the answer fully supported by the retrieved context?"
    )
    relevant: bool = Field(
        description="Does the answer directly address the user's question?"
    )
    feedback: Optional[str] = Field(
        description="If not grounded or relevant, provide specific feedback on what is missing or wrong."
    )


# --- Tools ---


class RetrievalTool:
    """Wrapper for RetrievalService to be used as a tool."""

    def __init__(self, service: RetrievalService):
        self.service = service

    async def search(self, query: str) -> List[RetrievalResult]:
        """Search for documents."""
        return await self.service.retrieve(query)


# --- Nodes ---


async def controller_node(state: AgentState, llm: LLMBase) -> Dict[str, Any]:
    """Decides the next action based on state."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    feedback = state.get("evaluation_feedback")
    iteration = state.get("iteration", 0)

    # Construct context summary
    context_summary = ""
    if docs:
        context_summary = f"We have {len(docs)} documents. "
        context_summary += "Content snippets: " + " | ".join(
            [d.text[:100] + "..." for d in docs[:3]]
        )
    else:
        context_summary = "No documents retrieved yet."

    prompt = f"""You are the Controller Agent for a RAG system on Marcus Aurelius' Meditations.

User Query: "{query}"

Current Context Status: {context_summary}
Iteration: {iteration}
Previous Feedback (if any): {feedback}

Your goal is to answer the user's question accurately using the retrieval tool.
- If you need information, choose 'search' and provide a specific query.
- If you have searched and have enough information, choose 'answer'.
- If the user's query is unclear, choose 'clarify'.
- If the query is very complex/philosophical and might need a stronger model (or if you are failing repeatedly), choose 'switch_model'.
- If you have searched multiple times (3+) without success, you should probably 'answer' with what you have or 'clarify'.

Decide the next step."""

    try:
        decision: ControllerDecision = await llm.generate_structured(
            prompt, ControllerDecision
        )
        logger.info(f"Controller decision: {decision.action} - {decision.reasoning}")

        updates: Dict[str, Any] = {
            "decision": decision.action,
            "iteration": iteration + 1,
        }
        if decision.action == "switch_model":
            updates["active_model"] = "strong"

        if decision.search_query:
            updates["search_queries"] = state.get("search_queries", []) + [
                decision.search_query
            ]
            # Store the query to be used by the retriever
            updates["current_search_query"] = decision.search_query

        return updates
    except Exception as e:
        logger.error(f"Controller failed: {e}")
        # Fallback
        return {"decision": "answer", "iteration": iteration + 1}


async def retriever_node(
    state: AgentState, retrieval_tool: RetrievalTool
) -> Dict[str, Any]:
    """Executes the retrieval."""
    query = state.get("current_search_query") or state["query"]
    logger.info(f"Retrieving for: {query}")

    new_docs = await retrieval_tool.search(query)

    # Merge with existing docs (avoiding duplicates by ID)
    existing_docs = state.get("retrieved_docs", [])
    existing_ids = {d.chunk_id for d in existing_docs}

    unique_new_docs = [d for d in new_docs if d.chunk_id not in existing_ids]
    all_docs = existing_docs + unique_new_docs

    return {"retrieved_docs": all_docs}


async def generator_node(state: AgentState, llm: LLMBase) -> Dict[str, Any]:
    """Generates the answer."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])

    context_text = "\n\n".join(
        [f"--- Document {i + 1} ---\n{d.text}" for i, d in enumerate(docs)]
    )

    prompt = f"""Answer the user's question based ONLY on the provided context from Meditations.

Context:
{context_text}

User Question: {query}

Answer:"""

    response = await llm.generate(prompt)
    return {
        "messages": state.get("messages", []) + [{"role": "ai", "content": response}],
        "answer": response,  # Store separately for easy access
    }


async def evaluator_node(state: AgentState, llm: LLMBase) -> Dict[str, Any]:
    """Evaluates the generated answer."""
    query = state["query"]
    answer = state.get("answer", "")
    docs = state.get("retrieved_docs", [])

    context_text = "\n\n".join([d.text for d in docs])

    prompt = f"""Evaluate the following answer for groundedness and relevance.

Context:
{context_text}

User Question: {query}
Generated Answer: {answer}

Is the answer grounded in the context? Is it relevant to the question?
If 'No', provide feedback on what is missing or hallucinated."""

    try:
        result: EvaluationResult = await llm.generate_structured(
            prompt, EvaluationResult
        )

        status = "pass" if result.grounded and result.relevant else "fail"
        logger.info(f"Evaluation: {status}. Feedback: {result.feedback}")

        return {"evaluation_status": status, "evaluation_feedback": result.feedback}
    except Exception as e:
        logger.error(f"Evaluator failed: {e}")
        return {"evaluation_status": "pass"}  # Fail open if evaluator breaks


async def guardrail_node(
    state: AgentState, guardrail: GuardrailService
) -> Dict[str, Any]:
    """Checks if the query is safe and relevant."""
    query = state["query"]
    result = await guardrail.validate(query)

    if not result.allowed:
        return {
            "guardrail_blocked": True,
            "guardrail_reason": result.reason,
            "answer": result.refusal_message,
            "messages": state.get("messages", [])
            + [{"role": "ai", "content": result.refusal_message}],
        }

    return {"guardrail_blocked": False}


# --- Graph Construction ---


def create_agentic_rag_graph(
    retrieval_service: RetrievalService,
    llm_fast: LLMBase,
    llm_strong: Optional[LLMBase] = None,
):
    """
    Creates the Agentic RAG LangGraph.

    Args:
        retrieval_service: The retrieval service.
        llm_fast: The default/fast LLM.
        llm_strong: Optional stronger LLM for complex queries/switching.
    """
    llm_strong = llm_strong or llm_fast
    retrieval_tool = RetrievalTool(retrieval_service)
    guardrail_service = GuardrailService(llm_fast)

    workflow = StateGraph(AgentState)

    # Define Node Wrappers to inject dependencies
    async def run_controller(state):
        # Choose model based on state
        model = llm_strong if state.get("active_model") == "strong" else llm_fast
        return await controller_node(state, model)

    async def run_retriever(state):
        return await retriever_node(state, retrieval_tool)

    async def run_generator(state):
        model = llm_strong if state.get("active_model") == "strong" else llm_fast
        return await generator_node(state, model)

    async def run_evaluator(state):
        # Always use strong model for evaluation if available
        return await evaluator_node(state, llm_strong)

    async def run_guardrail(state):
        return await guardrail_node(state, guardrail_service)

    # Add Nodes
    workflow.add_node("guardrail", run_guardrail)
    workflow.add_node("controller", run_controller)
    workflow.add_node("retriever", run_retriever)
    workflow.add_node("generator", run_generator)
    workflow.add_node("evaluator", run_evaluator)

    # Define Edges
    workflow.set_entry_point("guardrail")

    def guardrail_router(state: AgentState):
        if state.get("guardrail_blocked"):
            return END
        return "controller"

    workflow.add_conditional_edges(
        "guardrail",
        guardrail_router,
        {END: END, "controller": "controller"},
    )

    def router(state: AgentState):
        decision = state.get("decision")
        iteration = state.get("iteration", 0)

        if iteration > 5:  # Hard limit
            return "generator"

        if decision == "search":
            return "retriever"
        elif decision == "answer":
            return "generator"
        elif decision == "switch_model":
            # Logic to switch model is handled in state update, but we need to loop back
            # We'll update state to "strong" and go back to controller or retriever?
            # Let's go back to controller to re-evaluate with new brain
            return "controller"
        elif decision == "clarify":
            return END
        else:
            return "generator"

    workflow.add_conditional_edges(
        "controller",
        router,
        {
            "retriever": "retriever",
            "generator": "generator",
            "controller": "controller",
            END: END,
        },
    )

    workflow.add_edge("retriever", "controller")
    workflow.add_edge("generator", "evaluator")

    def evaluator_router(state: AgentState):
        status = state.get("evaluation_status")
        iteration = state.get("iteration", 0)

        if status == "pass" or iteration > 5:
            return END
        else:
            return "controller"  # Go back to controller with feedback

    workflow.add_conditional_edges(
        "evaluator", evaluator_router, {END: END, "controller": "controller"}
    )

    return workflow.compile()
