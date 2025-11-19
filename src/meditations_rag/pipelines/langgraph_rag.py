import time
from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END

from meditations_rag.services.retrieval import RetrievalService, RetrievalResult
from meditations_rag.core.llm.base import LLMBase
from meditations_rag.config import get_logger

logger = get_logger(__name__)

class RAGState(TypedDict):
    """State for the RAG pipeline."""
    query: str
    documents: List[RetrievalResult]
    answer: str
    retrieval_metadata: Dict[str, Any]
    generation_metadata: Dict[str, Any]
    start_time: float
    total_time_seconds: float

def _format_context(results: List[RetrievalResult], include_metadata: bool = True) -> str:
    """Format retrieval results as context for LLM."""
    if not results:
        return "[No relevant context found]"
    
    context_parts = []
    for i, result in enumerate(results, 1):
        context_block = f"[Passage {i}]\n{result.text}"
        
        if include_metadata:
            metadata_lines = []
            if result.metadata.get("topic"):
                metadata_lines.append(f"Topic: {result.metadata['topic']}")
            if result.metadata.get("book"):
                metadata_lines.append(f"Book: {result.metadata['book']}")
            if result.metadata.get("chapter"):
                metadata_lines.append(f"Chapter: {result.metadata['chapter']}")
            if result.matched_question:
                metadata_lines.append(f"Related Question: {result.matched_question}")
            
            if metadata_lines:
                context_block += "\n" + "\n".join(metadata_lines)
        
        context_parts.append(context_block)
    
    return "\n\n".join(context_parts)

def _default_system_prompt() -> str:
    return """You are a knowledgeable assistant specializing in Marcus Aurelius' "Meditations" and Stoic philosophy.

Your task is to answer questions about Meditations based SOLELY on the provided context passages. Follow these guidelines:

1. **Accuracy First**: Only provide information directly supported by the context. If the context doesn't contain enough information to answer the question fully, acknowledge this limitation.
2. **Quote Directly**: When possible, include relevant quotes from the text to support your answer.
3. **Attribution**: Reference the specific passages or books when citing ideas.
4. **Stoic Context**: Explain philosophical concepts clearly, making them accessible to modern readers while maintaining their original meaning.
5. **Honest Limitations**: If the provided context doesn't address the question, say so explicitly. Don't hallucinate or invent information.
6. **Practical Wisdom**: When appropriate, connect Marcus' teachings to practical application, but always ground it in the text.
7. **Concise but Complete**: Provide thorough answers without unnecessary verbosity.

Remember: Your authority comes from the text itself. Stay faithful to what Marcus actually wrote."""

def create_rag_graph(
    retrieval_service: RetrievalService,
    llm: LLMBase,
    system_prompt: Optional[str] = None
):
    """
    Create a LangGraph for the RAG pipeline.
    
    Args:
        retrieval_service: Service for retrieving documents
        llm: LLM for generating answers
        system_prompt: Optional system prompt override
        
    Returns:
        Compiled LangGraph runnable
    """
    
    sys_prompt = system_prompt or _default_system_prompt()

    async def retrieve_node(state: RAGState) -> Dict[str, Any]:
        """Retrieve documents based on the query."""
        query = state["query"]
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        start_time = time.time()
        results = await retrieval_service.retrieve(query=query)
        duration = time.time() - start_time
        
        logger.info(f"Retrieved {len(results)} documents in {duration:.2f}s")
        
        return {
            "documents": results,
            "retrieval_metadata": {
                "num_results": len(results),
                "retrieval_time_seconds": duration,
                "avg_score": sum(r.score for r in results) / len(results) if results else 0.0
            }
        }

    async def generate_node(state: RAGState) -> Dict[str, Any]:
        """Generate answer using retrieved documents."""
        query = state["query"]
        documents = state["documents"]
        
        if not documents:
            logger.warning("No documents found, returning fallback answer.")
            return {
                "answer": "I couldn't find any relevant passages in Meditations to answer your question. Could you rephrase or ask something else?",
                "generation_metadata": {"model": "none", "generation_time_seconds": 0.0}
            }
            
        context = _format_context(documents)
        
        prompt = f"""Based on the following passages from Marcus Aurelius' "Meditations", please answer the user's question.

**Context from Meditations:**

{context}

**User Question:**
{query}

**Your Answer:**"""

        full_prompt = f"{sys_prompt}\n\n{prompt}"
        
        logger.info("Generating answer...")
        start_time = time.time()
        try:
            answer = await llm.generate(full_prompt)
            duration = time.time() - start_time
            
            # Try to get model name
            model_name = "unknown"
            if hasattr(llm, "settings"):
                settings_obj = getattr(llm, "settings")
                model_name = getattr(settings_obj, "llm_model_name", "unknown")
                
            return {
                "answer": answer,
                "generation_metadata": {
                    "model": model_name,
                    "generation_time_seconds": duration
                }
            }
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "answer": f"Sorry, I encountered an error while generating the answer: {str(e)}",
                "generation_metadata": {"error": str(e)}
            }

    # Build the graph
    workflow = StateGraph(RAGState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
