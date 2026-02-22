"""
RAG Chain - Complete Retrieval-Augmented Generation pipeline.

Orchestrates the full flow:
Query → Retrieve → Generate → Respond with Sources
"""
import logging
from typing import Dict, Any, Optional, List

from .retriever import Retriever
from .generator import LLMGenerator
from ..vectorstore import VectorStore

logger = logging.getLogger(__name__)


from ..knowledge_graph import get_knowledge_graph

class RAGChain:
    """
    Complete RAG (Retrieval-Augmented Generation) chain.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        generator: LLMGenerator,
        top_k: int = 5,
        score_threshold: float = 0.1  # Lowered from 0.3
    ):
        """
        Initialize the RAG chain.
        """
        self.vector_store = vector_store
        self.generator = generator
        
        # Initialize retriever with lower threshold
        self.retriever = Retriever(
            vector_store=vector_store,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        logger.info("RAG chain initialized")
    
    def query(self, question: str, retrieval_mode: str = "hybrid") -> Dict[str, Any]:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: The user's question
            retrieval_mode: One of:
                - "vector_only": Traditional RAG (vector search only)
                - "graph_only": Graph-based retrieval only
                - "hybrid": Combined GraphRAG (default)
        
        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"Processing query [{retrieval_mode}]: {question[:100]}...")
        
        doc_count = self.vector_store.count()
        kg = get_knowledge_graph()
        
        # Step 1: Vector Search (skip if graph_only mode)
        retrieved_docs = []
        if retrieval_mode in ["vector_only", "hybrid"] and doc_count > 0:
            all_results = self.vector_store.search(question, k=self.retriever.top_k)
            retrieved_docs = all_results[:3] if all_results else []
            logger.info(f"Vector search returned {len(retrieved_docs)} documents")
        
        # Step 2: Graph Search (skip if vector_only mode)
        graph_context = ""
        found_entities = []
        
        if retrieval_mode in ["graph_only", "hybrid"]:
            try:
                # Search for matching entities in the graph
                matching_entities = kg.search_entities(question)
                
                for entity in matching_entities:
                    # Fetch details including relationships
                    related = kg.get_related(entity['id'])
                    found_entities.append(entity['name'])
                    
                    # Format graph context
                    graph_context += f"Entity: {entity['name']} ({entity['type']})\n"
                    if entity.get('properties'):
                        props = ", ".join([f"{k}: {v}" for k, v in entity['properties'].items() if v])
                        graph_context += f"Properties: {props}\n"
                    
                    if related:
                        rels = []
                        for rel in related[:5]:  # Top 5 relationships
                            direction = "->" if rel['direction'] == 'outgoing' else "<-"
                            rels.append(f"{direction} {rel['relationship']} {rel['entity']['name']} ({rel['entity']['type']})")
                        graph_context += "Relationships:\n- " + "\n- ".join(rels) + "\n"
                    graph_context += "\n"
                    
                    # Use only top 2 matched entities to avoid context overflow
                    if len(found_entities) >= 2:
                        break
                        
                if graph_context:
                    logger.info(f"Graph search found entities: {found_entities}")
            except Exception as e:
                logger.warning(f"Graph retrieval failed: {e}")

        # If no documents and no graph context, let LLM respond naturally
        if not retrieved_docs and not graph_context:
            answer = self.generator.generate(question)
            return {
                "answer": answer,
                "sources": [],
                "context_used": False,
                "documents_searched": doc_count,
                "retrieval_mode": retrieval_mode,
                "graph_entities_found": 0
            }
            
        # HYBRID STRATEGY: Combine both vector and graph results
        # Graph provides structured facts, vector provides broader context
        if retrieval_mode == "hybrid" and found_entities:
            logger.info(f"Hybrid: Found {len(found_entities)} graph entities. Combining with {len(retrieved_docs)} vector docs.")
            # Keep only top 1 vector doc to supplement graph data without noise
            retrieved_docs = retrieved_docs[:1]
        
        # Step 3: Combine Context based on mode
        context_parts = []
        
        # Add graph data first (structured, high-quality) with Strong Priority Header
        if graph_context and retrieval_mode in ["graph_only", "hybrid"]:
            context_parts.append(
                f"--- VERIFIED KNOWLEDGE GRAPH (High Reliability) ---\n"
                f"Contains structured facts and relationships. TRUST THIS DATA.\n\n"
                f"{graph_context}"
            )

        # Add vector data (broader document context) with Warning Header
        if retrieved_docs and retrieval_mode in ["vector_only", "hybrid"]:
            vector_section = "--- SEMANTIC SEARCH RESULTS (Lower Confidence) ---\n"
            vector_section += "May contain outdated or contradicting info. Use with caution.\n\n"
            
            for doc in retrieved_docs:
                content = doc.get("content", "")
                source = doc.get("metadata", {}).get("source", "unknown")
                vector_section += f"[Source: {source}]\n{content}\n\n"
            
            context_parts.append(vector_section)
            
        # Add Explicit Instruction for Hybrid Mode
        if retrieval_mode == "hybrid" and graph_context and retrieved_docs:
            context_parts.append(
                "IMPORTANT INSTRUCTION: You have two sources of information above.\n"
                "1. The KNOWLEDGE GRAPH is the source of truth. Always prioritize it for entities, brands, and relationships.\n"
                "2. The SEMANTIC SEARCH results are supplemental. If they contradict the Graph, IGNORE them.\n"
                "3. If the Graph provides a list of products (e.g. 'Adidas makes X, Y, Z'), do NOT add random products from Search unless you are certain."
            )
            
        context = "\n\n".join(context_parts)
        logger.info(f"Using context [{retrieval_mode}], length: {len(context)}")
        
        # Step 4: Extract sources
        sources = []
        
        # Add Graph as a source if used
        if graph_context and retrieval_mode in ["graph_only", "hybrid"]:
            sources.append({
                "source": "Knowledge Graph",
                "title": f"Graph Connections ({', '.join(found_entities)})",
                "relevance_score": 1.0
            })
            
        if retrieval_mode in ["vector_only", "hybrid"]:
            for doc in retrieved_docs:
                metadata = doc.get("metadata", {})
                sources.append({
                    "source": metadata.get("source", "unknown"),
                    "title": metadata.get("title", metadata.get("source", "Document")),
                    "relevance_score": doc.get("score", 0)
                })
        
        # Step 5: Generate answer
        result = self.generator.generate_with_sources(
            question=question,
            context=context,
            sources=sources
        )
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "context_used": True,
            "documents_retrieved": len(retrieved_docs),
            "graph_entities_found": len(found_entities),
            "retrieval_mode": retrieval_mode,
            "model": result["model"]
        }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        include_history: bool = True
    ) -> Dict[str, Any]:
        """Chat interface with conversation history."""
        if not messages:
            return {"answer": "Please ask a question.", "sources": []}
        
        # Get the latest user message
        latest_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                latest_message = msg.get("content", "")
                break
        
        if not latest_message:
            return {"answer": "Please ask a question.", "sources": []}
        
        return self.query(latest_message)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add documents to the knowledge base."""
        if not documents:
            return 0
        
        texts = []
        metadatas = []
        
        for doc in documents:
            content = doc.get("content", "")
            if content:
                texts.append(content)
                metadatas.append(doc.get("metadata", {}))
        
        if texts:
            self.vector_store.add_documents(texts, metadatas=metadatas)
        
        return len(texts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG chain statistics."""
        store_stats = self.vector_store.get_stats()
        
        return {
            **store_stats,
            "llm_model": self.generator.model_name,
            "retriever_top_k": self.retriever.top_k,
            "retriever_threshold": self.retriever.score_threshold
        }
