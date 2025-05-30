from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from loguru import logger

from models.llm.manager import llm_manager
from retrieval.classification.classifier import query_classifier
from retrieval.retriever import document_retriever
from retrieval.reranking.reranker import document_reranker
from config.settings import settings


class ConversationState(TypedDict):
    """State structure for the conversation graph."""
    messages: List[BaseMessage]
    query: str
    query_classification: Dict[str, Any]
    retrieved_documents: List[Dict[str, Any]]
    reranked_documents: List[Dict[str, Any]]
    context: str
    response: str
    conversation_history: List[str]
    metadata: Dict[str, Any]


class ChatbotWorkflow:
    """
    LangGraph workflow for the Genestory chatbot.
    Manages the conversation flow from query to response.
    """
    
    def __init__(self):
        self.llm_manager = llm_manager
        self.query_classifier = query_classifier
        self.document_retriever = document_retriever
        self.document_reranker = document_reranker
        self.graph = None
        self._build_graph()
        
    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("classify_query", self._classify_query_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        workflow.add_node("rerank_documents", self._rerank_documents_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("generate_simple_response", self._generate_simple_response_node)
        
        # Set entry point
        workflow.set_entry_point("classify_query")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "classify_query",
            self._should_retrieve,
            {
                "retrieve": "retrieve_documents",
                "direct": "generate_simple_response"
            }
        )
        
        # Add edges
        workflow.add_edge("retrieve_documents", "rerank_documents")
        workflow.add_edge("rerank_documents", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("generate_simple_response", END)
        
        # Compile the graph
        self.graph = workflow.compile()
        
    async def _classify_query_node(self, state: ConversationState) -> ConversationState:
        """Node to classify the user query."""
        try:
            query = state["query"]
            logger.info(f"=== QUERY CLASSIFICATION ===")
            logger.info(f"User Query: '{query}'")
            
            # Classify the query
            classification = await self.query_classifier.get_classification_details(query)
            
            state["query_classification"] = classification
            state["metadata"]["classification"] = classification
            
            logger.info(f"Classification Result: {classification['label_description']} "
                       f"(confidence: {classification.get('confidence', 0):.3f})")
            logger.info(f"Company Related: {classification.get('is_company_related', False)}")
            return state
            
        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}")
            # Default to company-related if classification fails
            state["query_classification"] = {
                "predicted_label": 2,
                "is_company_related": True,
                "confidence": 0.5
            }
            return state
    
    def _should_retrieve(self, state: ConversationState) -> str:
        """Decide whether to retrieve documents or generate direct response."""
        classification = state["query_classification"]
        is_company_related = classification.get("is_company_related", True)
        
        if is_company_related:
            logger.info("Query is company-related, will retrieve documents")
            return "retrieve"
        else:
            logger.info("Query is general/greeting, will generate direct response")
            return "direct"
    
    async def _retrieve_documents_node(self, state: ConversationState) -> ConversationState:
        """Node to retrieve relevant documents."""
        try:
            query = state["query"]
            logger.info(f"=== DOCUMENT RETRIEVAL ===")
            logger.info(f"Retrieving documents for query: '{query}'")
            
            # Retrieve documents
            documents = await self.document_retriever.retrieve_documents(query)
            
            state["retrieved_documents"] = documents
            state["metadata"]["num_retrieved"] = len(documents)
            
            logger.info(f"Retrieved {len(documents)} documents")
            
            # Log retrieved document details
            if documents:
                logger.info("Retrieved documents summary:")
                for i, doc in enumerate(documents[:3]):  # Log first 3 documents
                    score = doc.get('score', 'N/A')
                    source = doc.get('metadata', {}).get('source', 'Unknown')
                    content_preview = doc.get('content', '')[:100] + "..." if len(doc.get('content', '')) > 100 else doc.get('content', '')
                    logger.info(f"  {i+1}. Source: {source}, Score: {score}")
                    logger.info(f"     Content preview: {content_preview}")
                if len(documents) > 3:
                    logger.info(f"  ... and {len(documents) - 3} more documents")
            else:
                logger.info("No documents retrieved")
            
            return state
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            state["retrieved_documents"] = []
            return state
    
    async def _rerank_documents_node(self, state: ConversationState) -> ConversationState:
        """Node to rerank retrieved documents."""
        try:
            query = state["query"]
            documents = state["retrieved_documents"]
            conversation_history = state.get("conversation_history", [])
            
            logger.info(f"=== DOCUMENT RERANKING ===")
            logger.info(f"Reranking {len(documents)} documents for query: '{query}'")
            
            # Rerank documents with context awareness
            reranked_docs = await self.document_reranker.rerank_with_context_awareness(
                query=query,
                documents=documents,
                conversation_history=conversation_history
            )
            
            state["reranked_documents"] = reranked_docs
            state["metadata"]["num_reranked"] = len(reranked_docs)
            
            # Format context for LLM
            context = self.document_retriever.format_retrieved_documents(reranked_docs)
            state["context"] = context
            
            logger.info(f"Reranked to {len(reranked_docs)} documents")
            
            # Log reranked document details
            if reranked_docs:
                logger.info("Top reranked documents:")
                for i, doc in enumerate(reranked_docs[:3]):  # Log top 3 reranked documents
                    combined_score = doc.get('combined_score', 'N/A')
                    source = doc.get('metadata', {}).get('source', 'Unknown')
                    content_preview = doc.get('content', '')[:100] + "..." if len(doc.get('content', '')) > 100 else doc.get('content', '')
                    logger.info(f"  {i+1}. Source: {source}, Combined Score: {combined_score}")
                    logger.info(f"     Content preview: {content_preview}")
            
            # Log the formatted context that will be sent to LLM
            logger.info(f"=== RETRIEVAL CONTEXT ===")
            logger.info(f"Context length: {len(context)} characters")
            if context:
                # Log first 500 characters of context
                context_preview = context[:500] + "..." if len(context) > 500 else context
                logger.info(f"Context preview:\n{context_preview}")
            else:
                logger.info("No context available")
            
            return state
            
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            state["reranked_documents"] = state["retrieved_documents"]
            state["context"] = self.document_retriever.format_retrieved_documents(
                state["retrieved_documents"]
            )
            return state
    
    async def _generate_response_node(self, state: ConversationState) -> ConversationState:
        """Node to generate response with retrieved context."""
        try:
            query = state["query"]
            context = state["context"]
            
            logger.info(f"=== RESPONSE GENERATION ===")
            logger.info(f"Generating response with context for query: '{query}'")
            logger.info(f"Using context: {len(context)} characters")
            
            # System prompt for company-related queries
            system_prompt = """You are an AI assistant for Genestory, a technology company. 
Your role is to provide helpful, accurate, and professional responses about Genestory's services, products, and information.

Guidelines:
1. Use the provided context to answer questions about Genestory
2. If the context doesn't contain relevant information, politely say so and suggest contacting Genestory directly
3. Be concise but comprehensive in your responses
4. Maintain a professional and helpful tone
5. Focus on information directly related to Genestory

Always base your answers on the provided context when available."""
            
            # Generate response
            response = await self.llm_manager.generate_response(
                user_query=query,
                system_prompt=system_prompt,
                context=context
            )
            
            state["response"] = response
            state["metadata"]["response_type"] = "context_based"
            
            logger.info("Generated response with context successfully")
            logger.info(f"Response length: {len(response)} characters")
            return state
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            state["response"] = "I apologize, but I'm having trouble generating a response right now. Please try again or contact Genestory directly for assistance."
            return state
    
    async def _generate_simple_response_node(self, state: ConversationState) -> ConversationState:
        """Node to generate simple response for general queries."""
        try:
            query = state["query"]
            
            logger.info(f"=== SIMPLE RESPONSE GENERATION ===")
            logger.info(f"Generating simple response for general query: '{query}'")
            
            # System prompt for general queries
            system_prompt = """You are a helpful AI assistant. The user has asked a general question or greeted you.
Respond in a friendly and helpful manner. Keep responses concise and appropriate.

If the user is greeting you, respond politely and offer to help with questions about Genestory.
For general questions not related to Genestory, provide a brief helpful response and gently redirect them to ask about Genestory if they need company-specific information."""
            
            # Generate response without context
            response = await self.llm_manager.generate_response(
                user_query=query,
                system_prompt=system_prompt
            )
            
            state["response"] = response
            state["metadata"]["response_type"] = "direct"
            
            logger.info("Generated simple response successfully")
            logger.info(f"Response length: {len(response)} characters")
            return state
            
        except Exception as e:
            logger.error(f"Error generating simple response: {str(e)}")
            state["response"] = "Hello! I'm here to help you with questions about Genestory. How can I assist you today?"
            return state
    
    async def process_query(
        self, 
        query: str, 
        conversation_history: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete workflow.
        
        Args:
            query: User query
            conversation_history: Previous conversation turns
            
        Returns:
            Complete response with metadata
        """
        try:
            logger.info(f"=== WORKFLOW STARTED ===")
            logger.info(f"Processing query: '{query}'")
            if conversation_history:
                logger.info(f"Conversation history: {len(conversation_history)} previous messages")
            
            # Initialize state
            initial_state = ConversationState(
                messages=[HumanMessage(content=query)],
                query=query,
                query_classification={},
                retrieved_documents=[],
                reranked_documents=[],
                context="",
                response="",
                conversation_history=conversation_history or [],
                metadata={}
            )
            
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Prepare response
            response_data = {
                "query": query,
                "response": final_state["response"],
                "classification": final_state["query_classification"],
                "num_documents_retrieved": len(final_state["retrieved_documents"]),
                "num_documents_reranked": len(final_state["reranked_documents"]),
                "metadata": final_state["metadata"]
            }
            
            logger.info(f"=== WORKFLOW COMPLETED ===")
            logger.info(f"Final response generated successfully")
            logger.info(f"Response type: {final_state['metadata'].get('response_type', 'unknown')}")
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "response": "I apologize, but I encountered an error while processing your request. Please try again.",
                "classification": {},
                "num_documents_retrieved": 0,
                "num_documents_reranked": 0,
                "metadata": {"error": str(e)}
            }
    
    async def get_workflow_info(self) -> Dict[str, Any]:
        """Get workflow information."""
        return {
            "graph_nodes": list(self.graph.nodes.keys()) if self.graph else [],
            "llm_loaded": self.llm_manager.is_loaded(),
            "classifier_trained": self.query_classifier.is_trained,
            "embedding_loaded": self.document_retriever.embedding_manager.is_loaded()
        }


# Global instance
chatbot_workflow = ChatbotWorkflow()
