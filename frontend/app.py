import streamlit as st
import requests
import json
import time
import uuid
from typing import List, Dict, Any
import asyncio
import aiohttp
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import settings

# Configure Streamlit page
st.set_page_config(
    page_title="Genestory Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoints
API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"


def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request to the backend."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False


def display_chat_message(role: str, content: str, metadata: Dict = None, message_id: str = None):
    """Display a chat message."""
    with st.chat_message(role):
        st.write(content)
        
        if metadata and message_id and st.checkbox("Show details", key=f"details_{message_id}"):
            with st.expander("Message Details"):
                st.json(metadata)


def check_system_health() -> Dict:
    """Check system health."""
    return make_api_request("/health")


def get_system_info() -> Dict:
    """Get system information."""
    return make_api_request("/system/info")


def initialize_system():
    """Initialize the system."""
    return make_api_request("/system/initialize", method="POST")


def send_chat_message(query: str, conversation_history: List[str] = None) -> Dict:
    """Send chat message to the backend."""
    data = {
        "query": query,
        "conversation_history": conversation_history or []
    }
    return make_api_request("/chat", method="POST", data=data)


def upload_document(content: str, source: str, document_id: str, metadata: Dict = None) -> Dict:
    """Upload a document to the knowledge base."""
    data = {
        "content": content,
        "source": source,
        "document_id": document_id,
        "metadata": metadata or {}
    }
    return make_api_request("/documents/upload", method="POST", data=data)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Genestory Chatbot")
        st.markdown("---")
        
        # System Status
        st.subheader("System Status")
        
        if st.button("Check Health", type="secondary"):
            with st.spinner("Checking system health..."):
                health = check_system_health()
                
                if "error" not in health:
                    st.success("System is healthy")
                    
                    # Display model status
                    models = health.get("models", {})
                    st.write("**Models:**")
                    for model, status in models.items():
                        icon = "✅" if status else "❌"
                        st.write(f"{icon} {model.replace('_', ' ').title()}")
                    
                    # Display database status
                    db = health.get("database", {})
                    if db:
                        st.write(f"**Database:** {db.get('num_entities', 0)} documents")
                else:
                    st.error("System health check failed")
        
        # Initialize System
        st.markdown("---")
        if not st.session_state.system_initialized:
            if st.button("Initialize System", type="primary"):
                with st.spinner("Initializing system models..."):
                    result = initialize_system()
                    if "error" not in result:
                        st.success("System initialization started")
                        st.session_state.system_initialized = True
                    else:
                        st.error("Failed to initialize system")
        else:
            st.success("System initialized")
        
        # Document Management
        st.markdown("---")
        st.subheader("Document Management")
        
        with st.expander("Upload Document"):
            doc_content = st.text_area("Document Content", height=100)
            doc_source = st.text_input("Source", placeholder="e.g., company_info.pdf")
            doc_id = st.text_input("Document ID", placeholder="e.g., doc_001")
            
            if st.button("Upload Document"):
                if doc_content and doc_source and doc_id:
                    with st.spinner("Uploading document..."):
                        result = upload_document(doc_content, doc_source, doc_id)
                        if result.get("success"):
                            st.success(result.get("message"))
                        else:
                            st.error(result.get("message", "Upload failed"))
                else:
                    st.error("Please fill in all fields")
        
        # Clear Documents
        if st.button("Clear All Documents", type="secondary"):
            if st.button("Confirm Clear", type="secondary"):
                result = make_api_request("/documents/clear", method="DELETE")
                if result.get("success"):
                    st.success("All documents cleared")
                else:
                    st.error("Failed to clear documents")
        
        # System Information
        st.markdown("---")
        if st.button("Show System Info"):
            with st.spinner("Getting system information..."):
                info = get_system_info()
                if "error" not in info:
                    st.json(info)
    
    # Main Chat Interface
    st.title("💬 Chat with Genestory Assistant")
    st.markdown("Ask me anything about Genestory's services, products, or general questions!")
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        message_id = message.get("id", f"{i}_{message['role']}")
        display_chat_message(
            message["role"], 
            message["content"], 
            message.get("metadata"),
            message_id
        )
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        user_message_id = str(uuid.uuid4())
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "id": user_message_id
        })
        
        # Display user message
        display_chat_message("user", prompt, message_id=user_message_id)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Send request to backend
                response = send_chat_message(
                    prompt, 
                    st.session_state.conversation_history
                )
                
                if "error" not in response:
                    bot_response = response.get("response", "I'm sorry, I couldn't generate a response.")
                    
                    # Display response
                    st.write(bot_response)
                    
                    # Show response details
                    with st.expander("Response Details"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Processing Time", f"{response.get('processing_time_ms', 0):.0f} ms")
                        
                        with col2:
                            st.metric("Documents Retrieved", response.get('num_documents_retrieved', 0))
                        
                        with col3:
                            st.metric("Documents Reranked", response.get('num_documents_reranked', 0))
                        
                        # Classification info
                        classification = response.get('classification', {})
                        if classification:
                            st.write("**Query Classification:**")
                            st.write(f"- Type: {classification.get('label_description', 'Unknown')}")
                            st.write(f"- Confidence: {classification.get('confidence', 0):.3f}")
                            st.write(f"- Company Related: {classification.get('is_company_related', False)}")
                        
                        # Full metadata
                        if st.checkbox("Show Full Metadata"):
                            st.json(response.get('metadata', {}))
                    
                    # Add assistant message to chat
                    assistant_message_id = str(uuid.uuid4())
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": bot_response,
                        "metadata": response,
                        "id": assistant_message_id
                    })
                    
                    # Update conversation history
                    st.session_state.conversation_history.append(prompt)
                    st.session_state.conversation_history.append(bot_response)
                    
                    # Keep only last 10 turns
                    if len(st.session_state.conversation_history) > 10:
                        st.session_state.conversation_history = st.session_state.conversation_history[-10:]
                
                else:
                    error_msg = "I'm sorry, I encountered an error. Please try again."
                    st.error(error_msg)
                    error_message_id = str(uuid.uuid4())
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "id": error_message_id
                    })
    
    # Clear chat button
    if st.button("Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()


if __name__ == "__main__":
    main()
