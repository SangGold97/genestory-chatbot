# 🤖 Genestory Local Chatbot with RAG and LangGraph

A comprehensive local chatbot application built specifically for Genestory company information using Retrieval-Augmented Generation (RAG) and LangGraph workflow orchestration.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   LangGraph     │
│   Frontend      │◄──►│   Backend       │◄──►│   Workflow      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Milvus      │    │    Models       │    │   Retrieval     │
│  Vector Store   │◄──►│  LLM + Embed    │◄──►│  Classification │
│                 │    │                 │    │   Reranking     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## ✨ Features

### Core Functionality
- **🧠 RAG-based Question Answering**: Retrieves relevant company information before generating responses
- **🔀 LangGraph Workflow**: Intelligent conversation flow with conditional logic
- **🎯 Query Classification**: Automatically determines if queries are company-related or general
- **📊 Document Reranking**: Improves retrieval relevance with advanced reranking
- **💬 Multi-turn Conversations**: Maintains conversation history and context

### Technical Capabilities
- **🏠 Fully Local**: All models run locally (Gemma 3-4B, sentence-transformers)
- **⚡ Vector Search**: Fast similarity search with Milvus database
- **🔄 Streaming Responses**: Real-time response generation
- **📈 Comprehensive Monitoring**: Health checks, metrics, and system information
- **🔌 RESTful API**: Full-featured API with OpenAPI documentation

### User Interface
- **🎨 Modern Web UI**: Clean, responsive Streamlit interface
- **📁 Document Management**: Upload and manage knowledge base documents
- **📊 Response Analytics**: View classification details, retrieval stats, and processing time
- **⚙️ System Controls**: Initialize models, check health, and manage the system

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd genestory-chatbot
   ./scripts/setup.sh
   ```

2. **Start the Application**
   ```bash
   ./scripts/start.sh
   ```

3. **Access the Application**
   - Web Interface: http://localhost:8501
   - API Documentation: http://localhost:8000/docs
   - Milvus UI: http://localhost:3000

4. **Stop the Application**
   ```bash
   ./scripts/stop.sh
   ```

## 📁 Project Structure

```
genestory-chatbot/
├── 📁 backend/                 # FastAPI backend application
│   ├── main.py                # API server and endpoints
│   └── models.py              # Pydantic models for API
├── 📁 frontend/                # Streamlit web interface
│   └── app.py                 # Main UI application
├── 📁 models/                  # Model management
│   ├── llm/                   # Large Language Model (Gemma)
│   │   └── manager.py         # LLM loading and inference
│   └── embedding/             # Embedding model (MiniLM)
│       └── manager.py         # Embedding generation
├── 📁 database/                # Vector database
│   └── milvus_manager.py      # Milvus operations
├── 📁 retrieval/               # Retrieval system
│   ├── classification/        # Query classification
│   │   ├── classifier.py      # k-NN classifier
│   │   └── sample_queries.py  # Training data
│   ├── retriever.py           # Document retrieval
│   └── reranking/             # Document reranking
│       └── reranker.py        # Relevance reranking
├── 📁 langgraph/               # LangGraph workflow
│   └── workflow.py            # Conversation orchestration
├── 📁 knowledge_base/          # Knowledge base
│   ├── documents/             # User uploaded documents
│   └── sample_data/           # Sample company information
├── 📁 config/                  # Configuration
│   └── settings.py            # Application settings
├── 📁 scripts/                 # Utility scripts
│   ├── setup.sh               # Environment setup
│   ├── start.sh               # Start application
│   ├── stop.sh                # Stop application
│   └── load_documents.py      # Document loader
├── docker-compose.yml         # Milvus database setup
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Application
APP_NAME=Genestory Chatbot
DEBUG=true

# Models
LLM_MODEL_NAME=google/gemma-2-2b-it
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Database
MILVUS_HOST=localhost
MILVUS_PORT=19530

# API
API_HOST=localhost
API_PORT=8000
STREAMLIT_PORT=8501
```

### Key Settings (config/settings.py)
- **Model paths and configurations**
- **Database connection settings**
- **Retrieval parameters (top-k, thresholds)**
- **Generation parameters (temperature, max tokens)**

## 🧠 How It Works

### 1. Query Processing Flow
```
User Query → Classification → Routing Decision
     ↓                ↓              ↓
Label 1: General → Direct Response
Label 2: Company → Retrieval → Reranking → Context-aware Response
```

### 2. Query Classification
- Uses sentence-transformers embeddings
- k-NN classification with cosine similarity
- Two labels: General/Greeting vs Company-related
- Confidence threshold for routing decisions

### 3. RAG Pipeline
1. **Encode Query**: Convert text to embedding vector
2. **Vector Search**: Find similar documents in Milvus
3. **Rerank Results**: Improve relevance ordering
4. **Generate Response**: LLM with retrieved context

### 4. LangGraph Workflow
- **Nodes**: Classification, Retrieval, Reranking, Generation
- **Conditional Edges**: Route based on query type
- **State Management**: Conversation history and metadata
- **Error Handling**: Graceful degradation

## 📊 API Endpoints

### Chat
- `POST /chat` - Process chat query
- `POST /chat/stream` - Stream chat response

### System Management
- `GET /health` - System health check
- `GET /system/info` - Comprehensive system information
- `POST /system/initialize` - Initialize models

### Document Management
- `POST /documents/upload` - Upload document
- `DELETE /documents/clear` - Clear all documents
- `GET /documents/stats` - Document statistics

### Model Management
- `POST /models/load` - Load specific models

## 🔍 Usage Examples

### Chat API
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What services does Genestory offer?",
    "conversation_history": []
  }'
```

### Document Upload
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Genestory offers AI consulting services...",
    "source": "services.txt",
    "document_id": "doc_001"
  }'
```

## 🛠️ Development

### Adding New Documents
1. Place documents in `knowledge_base/documents/`
2. Run `python scripts/load_documents.py`
3. Supported formats: `.md`, `.txt`

### Customizing Models
- Update model names in `config/settings.py`
- Models will be automatically downloaded from Hugging Face
- Ensure sufficient disk space and memory

### Extending Functionality
- Add new nodes to LangGraph workflow
- Implement custom retrieval strategies
- Add support for additional file formats

## 📋 System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 20GB free space
- **Python**: 3.8+

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: CUDA-compatible (4GB+ VRAM)
- **Storage**: SSD with 50GB+ free space

## 🔧 Troubleshooting

### Common Issues

**Model Loading Errors**
- Ensure sufficient RAM/VRAM
- Check internet connection for downloads
- Verify Hugging Face model availability

**Database Connection Issues**
- Ensure Docker is running
- Check Milvus container status: `docker ps`
- Restart containers: `docker-compose restart`

**Port Conflicts**
- Change ports in `.env` file
- Kill existing processes: `pkill -f uvicorn`

### Logs
- Backend: `logs/backend.log`
- Frontend: `logs/frontend.log`
- Document Loader: `logs/document_loader.log`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** - For model hosting and transformers library
- **Milvus** - For vector database capabilities
- **LangGraph** - For workflow orchestration
- **Streamlit** - For rapid UI development
- **FastAPI** - For robust API framework

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `/docs`

---

**Built with ❤️ for Genestory by the AI Team**
