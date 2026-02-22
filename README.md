# 🧠 GenAI RAG Intelligent Q&A System

A **production-grade Retrieval-Augmented Generation (RAG)** system that transforms raw data into intelligence, providing AI-powered answers grounded in your own data.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **🌐 Web Scraping** - Ingest content from any web page
- **📁 Document Upload** - Support for PDF, DOCX, TXT files
- **🔍 Semantic Search** - Find relevant information using AI embeddings
- **🕸️ Knowledge Graph** - Visualize relationships between products, brands, and categories
- **🤖 AI-Powered Answers** - Get accurate answers grounded in your data
- **📚 Source Citations** - Every answer includes sources for verification
- **💬 Chat Interface** - Modern, responsive UI for easy interaction

## 🆓 100% Free Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| **Embeddings** | HuggingFace sentence-transformers | FREE (local) |
| **Vector Store** | ChromaDB | FREE (local) |
| **LLM** | Google Gemini | FREE (generous free tier) |
| **Backend** | FastAPI + Python | FREE |
| **Frontend** | HTML/CSS/JS | FREE |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/genai-rag-intelligent-qa-system.git
cd genai-rag-intelligent-qa-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
copy .env.example .env

# Edit .env and add your Google API key
# Get FREE key at: https://makersuite.google.com/app/apikey
```

### 3. Run the Application

```bash
# Start the backend server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the UI

Open your browser and go to: **http://localhost:8000**

---

## 📖 API Documentation

Once the server is running, access interactive API docs at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest/url` | POST | Scrape content from URLs |
| `/ingest/file` | POST | Upload documents |
| `/ingest/text` | POST | Add raw text |
| `/query` | POST | Ask a question |
| `/query/chat` | POST | Chat with history |
| `/query/stats` | GET | Get knowledge base stats |

---

## 🏗️ Project Structure

```
genai-rag-intelligent-qa-system/
├── config/
│   └── settings.py          # Configuration management
├── src/
│   ├── api/                  # FastAPI backend
│   │   ├── main.py           # Application entry
│   │   ├── models.py         # Pydantic models
│   │   └── routes/           # API endpoints
│   ├── data/                 # Data acquisition
│   │   ├── collectors.py     # Web scraper + API collector
│   │   ├── preprocessor.py   # Text cleaning
│   │   └── chunker.py        # Text chunking
│   ├── vectorstore/          # Vector database
│   │   ├── embeddings.py     # HuggingFace embeddings
│   │   └── store.py          # ChromaDB integration
│   └── rag/                  # RAG pipeline
│       ├── retriever.py      # Document retrieval
│       ├── generator.py      # LLM generation
│       └── chain.py          # RAG chain
├── frontend/                 # Web UI
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔧 Configuration

Edit `.env` file to configure:

```env
# LLM (FREE - Google Gemini)
GOOGLE_API_KEY=your_api_key_here

# Embeddings (FREE - local)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Store (FREE - local)
CHROMA_PERSIST_DIR=./data/chroma_db
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t rag-qa-system .

# Run the container
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key rag-qa-system
```

---

## 📝 Usage Examples

### Ingest a Web Page

```bash
curl -X POST "http://localhost:8000/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/article"]}'
```

### Ask a Question

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this article about?"}'
```

---

## 🎯 Use Cases

- **Knowledge Base Q&A** - Build searchable documentation
- **Research Assistant** - Query research papers
- **Customer Support** - Answer questions from product docs
- **Personal Knowledge** - Organize and query your notes

---

## 📜 License

MIT License - feel free to use for personal and commercial projects.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - RAG framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [HuggingFace](https://huggingface.co/) - Embeddings
- [Google Gemini](https://deepmind.google/technologies/gemini/) - LLM

---

**Built with ❤️ as a portfolio project demonstrating production-grade AI engineering**
