# 🏥 MediCopilot — Multi-Agent Generative AI Healthcare Copilot

> **⚠️ DISCLAIMER: This is NOT medical advice. This project is for educational and portfolio demonstration purposes only. Always consult a qualified healthcare professional for medical concerns.**

---

A production-style, multi-agent AI healthcare assistant that uses **LangChain**, **OpenAI GPT-4**, **FAISS RAG**, and a **FastAPI + Streamlit** stack to provide structured clinical decision support from free-text symptom descriptions.

---

## 🎯 Key Features

| Feature | Implementation |
|---|---|
| 🤖 Multi-Agent Architecture | 6 specialized LangChain agents + Orchestrator |
| 🧠 Chain-of-Thought Reasoning | Step-by-step differential diagnosis reasoning |
| 📚 RAG (Retrieval-Augmented Generation) | FAISS vector DB with curated medical knowledge |
| ⚠️ Risk Stratification | Rule-based + LLM hybrid risk assessment |
| 💬 Conversational Memory | Rolling conversation window for multi-turn Q&A |
| 📄 PDF Upload | Medical report text extraction |
| 📊 Query History | Session and server-side history log |
| 🔍 Explainability | Reasoning steps + RAG context transparency |

---

## 🏗️ Architecture

```
User Input (Symptoms / PDF)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                         │
│  ┌──────────────┐    ┌───────────────────────────────┐  │
│  │ Conversation │    │   Symptom Extraction Agent    │  │
│  │    Agent     │    │  (GPT-4 + JSON structured)    │  │
│  └──────────────┘    └───────────────────────────────┘  │
│                                    │                    │
│                                    ▼                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Retrieval Agent (RAG)                  │   │
│  │    FAISS Vector DB ← OpenAI Embeddings          │   │
│  │    Medical Knowledge Documents (.txt)            │   │
│  └─────────────────────────────────────────────────┘   │
│                                    │                    │
│                                    ▼                    │
│  ┌──────────────────────┐  ┌──────────────────────┐   │
│  │   Reasoning Agent    │  │  Risk Assessment      │   │
│  │  (Chain-of-Thought   │  │  Agent               │   │
│  │   Differential Dx)   │  │  (Rules + LLM)       │   │
│  └──────────────────────┘  └──────────────────────┘   │
│                                    │                    │
│                                    ▼                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Recommendation Agent                     │   │
│  │  (Next steps + Plain-language explanation)       │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Structured Response:
  • Extracted Symptoms   • Possible Conditions
  • Risk Level          • Reasoning Steps
  • Explanation         • Recommendations
  • Follow-up Questions • RAG Context Used
```

---

## 📂 Project Structure

```
healthcare_copilot/
├── main.py                          # FastAPI application entrypoint
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
│
├── backend/
│   ├── config.py                    # Centralized configuration
│   ├── agents/
│   │   ├── orchestrator.py          # Master pipeline coordinator
│   │   ├── conversation_agent.py    # Chat context + conversational replies
│   │   ├── symptom_extraction_agent.py  # NLP symptom extraction
│   │   ├── retrieval_agent.py       # RAG query + context retrieval
│   │   ├── reasoning_agent.py       # Chain-of-thought differential diagnosis
│   │   ├── risk_assessment_agent.py # Risk level classification
│   │   └── recommendation_agent.py # Next steps + explanation
│   ├── api/
│   │   └── routes.py                # FastAPI route handlers
│   ├── rag/
│   │   └── vector_store.py          # FAISS index build + retrieval
│   └── utils/
│       ├── models.py                # Pydantic data models
│       └── logger.py                # Logging configuration
│
├── frontend/
│   └── app.py                       # Streamlit UI
│
├── data/
│   ├── medical_knowledge/           # Knowledge base (.txt files)
│   │   ├── conditions_guidelines.txt
│   │   ├── triage_protocols.txt
│   │   └── differential_diagnosis.txt
│   └── faiss_index/                 # Generated FAISS index (auto-created)
│
├── scripts/
│   └── build_index.py               # Index builder script
│
├── tests/
│   └── test_agents.py               # Unit tests
│
└── docs/
    └── example_prompts.md           # Test prompts
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd healthcare_copilot

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

### 3. Build the FAISS Index

```bash
python scripts/build_index.py --verify
```

### 4. Start the Backend

```bash
# From project root
uvicorn main:app --reload --port 8000
```

The API will be available at:
- Backend: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 5. Start the Frontend

In a new terminal:

```bash
streamlit run frontend/app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

---

## 🧪 Example Prompts for Testing

### Low Risk
```
I have a runny nose, sneezing, and mild sore throat since yesterday. No fever.
```

### Moderate Risk
```
I've had a high fever of 38.8°C for 2 days with severe body aches, fatigue, 
and a dry cough. The fever isn't coming down much with paracetamol.
```

### High Risk (Emergency)
```
I'm having crushing chest pain that radiates to my left arm. I'm sweating 
a lot and feel nauseous. The pain has been going on for 20 minutes.
```

### Complex Multi-Symptom
```
For the past week I've had fatigue, frequent urination especially at night, 
excessive thirst, and blurred vision. I've also lost about 3kg without trying.
```

### With Follow-Up Context
```
[First turn]: I have a headache and fever
[Second turn]: The headache is at the back of my neck, very stiff, 
               and the light really hurts my eyes. Temperature is 39.2°C.
```

---

## 🔌 API Reference

### POST `/api/v1/chat`
Main analysis endpoint.

```json
// Request
{
  "session_id": "abc123",
  "message": "I have fever and chest pain",
  "conversation_history": [],
  "pdf_text": null
}

// Response
{
  "session_id": "abc123",
  "original_query": "I have fever and chest pain",
  "extracted_symptoms": [
    {"name": "fever", "severity": "moderate", "duration": null},
    {"name": "chest pain", "severity": null, "duration": null}
  ],
  "possible_conditions": [
    {"name": "Pericarditis", "confidence": "moderate", "matching_symptoms": ["fever", "chest pain"]},
    {"name": "Pneumonia", "confidence": "moderate", "matching_symptoms": ["fever"]}
  ],
  "risk_level": "high",
  "risk_rationale": "Chest pain combined with fever requires urgent evaluation...",
  "reasoning_steps": [...],
  "explanation": "...",
  "recommendations": [...],
  "follow_up_questions": [...],
  "disclaimer": "⚠️ This is not medical advice..."
}
```

### POST `/api/v1/upload-pdf`
Upload a medical PDF for text extraction.

```bash
curl -X POST "http://localhost:8000/api/v1/upload-pdf" \
  -F "file=@medical_report.pdf"
```

### GET `/api/v1/health`
System health check.

### GET `/api/v1/history`
Retrieve recent query log.

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=backend --cov-report=html

# Run specific test class
pytest tests/test_agents.py::TestRiskAssessmentAgent -v
```

---

## ⚙️ Configuration Options

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model to use |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `512` | RAG document chunk size |
| `CHUNK_OVERLAP` | `64` | RAG chunk overlap |
| `RAG_TOP_K` | `5` | Number of RAG results |
| `MAX_CONVERSATION_HISTORY` | `10` | Conversation memory turns |
| `BACKEND_PORT` | `8000` | FastAPI server port |

---

## 🔧 Adding Custom Knowledge

Add `.txt` files to `data/medical_knowledge/` and rebuild the index:

```bash
python scripts/build_index.py --force --verify
```

---

## 🚢 Production Considerations

For production deployment:

1. **Replace in-memory history** with PostgreSQL/Redis
2. **Add authentication** (JWT/OAuth2)
3. **Rate limiting** (slowapi)
4. **Disable CORS wildcard** (`allow_origins=["https://yourdomain.com"]`)
5. **Use gunicorn** instead of uvicorn reload
6. **Add monitoring** (Prometheus/Grafana)
7. **Containerize** with Docker

---

## 📜 License

MIT License — Free for educational and personal use.

---

## ⚠️ Important Disclaimer

**This system is for educational and demonstration purposes only.**

- It does NOT provide medical diagnosis
- It does NOT replace professional medical advice
- NEVER use this for actual medical decisions
- Always consult a qualified healthcare provider
- In emergencies, call 911/112/999 immediately

---

*Built with ❤️ using LangChain, OpenAI, FAISS, FastAPI, and Streamlit*
