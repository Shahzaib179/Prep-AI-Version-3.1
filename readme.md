# Prep AI V3.1

Prep AI V3.1 extends Prep AI V3 with a **single Prep AI Agent built with CrewAI**. The application remains a Streamlit app and keeps the Advanced RAG learning modes from V3.

## Main features

### 1. Personalized Learning
- PDF, DOCX, TXT and MD uploads
- Public Google Drive file/folder loading
- Page-aware PDF extraction
- Metadata-preserving overlapping chunks
- Sentence Transformer embeddings
- FAISS semantic retrieval
- Keyword retrieval
- Hybrid semantic + keyword ranking
- Retrieved sources shown below answers

### 2. Database Learning
The database mode uses the pre-built FAISS database from V3. The original database PDFs are **not required at runtime**.

Subjects:
- Biology
- Chemistry
- Physics
- English

Required artifacts:
```text
faiss_index/
├── database.faiss
├── metadata.json
└── config.json
```

`embeddings.npy` is not required because the vectors are already stored inside the FAISS index.

### 3. Prep AI Agent
A single CrewAI agent powered by **Gemini 3.5 Flash-Lite**. The official Gemini model ID is `gemini-3.5-flash-lite`. Google describes it as a low-latency, cost-efficient model optimized for high-throughput agentic tasks.

The agent can:
- understand a research request
- ask for missing essential information
- search the public web with DuckDuckGo
- synthesize search results
- produce research reports, study notes, explanations, MCQ study sets, or study plans
- show web sources

The agent is intentionally a **single agent**, not a multi-agent crew. CrewAI is used for the agent/tool/task orchestration layer.

## Architecture

```text
                         PREP AI V3.1
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
     Personalized       Database Learning   Prep AI Agent
       Learning              │                │
             │          Pre-built FAISS      │
             │          + metadata           │
             ▼                │                ▼
      Extraction             │          CrewAI Agent
             │                │                │
         Chunking            │          Gemini 3.5
             │                │           Flash-Lite
         Embedding           │                │
             │                │          DuckDuckGo
             ▼                ▼           Web Search
            FAISS ◄───────────┘                │
             │                                 │
             ▼                                 ▼
       Hybrid RAG                         Research Answer
     Semantic + Keyword                   + Web Sources
             │
             ▼
          Gemini
```

## GitHub structure

```text
prep-ai-v3-1/
│
├── app.py
├── requirements.txt
├── readme.md
│
├── faiss_index/
│   ├── database.faiss
│   ├── metadata.json
│   └── config.json
│
└── .streamlit/
    └── secrets.toml       # local only; NEVER commit this
```

Do not upload the original `Database pdfs/` directory.
Do not commit API keys.

## API key

Create `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

For Streamlit Community Cloud, paste the same TOML content into the app's **Secrets** field. Streamlit's Community Cloud deployment interface supports secrets and lets you select the Python version in Advanced settings.

## Python version

Use **Python 3.12** for this project. CrewAI currently requires Python >=3.10 and <3.14, while Streamlit Community Cloud currently defaults to Python 3.12. Select Python 3.12 in Streamlit Cloud Advanced settings.

## Install locally

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub deployment

1. Create a new GitHub repository, for example `prep-ai-v3-1`.
2. Upload `app.py`, `requirements.txt`, `readme.md`.
3. Upload the three FAISS artifacts into `faiss_index/`.
4. Do **not** upload `.streamlit/secrets.toml`.
5. Push the repository.

For a large FAISS index, Git LFS is recommended. Streamlit Community Cloud can use Git LFS files.

## Streamlit Community Cloud

1. Open Streamlit Community Cloud.
2. Sign in with GitHub.
3. Click **Create app**.
4. Select your repository and branch.
5. Set the entrypoint to `app.py`.
6. In Advanced settings select **Python 3.12**.
7. Add:

```toml
GEMINI_API_KEY = "your-key"
```

under Secrets.
8. Deploy.

Streamlit Community Cloud reads `requirements.txt` from the repository and installs the Python dependencies.

## Important Gemini note

Gemini 3.5 Flash-Lite is a current stable model with model ID `gemini-3.5-flash-lite`. The app uses the modern Google Gen AI SDK and CrewAI's current Gemini LLM integration rather than the deprecated `google-generativeai` package.

## Security

Never put this in GitHub:

```text
GEMINI_API_KEY = "..."
```

If a key is accidentally committed, revoke it and create a new key.
