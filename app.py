import os
import io
import re
import json
import hashlib
import tempfile
import zipfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from ddgs import DDGS
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


st.set_page_config(page_title="Prep AI V3.1", page_icon="🎓", layout="wide")

APP_ROOT = Path(__file__).resolve().parent
DATABASE_DIR = APP_ROOT / "faiss_index"
DATABASE_INDEX = DATABASE_DIR / "database.faiss"
DATABASE_METADATA = DATABASE_DIR / "metadata.json"
DATABASE_CONFIG = DATABASE_DIR / "config.json"
DATABASE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATABASE_SUBJECTS = ["Biology", "Chemistry", "Physics", "English"]

# -----------------------------
# Document extraction
# -----------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def normalize_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf(file_bytes, filename, source_path=None):
    records = []
    reader = PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)

    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            records.append(
                {
                    "text": text,
                    "filename": filename,
                    "source_path": source_path or filename,
                    "page": page_number,
                    "page_count": total_pages,
                }
            )

    return records


def extract_docx(file_bytes, filename, source_path=None):
    document = Document(io.BytesIO(file_bytes))

    paragraphs = []
    for paragraph in document.paragraphs:
        value = normalize_text(paragraph.text)
        if value:
            paragraphs.append(value)

    # DOCX does not reliably expose physical page numbers through python-docx.
    # Keep page as None instead of inventing page numbers.
    text = " ".join(paragraphs)

    return (
        [
            {
                "text": text,
                "filename": filename,
                "source_path": source_path or filename,
                "page": None,
                "page_count": None,
            }
        ]
        if text
        else []
    )


def extract_txt(file_bytes, filename, source_path=None):
    text = normalize_text(
        file_bytes.decode("utf-8", errors="ignore")
    )

    return (
        [
            {
                "text": text,
                "filename": filename,
                "source_path": source_path or filename,
                "page": None,
                "page_count": None,
            }
        ]
        if text
        else []
    )


def extract_md(file_bytes, filename, source_path=None):
    text = file_bytes.decode("utf-8", errors="ignore")

    # Remove fenced code, images and links while keeping useful link text.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`>-]", " ", text)
    text = normalize_text(text)

    return (
        [
            {
                "text": text,
                "filename": filename,
                "source_path": source_path or filename,
                "page": None,
                "page_count": None,
            }
        ]
        if text
        else []
    )


def detect_content_extension(file_bytes, filename=""):
    """
    Determine the supported document type from both the filename and bytes.

    This prevents a Google Drive HTML/folder response from accidentally being
    treated as a giant TXT document.
    """
    suffix = Path(filename).suffix.lower()

    if suffix in SUPPORTED_EXTENSIONS:
        return suffix

    if file_bytes.startswith(b"%PDF"):
        return ".pdf"

    if file_bytes.startswith(b"PK"):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                names = set(zf.namelist())
                if (
                    "word/document.xml" in names
                    and "[Content_Types].xml" in names
                ):
                    return ".docx"
        except zipfile.BadZipFile:
            pass

    # Only treat extension-less content as text after excluding common binary
    # signatures. Markdown and TXT are both text-based.
    try:
        decoded = file_bytes.decode("utf-8")
        if decoded.strip():
            return ".txt"
    except UnicodeDecodeError:
        pass

    return ""


def extract_document(file_bytes, filename, source_path=None):
    ext = detect_content_extension(file_bytes, filename)

    if not ext:
        raise ValueError(
            f"Unsupported or undetectable file type: {filename!r}"
        )

    if ext == ".pdf":
        return extract_pdf(file_bytes, filename, source_path)

    if ext == ".docx":
        return extract_docx(file_bytes, filename, source_path)

    if ext == ".txt":
        return extract_txt(file_bytes, filename, source_path)

    if ext == ".md":
        return extract_md(file_bytes, filename, source_path)

    raise ValueError(f"Unsupported file type: {ext}")


# -----------------------------
# Chunking
# -----------------------------
def chunk_text_records(records, chunk_size=900, overlap=150):
    """
    Word-based overlapping chunking.

    Every chunk retains:
    - filename
    - source path
    - page
    - page_count
    - subject/source type
    - chunk number
    """
    chunks = []

    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    file_chunk_counters = {}

    for record in records:
        words = record["text"].split()
        start_word = 0
        local_chunks = []

        while start_word < len(words):
            end_word = min(start_word + chunk_size, len(words))
            chunk_text = " ".join(words[start_word:end_word]).strip()

            if chunk_text:
                local_chunks.append(
                    {
                        "text": chunk_text,
                        "filename": record["filename"],
                        "source_path": record.get(
                            "source_path", record["filename"]
                        ),
                        "page": record.get("page"),
                        "page_count": record.get("page_count"),
                    }
                )

            if end_word >= len(words):
                break

            start_word = end_word - overlap

        file_key = record.get("source_path", record["filename"])
        file_chunk_counters[file_key] = (
            file_chunk_counters.get(file_key, 0) + len(local_chunks)
        )

        chunks.extend(local_chunks)

    # Add stable metadata after all chunks are created.
    per_file_seen = {}

    for global_index, chunk in enumerate(chunks):
        file_key = chunk.get("source_path", chunk["filename"])
        per_file_seen[file_key] = per_file_seen.get(file_key, 0) + 1

        chunk["chunk_id"] = global_index
        chunk["chunk_number"] = per_file_seen[file_key]
        chunk["total_file_chunks"] = file_chunk_counters[file_key]

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model(model_name=DATABASE_EMBEDDING_MODEL):
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner="Loading database FAISS index...")
def load_database():
    if not DATABASE_INDEX.exists() or not DATABASE_METADATA.exists():
        raise FileNotFoundError(
            "Database artifacts are missing. Put database.faiss, metadata.json "
            "and config.json inside the faiss_index folder beside app.py."
        )
    index = faiss.read_index(str(DATABASE_INDEX))
    with DATABASE_METADATA.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    config = {}
    if DATABASE_CONFIG.exists():
        with DATABASE_CONFIG.open("r", encoding="utf-8") as f:
            config = json.load(f)
    if index.ntotal != len(metadata):
        raise ValueError("FAISS vector count does not match metadata count.")
    return index, metadata, config


def build_vector_index(chunks):
    model = load_embedding_model()
    embeddings = model.encode(
        [c["text"] for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def fingerprint_chunks(chunks):
    payload = json.dumps(chunks, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# -----------------------------
# Hybrid retrieval
# -----------------------------
def important_words(text):
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "is", "are", "was", "were", "with", "from", "by", "as", "at",
        "what", "which", "who", "how", "why", "when", "where", "that",
        "this", "these", "those", "chapter", "topic",
    }
    return {
        w for w in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(w) > 2 and w not in stopwords
    }


def keyword_scores(query, chunks):
    qwords = important_words(query)
    scores = []
    for chunk in chunks:
        cwords = important_words(chunk["text"])
        scores.append(len(qwords & cwords) / len(qwords) if qwords else 0.0)
    return np.array(scores, dtype="float32")


def hybrid_search(query, chunks, index, top_k=8, semantic_weight=0.7, subject=None):
    model = load_embedding_model()
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    if index is None or not chunks:
        return []

    candidate_k = min(index.ntotal, max(top_k * 12, 50))
    semantic_scores, indices = index.search(q, candidate_k)

    qwords = important_words(query)
    candidates = []
    for semantic, idx in zip(semantic_scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[int(idx)]
        if subject and chunk.get("subject", "").lower() != subject.lower():
            continue
        cwords = important_words(chunk["text"])
        keyword = len(qwords & cwords) / len(qwords) if qwords else 0.0
        candidates.append((int(idx), float(semantic), float(keyword)))

    if not candidates:
        return []

    sem_values = [x[1] for x in candidates]
    key_values = [x[2] for x in candidates]
    sem_min, sem_max = min(sem_values), max(sem_values)
    key_min, key_max = min(key_values), max(key_values)

    def norm(value, low, high):
        return 0.0 if high - low < 1e-9 else (value - low) / (high - low)

    results = []
    for idx, semantic, keyword in candidates:
        sem_n = norm(semantic, sem_min, sem_max)
        key_n = norm(keyword, key_min, key_max)
        hybrid = semantic_weight * sem_n + (1 - semantic_weight) * key_n
        results.append((hybrid, semantic, keyword, idx))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def build_context(chunks, results):
    parts = []
    for n, (_, _, _, idx) in enumerate(results, start=1):
        c = chunks[idx]
        parts.append(
            f"[SOURCE {n}]\nSubject: {c.get('subject') or 'Personalized'}\nFile: {c['filename']}\nPage: {c.get('page') or 'N/A'}\nText: {c['text']}"
        )
    return "\n\n".join(parts)


def source_label(chunk):
    return f"{chunk['filename']} — page {chunk['page']}" if chunk.get("page") else chunk["filename"]


# -----------------------------
# Gemini LLM
# -----------------------------
def get_gemini_key():
    key = st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to Streamlit secrets."
        )
    return key


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    return genai.Client(api_key=get_gemini_key())


def ask_gemini(topic, context, mode, difficulty, count, instruction, model_name):
    client = get_gemini_client()

    if mode in {"MCQs", "Quiz"}:
        system_prompt = f"""
You are an expert MDCAT exam question writer.
Create up to {count} high-quality single-best-answer MCQs using ONLY the supplied SOURCE CONTEXT.
Do not use outside knowledge. If the context does not support enough questions, generate fewer.
Difficulty: {difficulty}
Each question must have exactly four options A, B, C, D and exactly one correct answer.
Return ONLY valid JSON with this exact shape:
{{"questions":[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct_answer":"A","explanation":"...","source":1}}]}}
Rules:
- correct_answer must be A, B, C, or D.
- source must be the SOURCE number supporting the question.
- Do not invent facts outside the context.
- Avoid duplicate questions.
"""
        response = client.models.generate_content(
            model=model_name,
            contents=(
                system_prompt.strip()
                + "\n\nStudent topic: " + topic
                + "\nAdditional instruction: " + (instruction or "None")
                + "\n\nSOURCE CONTEXT:\n" + context
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=8000,
            ),
        )
    else:
        system_prompt = """
You are an expert study assistant. Answer ONLY from the supplied SOURCE CONTEXT.
If the information is not available, clearly say it is not available in the provided material.
Use headings and concise explanations suitable for a student.
"""
        response = client.models.generate_content(
            model=model_name,
            contents=(
                system_prompt.strip()
                + "\n\nStudent topic: " + topic
                + "\nAdditional instruction: " + (instruction or "None")
                + "\n\nSOURCE CONTEXT:\n" + context
            ),
            config=types.GenerateContentConfig(
                max_output_tokens=8000,
            ),
        )

    return (response.text or "").strip()


def parse_mcq_json(raw):
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    data = json.loads(text)
    cleaned = []
    for i, item in enumerate(data.get("questions", []), start=1):
        if not isinstance(item, dict) or not item.get("question"):
            continue
        options = item.get("options", {})
        if not all(str(options.get(x, "")).strip() for x in ["A", "B", "C", "D"]):
            continue
        correct = str(item.get("correct_answer", "")).upper().strip()
        if correct not in {"A", "B", "C", "D"}:
            continue
        try:
            source = int(item.get("source"))
        except (TypeError, ValueError):
            source = None
        cleaned.append({
            "id": f"q{i}",
            "question": str(item["question"]).strip(),
            "options": {x: str(options[x]).strip() for x in ["A", "B", "C", "D"]},
            "correct_answer": correct,
            "explanation": str(item.get("explanation", "")).strip(),
            "source": source,
        })
    if not cleaned:
        raise ValueError("No valid MCQs were returned by Gemini.")
    return cleaned


# -----------------------------
# Prep AI Agent — single CrewAI agent + DuckDuckGo
# -----------------------------
class DuckDuckGoSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the public web with DuckDuckGo. Use this for current or external "
        "information, definitions, research topics, recent developments, and source discovery. "
        "Return concise snippets with title, URL, and source text."
    )

    def _run(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "No search query was provided."

        results = DDGS().text(query, max_results=8)
        if not results:
            return "No web results found."

        formatted = []
        for i, result in enumerate(results, start=1):
            title = result.get("title", "Untitled")
            url = result.get("href", "")
            snippet = result.get("body", "")
            formatted.append(
                f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}"
            )
            try:
                st.session_state["agent_sources"].append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                })
            except Exception:
                pass

        return "\n\n".join(formatted)


def run_prep_ai_agent(user_input, previous_history):
    gemini_key = get_gemini_key()

    # CrewAI's current Gemini integration uses the gemini/<model-id> form.
    gemini_llm = LLM(
        model="gemini/gemini-3.5-flash-lite",
        api_key=gemini_key,
    )

    agent = Agent(
        role="Prep AI Research and Learning Agent",
        goal=(
            "Help a student research, understand, organize, and study a topic. "
            "Ask for missing information before producing a final result, and use "
            "web search when external/current information is useful."
        ),
        backstory=(
            "You are a careful academic research assistant. You clarify ambiguous "
            "requests, search the web when needed, distinguish facts from uncertainty, "
            "and produce student-friendly answers with sources. Never invent sources."
        ),
        tools=[DuckDuckGoSearchTool()],
        llm=gemini_llm,
        verbose=False,
        allow_delegation=False,
    )

    history_text = "\n".join(
        f"{item['role'].upper()}: {item['content']}"
        for item in previous_history[-8:]
    )

    task_description = f"""
Student request:
{user_input}

Previous conversation:
{history_text or 'No previous conversation.'}

Instructions:
1. First decide whether you have enough information to help.
2. If an essential detail is missing, DO NOT search yet. Return exactly:
   NEEDS_INPUT: <one clear question for the student>
3. If enough information is available, use web_search when the task needs research,
   current information, source discovery, or information outside the supplied conversation.
4. For research requests, search multiple targeted queries when useful and synthesize the results.
5. Give a useful, structured answer for a student.
6. Include a Sources section with the URLs returned by web_search when web search was used.
7. Do not fabricate URLs or citations.
"""

    task = Task(
        description=task_description,
        expected_output=(
            "Either one NEEDS_INPUT question, or a complete student-friendly research/study answer "
            "with a Sources section when web search was used."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return result.raw if hasattr(result, "raw") else str(result)


# -----------------------------
# Google Drive + document processing
# -----------------------------
def is_drive_folder_url(url):
    return bool(
        re.search(
            r"drive\.google\.com/drive/(?:u/\d+/)?folders/",
            url,
            flags=re.IGNORECASE,
        )
    )


def is_drive_file_url(url):
    return bool(
        re.search(
            r"drive\.google\.com/(?:file/d/|open\?|uc\?)",
            url,
            flags=re.IGNORECASE,
        )
    )


def load_drive_file(url):
    """
    Download one public Google Drive file while preserving its real
    Google Drive filename.

    Do NOT give gdown a fake output filename such as "drive_download".
    Current gdown can resolve the real Drive filename when the output
    argument is a directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # A directory output tells current gdown to use the filename
        # reported by Google Drive.
        downloaded = gdown.download(
            url=url,
            output=str(output_dir) + os.sep,
            quiet=True,
            use_cookies=False,
        )

        if not downloaded:
            raise ValueError(
                "Google Drive file download failed. "
                "Check that the file is shared as Anyone with the link / Viewer."
            )

        path = Path(downloaded)

        if not path.exists() or not path.is_file():
            raise ValueError(
                "Google Drive download completed but the downloaded file "
                "could not be located."
            )

        file_bytes = path.read_bytes()

        # This is the actual filename reported by Google Drive.
        real_filename = path.name

    ext = detect_content_extension(file_bytes, real_filename)

    if not ext:
        raise ValueError(
            f"Unsupported Google Drive file type: {real_filename!r}. "
            "Supported formats are PDF, DOCX, TXT and MD."
        )

    # Normally the real filename already has the correct extension.
    # Only add one when Drive returns an extensionless text file.
    if Path(real_filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        real_filename = f"{real_filename}{ext}"
        
    default_names = {
        ".pdf": "Google Drive document.pdf",
        ".docx": "Google Drive document.docx",
        ".txt": "Google Drive document.txt",
        ".md": "Google Drive document.md",
    }
    return [
        {
            "name": real_filename,
            "bytes": file_bytes,
            "source_path":default_names[ext],
        }
    ]


def load_drive_folder(url):
    """
    Download a public Google Drive folder recursively and return only
    supported study documents.

    The original filenames and relative folder paths are preserved.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        download_root = Path(temp_dir) / "drive_folder"
        download_root.mkdir(parents=True, exist_ok=True)

        try:
            result = gdown.download_folder(
                url=url,
                output=str(download_root),
                quiet=True,
                use_cookies=False,
            )
        except Exception as exc:
            raise ValueError(
                "Google Drive folder download failed. Make sure the folder "
                "is shared as Anyone with the link / Viewer."
            ) from exc

        # gdown returns downloaded file descriptors in current versions,
        # but scanning the output directory is more robust across releases.
        paths = [
            p for p in download_root.rglob("*")
            if p.is_file()
        ]

        # Some gdown versions may place files one level below the requested
        # directory. If scanning is empty, also inspect returned paths.
        if not paths and result:
            for item in result:
                candidate = getattr(item, "path", None)
                if candidate:
                    candidate_path = Path(candidate)
                    if candidate_path.exists() and candidate_path.is_file():
                        paths.append(candidate_path)

        supported = []

        for path in paths:
            ext = path.suffix.lower()

            # Never ingest HTML pages, Google Drive metadata or unknown binaries.
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            file_bytes = path.read_bytes()

            detected = detect_content_extension(
                file_bytes,
                path.name,
            )

            if detected not in SUPPORTED_EXTENSIONS:
                continue

            relative = path.relative_to(download_root).as_posix()
            
       default_names = {
        ".pdf": "Google Drive document.pdf",
        ".docx": "Google Drive document.docx",
        ".txt": "Google Drive document.txt",
        ".md": "Google Drive document.md",
    }
            supported.append(
                {
                    "name": path.name,
                    "bytes": file_bytes,
                    "source_path": default_names[ext],
                }
            )

        if not supported:
            raise ValueError(
                "The Google Drive folder was downloaded, but no supported "
                "PDF, DOCX, TXT or MD files were found."
            )

        return supported


def load_drive_source(url):
    url = url.strip()

    if is_drive_folder_url(url):
        return load_drive_folder(url)

    if is_drive_file_url(url):
        return load_drive_file(url)

    raise ValueError(
        "Please provide a Google Drive file or folder sharing link."
    )


def process_files(items, chunk_size, overlap):
    records = []
    document_info = []

    for item in items:
        filename = item["name"]
        source_path = item.get("source_path", filename)

        extracted = extract_document(
            item["bytes"],
            filename,
            source_path,
        )

        records.extend(extracted)

        extension = detect_content_extension(
            item["bytes"],
            filename,
        )

        if extension == ".pdf":
            page_values = [
                r.get("page_count")
                for r in extracted
                if r.get("page_count")
            ]
            pages = max(page_values) if page_values else 0
        else:
            pages = None

        document_info.append(
            {
                "filename": filename,
                "source": source_path,
                "file_type": extension.upper().replace(".", ""),
                "characters": sum(
                    len(r["text"]) for r in extracted
                ),
                "pages": pages if pages else "N/A",
                "chunks": 0,
            }
        )

    chunks = chunk_text_records(
        records,
        chunk_size,
        overlap,
    )

    # Count chunks per source file.
    chunk_counts = {}
    for chunk in chunks:
        source_key = chunk.get("source_path", chunk["filename"])
        chunk_counts[source_key] = chunk_counts.get(source_key, 0) + 1

    for row in document_info:
        row["chunks"] = chunk_counts.get(
            row["source"],
            0,
        )

    return chunks, document_info


# -----------------------------
# PDF exports
# -----------------------------
def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def create_mcq_pdf(questions, title, include_answers=True, user_answers=None, score=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45, title=title)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=18)
    qstyle = ParagraphStyle("Q", parent=styles["Heading3"], spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("B", parent=styles["BodyText"], leading=14, spaceAfter=5)
    story = [Paragraph(esc(title), title_style)]
    if score:
        story.append(Paragraph(f"<b>Score:</b> {score['correct']} / {score['total']} &nbsp;&nbsp; <b>Percentage:</b> {score['percentage']:.1f}%", body))
        story.append(Spacer(1, 10))
    for n, q in enumerate(questions, 1):
        story.append(Paragraph(f"{n}. {esc(q['question'])}", qstyle))
        for letter in ["A", "B", "C", "D"]:
            story.append(Paragraph(f"<b>{letter}.</b> {esc(q['options'][letter])}", body))
        if user_answers is not None:
            story.append(Paragraph(f"<b>Your Answer:</b> {esc(user_answers.get(q['id'], 'Not attempted'))}", body))
        if include_answers:
            story.append(Paragraph(f"<b>Correct Answer:</b> {esc(q['correct_answer'])}", body))
            if q.get("explanation"):
                story.append(Paragraph(f"<b>Explanation:</b> {esc(q['explanation'])}", body))
        story.append(Spacer(1, 8))
    doc.build(story)
    return buffer.getvalue()


def create_explanation_pdf(topic, answer, chunks, title):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45, title=title)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=18)
    body = ParagraphStyle("B", parent=styles["BodyText"], leading=14, spaceAfter=9)
    heading = ParagraphStyle("H", parent=styles["Heading2"], spaceBefore=12, spaceAfter=8)
    story = [Paragraph(esc(title), title_style), Paragraph(f"<b>Topic:</b> {esc(topic)}", body)]
    story.append(Paragraph(esc(answer).replace("\n", "<br/>"), body))
    story.append(Paragraph("Retrieved Sources", heading))
    for i, c in enumerate(chunks, 1):
        story.append(Paragraph(f"<b>Source {i}:</b> {esc(c['filename'])} — Page {esc(c.get('page') or 'N/A')}", body))
        story.append(Paragraph(esc(c["text"]), body))
    doc.build(story)
    return buffer.getvalue()


# -----------------------------
# Session state
# -----------------------------
defaults = {
    "learning_mode": "Personalized Learning",
    "chunks": [], "index": None, "embeddings": None, "fingerprint": None,
    "document_info": [], "last_results": [], "last_context": "", "last_answer": "",
    "generated_questions": [], "quiz_submitted": False, "quiz_answers": {},
    "quiz_score": None, "quiz_nonce": 0, "last_mode": "", "last_topic": "",
    "ui_color": "Blue", "selected_model": "gemini-3.5-flash-lite",
    "agent_history": [], "agent_result": "", "agent_sources": [], "agent_question": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# UI settings
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    color_options = {
        "Blue": "#2563EB", "Green": "#16A34A", "Purple": "#7C3AED",
        "Orange": "#EA580C", "Red": "#DC2626",
    }
    st.session_state["ui_color"] = st.selectbox(
        "Change UI color", list(color_options),
        index=list(color_options).index(st.session_state["ui_color"]),
    )
    models = ["gemini-3.5-flash-lite"]
    st.session_state["selected_model"] = st.selectbox(
        "LLM Model", models,
        index=0,
        help="Prep AI V3.1 uses Gemini 3.5 Flash-Lite for study generation and the Prep AI Agent.",
    )
    st.markdown(f"""
    <style>
    div.stButton > button[kind="primary"] {{background-color:{color_options[st.session_state['ui_color']]}; border-color:{color_options[st.session_state['ui_color']]};}}
    </style>
    """, unsafe_allow_html=True)

    st.divider()
    st.header("Retrieval Settings")
    chunk_size = st.slider("Chunk size (words)", 300, 1800, 900, 100)
    overlap = st.slider("Chunk overlap (words)", 0, 400, 150, 25)
    top_k = st.slider("Retrieved chunks", 2, 15, 8)
    semantic_weight = st.slider("Semantic search weight", 0.0, 1.0, 0.7, 0.05)
    st.info("Add GEMINI_API_KEY to Streamlit secrets. Never hardcode it in app.py.")


st.title("🎓 Prep AI V3.1")
st.caption("Advanced RAG + Agentic Learning — Personalized, Database, and Prep AI Agent")

st.subheader("1. Choose Learning Mode")
learning_mode = st.radio(
    "Learning mode",
    ["Personalized Learning", "Database Learning", "Prep AI Agent"],
    horizontal=True,
)
st.session_state["learning_mode"] = learning_mode

# ------------------------------------------------------------
# Prep AI Agent
# ------------------------------------------------------------
if learning_mode == "Prep AI Agent":
    st.info(
        "Prep AI Agent is a single CrewAI agent powered by Gemini 3.5 Flash-Lite. "
        "It can clarify your request and search the public web with DuckDuckGo."
    )

    agent_topic = st.text_area(
        "Research topic / request",
        placeholder=(
            "Example: Research the latest advances in CRISPR gene editing for an MDCAT student. "
            "Include key concepts, recent developments, advantages, limitations, and sources."
        ),
        height=150,
    )
    agent_level = st.selectbox(
        "Student level",
        ["MDCAT", "Intermediate", "University", "General"],
        key="agent_level",
    )
    agent_output = st.selectbox(
        "Output type",
        ["Research report", "Study notes", "Concept explanation", "MCQ study set", "Study plan"],
        key="agent_output",
    )
    agent_instruction = st.text_area(
        "Optional instructions",
        placeholder="Example: Use simple English and highlight important exam points.",
        key="agent_instruction",
    )

    if st.session_state["agent_history"]:
        st.subheader("Agent conversation")
        for message in st.session_state["agent_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.button("🤖 Run Prep AI Agent", type="primary"):
        if not agent_topic.strip():
            st.warning("Please enter a research topic or request.")
        else:
            full_request = (
                f"Student level: {agent_level}\n"
                f"Desired output: {agent_output}\n"
                f"Request: {agent_topic}\n"
                f"Additional instructions: {agent_instruction or 'None'}"
            )
            st.session_state["agent_sources"] = []
            st.session_state["agent_history"].append({"role": "user", "content": full_request})
            try:
                with st.spinner("Prep AI Agent is thinking and searching when needed..."):
                    answer = run_prep_ai_agent(
                        full_request,
                        st.session_state["agent_history"],
                    )
                st.session_state["agent_result"] = answer
                st.session_state["agent_history"].append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as exc:
                st.error(f"Agent error: {exc}")

    if st.session_state["agent_result"]:
        result = st.session_state["agent_result"]
        if result.startswith("NEEDS_INPUT:"):
            st.warning(result.replace("NEEDS_INPUT:", "", 1).strip())
            st.info("Enter the missing information in the request box above and run the agent again.")
        else:
            st.subheader("🤖 Prep AI Agent Result")
            st.markdown(result)

            if st.session_state["agent_sources"]:
                st.subheader("🌐 Web Sources")
                seen = set()
                for source in st.session_state["agent_sources"]:
                    url = source.get("url", "")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    st.markdown(f"- [{source.get('title', 'Source')}]({url})")

    st.stop()

# ------------------------------------------------------------
# Personalized learning
# ------------------------------------------------------------
if learning_mode == "Personalized Learning":
    st.info("Upload your own study material. These files are processed in the current session and are not stored in the database.")
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    drive_url = st.text_input(
        "Optional Google Drive file or folder link",
        placeholder="Paste a public Google Drive file or folder link",
        help="Folder links are downloaded recursively. PDF, DOCX, TXT and MD files are processed.",
    )

    if uploaded or drive_url.strip():
        if st.button("Process Personalized Material", type="primary"):
            items = []

            if uploaded:
                items.extend(
                    {
                        "name": f.name,
                        "bytes": f.getvalue(),
                        "source_path": f.name,
                    }
                    for f in uploaded
                )

            if drive_url.strip():
                try:
                    with st.spinner(
                        "Downloading Google Drive file/folder..."
                    ):
                        drive_items = load_drive_source(
                            drive_url.strip()
                        )
                        items.extend(drive_items)

                    st.success(
                        f"Google Drive source loaded: "
                        f"{len(drive_items)} supported document(s)."
                    )

                except Exception as exc:
                    st.error(
                        f"Google Drive error: {exc}"
                    )

            if items:
                try:
                    with st.spinner(
                        "Extracting text, creating metadata, "
                        "chunking, embedding and building FAISS index..."
                    ):
                        chunks, info = process_files(
                            items,
                            chunk_size,
                            overlap,
                        )

                        if not chunks:
                            raise ValueError(
                                "No text could be extracted from the supplied files."
                            )

                        index, embeddings = build_vector_index(
                            chunks
                        )

                        st.session_state["chunks"] = chunks
                        st.session_state["index"] = index
                        st.session_state["embeddings"] = embeddings
                        st.session_state["fingerprint"] = fingerprint_chunks(
                            chunks
                        )
                        st.session_state["document_info"] = info
                        st.session_state["generated_questions"] = []
                        st.session_state["last_results"] = []
                        st.session_state["quiz_submitted"] = False
                        st.session_state["quiz_answers"] = {}
                        st.session_state["quiz_score"] = None

                    st.success(
                        f"Processed {len(info)} document(s) "
                        f"and created {len(chunks)} chunks."
                    )

                except Exception as exc:
                    st.error(
                        f"Processing error: {exc}"
                    )

    if st.session_state["document_info"]:
        st.subheader("2. Extracted document information")
        st.dataframe(st.session_state["document_info"], use_container_width=True, hide_index=True)
        st.caption(f"Total documents: {len(st.session_state["document_info"])}  |  Total chunks: {len(st.session_state["chunks"])}")

# ------------------------------------------------------------
# Database learning
# ------------------------------------------------------------
else:
    st.info("Database Learning uses only the pre-built FAISS index and metadata. The original PDFs are NOT required by the app.")
    try:
        db_index, db_chunks, db_config = load_database()
        st.success(f"Database loaded — {db_index.ntotal:,} vectors")
        subject = st.selectbox("Select Subject", DATABASE_SUBJECTS)
        subject_count = sum(1 for c in db_chunks if c.get("subject", "").lower() == subject.lower())
        st.caption(f"{subject}: {subject_count:,} indexed chunks")
    except Exception as exc:
        db_index, db_chunks, db_config = None, [], {}
        subject = DATABASE_SUBJECTS[0]
        st.error(f"Database loading error: {exc}")
        st.info("Put faiss_index/database.faiss, metadata.json, and config.json beside app.py.")

# ------------------------------------------------------------
# Study controls
# ------------------------------------------------------------
st.subheader("3. Study")
topic = st.text_input("Chapter / topic", placeholder="Example: Cell membrane, Genetics, Newton's laws, Tenses...")
mode = st.selectbox("Mode", ["MCQs", "Answer explanation", "Quiz"])
difficulty = st.selectbox("Difficulty level", ["Easy", "Medium", "Hard", "Mixed MDCAT Level"], index=3, disabled=(mode == "Answer explanation"))
count = st.number_input("Number of MCQs", 1, 100, 20, 1, disabled=(mode == "Answer explanation"))
instruction = st.text_area("Optional instruction", placeholder="Example: Focus on conceptual questions.")
button_label = {"MCQs": "Generate MCQs", "Answer explanation": "Get Explanation", "Quiz": "Start Quiz"}[mode]

if st.button(button_label, type="primary"):
    if not topic.strip():
        st.warning("Please enter a chapter or topic.")
    else:
        try:
            if learning_mode == "Personalized Learning":
                chunks = st.session_state["chunks"]
                index = st.session_state["index"]
                subject_filter = None
                if not chunks or index is None:
                    st.warning("Please process personalized material first.")
                    st.stop()
            else:
                chunks = db_chunks
                index = db_index
                subject_filter = subject
                if not chunks or index is None:
                    st.warning("Database is not available.")
                    st.stop()

            with st.spinner("Running hybrid RAG and generating response..."):
                results = hybrid_search(topic, chunks, index, top_k, semantic_weight, subject_filter)
                if not results:
                    st.warning("No relevant material was retrieved for this topic.")
                    st.stop()
                context = build_context(chunks, results)
                raw = ask_gemini(
                    topic, context, mode, difficulty, int(count), instruction,
                    st.session_state["selected_model"],
                )

                st.session_state["last_results"] = results
                st.session_state["last_context"] = context
                st.session_state["last_topic"] = topic
                st.session_state["last_mode"] = mode
                st.session_state["last_answer"] = ""
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_score"] = None

                if mode in {"MCQs", "Quiz"}:
                    st.session_state["quiz_nonce"] += 1
                    questions = parse_mcq_json(raw)[:int(count)]
                    for i, q in enumerate(questions, 1):
                        q["id"] = f"q{st.session_state['quiz_nonce']}_{i}"
                    st.session_state["generated_questions"] = questions
                else:
                    st.session_state["generated_questions"] = []
                    st.session_state["last_answer"] = raw

            st.success("Response generated.")
        except Exception as exc:
            st.error(f"Generation error: {exc}")

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------
if st.session_state["generated_questions"] and st.session_state["last_mode"] == "MCQs":
    st.divider()
    st.subheader("🧠 Generated MCQs with Answer Key")
    questions = st.session_state["generated_questions"]
    for n, q in enumerate(questions, 1):
        st.markdown(f"### {n}. {q['question']}")
        for letter in ["A", "B", "C", "D"]:
            st.write(f"**{letter}.** {q['options'][letter]}")
        st.success(f"**Answer:** {q['correct_answer']}")
        st.info(f"**Explanation:** {q['explanation']}")
        st.divider()
    pdf = create_mcq_pdf(questions, f"Prep AI MCQs - {st.session_state['last_topic']}", True)
    st.download_button("⬇️ Download MCQs PDF", pdf, "prep_ai_mcqs.pdf", "application/pdf")

if st.session_state["generated_questions"] and st.session_state["last_mode"] == "Quiz":
    st.divider()
    st.subheader("📝 Interactive Quiz")
    questions = st.session_state["generated_questions"]
    if not st.session_state["quiz_submitted"]:
        with st.form("quiz_form"):
            answers = {}
            for n, q in enumerate(questions, 1):
                st.markdown(f"### Question {n}")
                st.write(q["question"])
                opts = q["options"]
                selected = st.radio(
                    "Choose an answer:",
                    ["Not attempted", "A", "B", "C", "D"],
                    format_func=lambda x, opts=opts: "Not attempted" if x == "Not attempted" else f"{x}. {opts[x]}",
                    key=f"quiz_{q['id']}",
                )
                answers[q["id"]] = selected
            submit = st.form_submit_button("Submit Quiz", type="primary")
        if submit:
            correct = sum(answers.get(q["id"]) == q["correct_answer"] for q in questions)
            attempted = sum(answers.get(q["id"]) in {"A", "B", "C", "D"} for q in questions)
            total = len(questions)
            st.session_state["quiz_answers"] = answers
            st.session_state["quiz_score"] = {
                "correct": correct, "wrong": attempted-correct,
                "unattempted": total-attempted, "total": total,
                "percentage": (correct/total*100) if total else 0,
            }
            st.session_state["quiz_submitted"] = True
            st.rerun()
    else:
        score = st.session_state["quiz_score"]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Score", f"{score['correct']} / {score['total']}")
        c2.metric("Percentage", f"{score['percentage']:.1f}%")
        c3.metric("Wrong", score["wrong"])
        c4.metric("Unattempted", score["unattempted"])
        for n,q in enumerate(questions,1):
            user = st.session_state["quiz_answers"].get(q["id"], "Not attempted")
            status = "✅ Correct" if user == q["correct_answer"] else ("⚪ Unattempted" if user == "Not attempted" else "❌ Incorrect")
            st.markdown(f"### {n}. {status}")
            st.write(q["question"])
            for letter in ["A","B","C","D"]:
                st.write(f"**{letter}.** {q['options'][letter]}")
            st.write(f"**Your answer:** {user}")
            st.write(f"**Correct answer:** {q['correct_answer']}")
            st.info(f"**Explanation:** {q['explanation']}")
            st.divider()
        pdf=create_mcq_pdf(questions, f"Prep AI Quiz Review - {st.session_state['last_topic']}", True, st.session_state["quiz_answers"], score)
        st.download_button("⬇️ Download Quiz PDF", pdf, "prep_ai_quiz.pdf", "application/pdf")
        if st.button("🔄 Generate New Quiz"):
            st.session_state["generated_questions"]=[]
            st.session_state["quiz_submitted"]=False
            st.session_state["quiz_answers"]={}
            st.session_state["quiz_score"]=None
            st.rerun()

if st.session_state["last_answer"] and st.session_state["last_mode"] == "Answer explanation":
    st.divider()
    st.subheader("📖 Answer Explanation")
    st.markdown(st.session_state["last_answer"])
    source_chunks = [
        (st.session_state["chunks"] if learning_mode == "Personalized Learning" else db_chunks)[idx]
        for _,_,_,idx in st.session_state["last_results"]
    ]
    pdf=create_explanation_pdf(st.session_state["last_topic"], st.session_state["last_answer"], source_chunks, f"Prep AI Explanation - {st.session_state['last_topic']}")
    st.download_button("⬇️ Download Explanation PDF", pdf, "prep_ai_explanation.pdf", "application/pdf")

if st.session_state["last_results"]:
    st.divider()
    st.subheader("🔎 Retrieved RAG Sources")
    source_chunks = st.session_state["chunks"] if learning_mode == "Personalized Learning" else db_chunks
    for n, (_, semantic, keyword, idx) in enumerate(st.session_state["last_results"],1):
        chunk=source_chunks[idx]
        page=chunk.get("page") or "N/A"
        subject_label=chunk.get("subject") or "Personalized"
        with st.expander(
            f"Source {n}: {chunk['filename']} | "
            f"Page {page} | {subject_label}"
        ):
            st.write(chunk["text"])
            st.caption(
                f"Chunk {chunk.get('chunk_number', 'N/A')} / "
                f"{chunk.get('total_file_chunks', 'N/A')} | "
                f"Semantic: {semantic:.3f} | Keyword: {keyword:.3f}"
            )
            st.caption(
                f"Source path: {chunk.get('source_path', chunk['filename'])}"
            )
