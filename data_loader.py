import os
import google.generativeai as genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
EMBED_DIM = 768  # Gemini embedding dimension (important!)

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def _normalize_model_name(name: str) -> str:
    return name if name.startswith("models/") else f"models/{name}"


def _supports_embedding(model_obj) -> bool:
    methods = getattr(model_obj, "supported_generation_methods", None) or []
    return "embedContent" in methods


def _choose_embedding_model() -> str:
    configured = _normalize_model_name(EMBED_MODEL)
    try:
        models = list(genai.list_models())
    except Exception:
        # If model discovery fails (network/permissions), still try configured model.
        return configured

    available = [m for m in models if _supports_embedding(m)]
    available_names = {getattr(m, "name", "") for m in available}

    if configured in available_names:
        return configured

    # Prefer the latest Gemini embedding model first, then legacy fallbacks.
    for preferred in (
        "models/gemini-embedding-001",
        "models/embedding-001",
        "models/text-embedding-004",
    ):
        if preferred in available_names:
            return preferred

    if available:
        return available[0].name

    return configured


RESOLVED_EMBED_MODEL = _choose_embedding_model()


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []

    for text in texts:
        response = genai.embed_content(
            model=RESOLVED_EMBED_MODEL,
            content=text,
            task_type="retrieval_document",  # important for RAG
            output_dimensionality=EMBED_DIM,
        )
        embeddings.append(response["embedding"])

    return embeddings