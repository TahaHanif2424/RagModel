import os
import google.generativeai as genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBED_MODEL = "models/embedding-001"  # Gemini embedding model
EMBED_DIM = 768  # Gemini embedding dimension (important!)

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


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
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_document",  # important for RAG
        )
        embeddings.append(response["embedding"])

    return embeddings