import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental.ai import gemini
from dotenv import load_dotenv
import uuid
import os
from data_loader import load_and_chunk_pdf,embed_texts
from vector_db import QdrantStorage
from custom_types import RAGSearchResult,RAGUpsertResult,RAGChunkAndSrc

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx:inngest.Context):
    def _load(ctx:inngest.Context) -> RAGChunkAndSrc:
        pdf_path=ctx.event.data["pdf_path"]
        source_id=ctx.event.data.get("source_id", pdf_path)
        chunks=load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src= await ctx.step.run("load_and_chunk", lambda : _load(ctx), output_type=RAGChunkAndSrc)
    ingested= await ctx.step.run("embed_and_upsert", lambda : _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(
        step_id="embed-and-search",
        handler=lambda: _search(question, top_k),
        output_type=RAGSearchResult
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY. Add it to your .env file before running rag/query_pdf_ai.")

    configured_model = os.getenv("GEMINI_MODEL", "").strip()
    model_candidates = [m for m in [
        configured_model,
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash",
    ] if m]

    # Keep order while removing duplicates.
    model_candidates = list(dict.fromkeys(model_candidates))

    res = None
    last_error = None
    for model_name in model_candidates:
        adapter = gemini.Adapter(auth_key=api_key, model=model_name)
        try:
            step_id = f"llm-answer-{model_name.replace('.', '-')}"
            res = await ctx.step.ai.infer(
                step_id,
                adapter=adapter,
                body={
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": (
                                        "You answer questions using only the provided context. "
                                        "If the answer is not in context, say you do not know.\n\n"
                                        + user_content
                                    )
                                }
                            ],
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 1024
                    }
                }
            )
            break
        except Exception as exc:
            last_error = exc
            msg = str(exc).lower()
            if "not_found" in msg or "is not found" in msg or "models/" in msg:
                continue
            raise

    if res is None:
        raise RuntimeError(
            "No supported Gemini model found for this key/project. "
            f"Tried: {', '.join(model_candidates)}. Last error: {last_error}"
        )
    answer = ""
    if isinstance(res, dict):
        candidates = res.get("candidates", [])
        if candidates:
            parts = (((candidates[0] or {}).get("content", {}) or {}).get("parts", []))
            if parts and isinstance(parts[0], dict):
                answer = parts[0].get("text", "") or ""
    answer = answer.strip() if isinstance(answer, str) else str(answer)
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


app = FastAPI()

inngest.fast_api.serve(app,inngest_client,functions=[rag_ingest_pdf, rag_query_pdf_ai])