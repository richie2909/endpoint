# app.py
import os
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import requests

# -------------------- Load Environment --------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Use the Service Role key on the server
HF_TOKEN = os.getenv("HF_TOKEN")

# Use a 384-dim model (MiniLM-L6-v2 is widely available and light)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "mistralai/Mistral-8B-Instruct-v0.2")

TOP_K = int(os.getenv("TOP_K", 3))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 3000))

if not SUPABASE_URL or not SUPABASE_KEY or not HF_TOKEN:
    raise RuntimeError("Missing SUPABASE_URL, SUPABASE_KEY, or HF_TOKEN. Check your .env file.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info(
    "✅ Environment loaded: SUPABASE_URL=%s, EMBEDDING_MODEL=%s, GENERATION_MODEL=%s",
    SUPABASE_URL, EMBEDDING_MODEL, GENERATION_MODEL
)

# -------------------- App --------------------
app = FastAPI(title="Posts AI Q&A Service (Supabase Vector)")

# Lazy-load to reduce cold start time if needed
_embed_model: Optional[SentenceTransformer] = None
_EMBED_DIM: Optional[int] = None

def get_embedder() -> SentenceTransformer:
    global _embed_model, _EMBED_DIM
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        _EMBED_DIM = _embed_model.get_sentence_embedding_dimension()
        logging.info(f"🔤 Loaded embedding model: {EMBEDDING_MODEL} (dim={_EMBED_DIM})")
    return _embed_model

# -------------------- Schemas --------------------
class Question(BaseModel):
    text: str
    user_id: Optional[str] = None  # optional: restrict to saved_posts for this user

class AnswerResponse(BaseModel):
    answer: str
    context: List[str]

# -------------------- Helpers --------------------
def sb_headers(json: bool = True) -> Dict[str, str]:
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    if json:
        h["Content-Type"] = "application/json"
    return h

def clean_text(text: str) -> str:
    return " ".join(line.strip() for line in (text or "").splitlines() if line.strip())

def format_context_row(row: Dict[str, Any]) -> str:
    return (
        f"[{row.get('page_name','Unknown Page')} | Created: {row.get('created_at','N/A')} | "
        f"Last Synced: {row.get('last_synced','N/A')}] {row.get('content','')[:MAX_CONTEXT_CHARS]} "
        f"(Link: {row.get('permalink','')})"
    )

def call_match_posts(query_vec: List[float], top_k: int, user_id: Optional[str]) -> List[Dict[str, Any]]:
    """
    Calls a Supabase SQL function (RPC) that performs vector similarity search over posts.embedding.
    If user_id is provided, it filters to that user's saved_posts; otherwise it searches all posts.
    """
    payload = {
        "query_embedding": query_vec,
        "match_count": top_k,
        "p_user_id": user_id,  # nullable
    }
    url = f"{SUPABASE_URL}/rest/v1/rpc/match_posts"
    resp = requests.post(url, headers=sb_headers(), json=payload, timeout=30)
    if not resp.ok:
        raise HTTPException(status_code=500, detail=f"Supabase RPC error: {resp.text}")
    return resp.json()

def ensure_embeddings_for_missing(max_rows: int = 200) -> int:
    """
    Finds posts with NULL embedding and fills them.
    This is optional (run on-demand) and uses the Service Role key.
    """
    # 1) Pull posts missing embeddings
    url = f"{SUPABASE_URL}/rest/v1/posts?select=id,content&embedding=is.null&limit={max_rows}"
    resp = requests.get(url, headers=sb_headers(json=False), timeout=30)
    if not resp.ok:
        raise HTTPException(status_code=500, detail=f"Supabase fetch missing error: {resp.text}")
    rows = resp.json()
    if not rows:
        return 0

    model = get_embedder()
    to_update = []
    for r in rows:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        vec = model.encode([content], normalize_embeddings=True)[0].tolist()
        to_update.append({"id": r["id"], "embedding": vec})

    # 2) Upsert embeddings via PATCH (PostgREST)
    updated = 0
    for payload in to_update:
        url = f"{SUPABASE_URL}/rest/v1/posts?id=eq.{payload['id']}"
        resp = requests.patch(url, headers=sb_headers(), json={"embedding": payload["embedding"]}, timeout=30)
        if resp.ok:
            updated += 1
        else:
            logging.warning(f"Failed to update embedding for {payload['id']}: {resp.text}")

    return updated

# -------------------- LLM (Hugging Face Inference) --------------------
from huggingface_hub import InferenceClient

def generate_answer(context_blocks: List[str], question_text: str) -> str:
    if not context_blocks:
        return "I couldn't find any relevant information in the posts."
    context_text = "\n".join(f"- {c}" for c in context_blocks)

    prompt = (
        "Use ONLY the context below to answer the question. Be concise, clear, and factual.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question_text}\n"
        "Answer:"
    )

    client = InferenceClient(model=GENERATION_MODEL, token=HF_TOKEN)
    # Use sensible defaults; negative temperature/top_p values cause errors.
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "Provide short, clear, direct answers. Avoid long explanations."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
        temperature=0.2,
        top_p=0.95,
    )
    answer = "".join(
        choice.message.content for choice in response.choices if getattr(choice, "message", None)
        and getattr(choice.message, "content", None)
    ).strip()
    return clean_text(answer) or "I couldn't find any relevant information in the posts."

# -------------------- Routes --------------------
@app.get("/")
def root():
    return {"message": "✅ Service running (Supabase Vector)"}

@app.post("/ask", response_model=AnswerResponse)
def ask(q: Question):
    text = (q.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    model = get_embedder()
    query_vec = model.encode([text], normalize_embeddings=True)[0].tolist()

    # Vector search in Supabase (optionally restricted to user's saved posts)
    try:
        rows = call_match_posts(query_vec, TOP_K, q.user_id)
    except Exception as e:
        logging.exception("match_posts RPC failed")
        raise HTTPException(status_code=500, detail=str(e))

    if not rows:
        return AnswerResponse(answer="I couldn't find any relevant information in the posts.", context=[])

    contexts = [format_context_row(r) for r in rows]
    answer = generate_answer(contexts, text)
    return AnswerResponse(answer=answer, context=contexts)

# Optional admin endpoint to backfill embeddings for rows with NULL embedding
@app.post("/admin/embed-missing")
def admin_embed_missing(x_admin_token: Optional[str] = Header(default=None)):
    # simple guard; set ADMIN_TOKEN in your env to restrict usage
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    updated = ensure_embeddings_for_missing()
    return {"updated": updated}

# -------------------- Local dev entrypoint --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "true").lower() == "true",
    )
