// server.js
import express from "express";
import dotenv from "dotenv";
import fetch from "node-fetch";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 8000;

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;
const HF_TOKEN = process.env.HF_TOKEN;

const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "sentence-transformers/all-MiniLM-L6-v2";
const GENERATION_MODEL = process.env.GENERATION_MODEL || "mistralai/Mistral-8B-Instruct-v0.2";

const TOP_K = parseInt(process.env.TOP_K || "3");
const MAX_CONTEXT_CHARS = parseInt(process.env.MAX_CONTEXT_CHARS || "3000");

if (!SUPABASE_URL || !SUPABASE_KEY || !HF_TOKEN) {
  throw new Error("Missing SUPABASE_URL, SUPABASE_KEY, or HF_TOKEN. Check your .env file.");
}

console.log(`✅ Environment loaded: SUPABASE_URL=${SUPABASE_URL}, EMBEDDING_MODEL=${EMBEDDING_MODEL}, GENERATION_MODEL=${GENERATION_MODEL}`);

app.use(express.json());

// -------------------- Helpers --------------------
function sbHeaders(json = true) {
  const h = {
    apikey: SUPABASE_KEY,
    Authorization: `Bearer ${SUPABASE_KEY}`,
  };
  if (json) h["Content-Type"] = "application/json";
  return h;
}

function cleanText(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .join(" ");
}

function formatContextRow(row) {
  return `[${row.page_name || "Unknown Page"} | Created: ${row.created_at || "N/A"} | Last Synced: ${row.last_synced || "N/A"}] ${row.content?.slice(0, MAX_CONTEXT_CHARS) || ""} (Link: ${row.permalink || ""})`;
}

// -------------------- Embeddings --------------------
async function getEmbedding(text) {
  const response = await fetch(`https://api-inference.huggingface.co/pipeline/feature-extraction/${EMBEDDING_MODEL}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ inputs: text }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Embedding request failed: ${err}`);
  }

  const embedding = await response.json();
  return embedding[0]; // HF returns [[vector]]
}

// -------------------- Supabase Vector Search --------------------
async function callMatchPosts(queryVec, top_k, user_id) {
  const payload = {
    query_embedding: queryVec,
    match_count: top_k,
    p_user_id: user_id || null,
  };

  const resp = await fetch(`${SUPABASE_URL}/rest/v1/rpc/match_posts`, {
    method: "POST",
    headers: sbHeaders(),
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    throw new Error(`Supabase RPC error: ${await resp.text()}`);
  }

  return resp.json();
}

async function ensureEmbeddingsForMissing(maxRows = 200) {
  // 1) Pull posts missing embeddings
  const resp = await fetch(`${SUPABASE_URL}/rest/v1/posts?select=id,content&embedding=is.null&limit=${maxRows}`, {
    method: "GET",
    headers: sbHeaders(false),
  });

  if (!resp.ok) throw new Error(`Supabase fetch missing error: ${await resp.text()}`);

  const rows = await resp.json();
  if (!rows.length) return 0;

  let updated = 0;
  for (const r of rows) {
    const content = (r.content || "").trim();
    if (!content) continue;
    const vec = await getEmbedding(content);

    const patchResp = await fetch(`${SUPABASE_URL}/rest/v1/posts?id=eq.${r.id}`, {
      method: "PATCH",
      headers: sbHeaders(),
      body: JSON.stringify({ embedding: vec }),
    });

    if (patchResp.ok) updated++;
    else console.warn(`Failed to update embedding for ${r.id}: ${await patchResp.text()}`);
  }

  return updated;
}

// -------------------- LLM Generation --------------------
async function generateAnswer(contextBlocks, questionText) {
  if (!contextBlocks.length) {
    return "I couldn't find any relevant information in the posts.";
  }

  const contextText = contextBlocks.map((c) => `- ${c}`).join("\n");

  const prompt = `Use ONLY the context below to answer the question. Be concise, clear, and factual.\n\nContext:\n${contextText}\n\nQuestion: ${questionText}\nAnswer:`;

  const resp = await fetch(`https://api-inference.huggingface.co/models/${GENERATION_MODEL}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      inputs: [
        { role: "system", content: "Provide short, clear, direct answers. Avoid long explanations." },
        { role: "user", content: prompt },
      ],
      parameters: { max_new_tokens: 256, temperature: 0.2, top_p: 0.95 },
    }),
  });

  if (!resp.ok) throw new Error(`HF generation failed: ${await resp.text()}`);

  const data = await resp.json();
  // Hugging Face returns different formats depending on model
  const answer = data.generated_text || data[0]?.generated_text || "";
  return cleanText(answer) || "I couldn't find any relevant information in the posts.";
}

// -------------------- Routes --------------------
app.get("/", (req, res) => {
  res.json({ message: "✅ Service running (Supabase Vector)" });
});

app.post("/ask", async (req, res) => {
  try {
    const { text, user_id } = req.body;
    if (!text?.trim()) return res.status(400).json({ detail: "Question cannot be empty." });

    const queryVec = await getEmbedding(text);
    const rows = await callMatchPosts(queryVec, TOP_K, user_id);

    if (!rows.length) return res.json({ answer: "I couldn't find any relevant information in the posts.", context: [] });

    const contexts = rows.map(formatContextRow);
    const answer = await generateAnswer(contexts, text);

    res.json({ answer, context: contexts });
  } catch (err) {
    console.error("ask endpoint failed:", err);
    res.status(500).json({ error: err.message });
  }
});

app.post("/admin/embed-missing", async (req, res) => {
  const adminToken = process.env.ADMIN_TOKEN;
  const provided = req.headers["x-admin-token"];
  if (adminToken && provided !== adminToken) {
    return res.status(403).json({ detail: "Forbidden" });
  }
  try {
    const updated = await ensureEmbeddingsForMissing();
    res.json({ updated });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// -------------------- Start Server --------------------
app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));
