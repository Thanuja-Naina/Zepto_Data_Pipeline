import os
from pathlib import Path
from typing import TypedDict, Literal

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"},
)

def load_and_index_documents():
    existing = collection.count()
    if existing >= 8:
        return

    ids, documents, metadatas, embeddings = [], [], [], []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        chunk_id = path.stem
        ids.append(chunk_id)
        documents.append(text)
        metadatas.append({"document_id": chunk_id, "source": path.name})
        embeddings.append(model.encode(text).tolist())

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

load_and_index_documents()

PROMPT_TEMPLATE = """ROLE:
You are Zepto's support policy assistant.

CONTEXT:
Use only the retrieved Zepto policy context supplied below.
{context}

TASK:
Answer the user's question using the provided policy context.

FORMAT:
Return a concise answer and identify the supporting source IDs.

LENGTH:
Keep the answer concise and directly relevant.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent or assume Zepto policies.

FEW-SHOT EXAMPLE:
User: How long can I report a damaged grocery item?
Context: doc_02 says grocery and perishable items may be reported within 24 hours.
Answer: Grocery and perishable items may be reported within 24 hours of delivery if damaged, spoiled, or incorrect.
"""

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours"
]

class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved: list
    answer: str
    sources: list[str]
    confidence: float

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

def classify_intent(state: GraphState):
    query = state["query"].lower()
    intent = (
        "policy_question"
        if any(keyword in query for keyword in POLICY_KEYWORDS)
        else "general_question"
    )
    return {"intent": intent}

def retrieve_and_answer(state: GraphState):
    query = state["query"]
    query_embedding = model.encode(query).tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    docs = result.get("documents", [[]])[0]
    ids = result.get("ids", [[]])[0]

    retrieved = [
        {"id": chunk_id, "text": text}
        for chunk_id, text in zip(ids, docs)
    ]

    if not retrieved:
        return {
            "answer": "No relevant policy context was retrieved.",
            "sources": [],
            "confidence": 0.0,
            "retrieved": [],
        }

    if MOCK_LLM:
        top_chunk_snippet = retrieved[0]["text"][:200]
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
    else:
        answer = real_llm_answer(query, retrieved)

    return {
        "retrieved": retrieved,
        "answer": answer,
        "sources": [item["id"] for item in retrieved],
        "confidence": 1.0 if MOCK_LLM else 0.9,
    }

def direct_answer(state: GraphState):
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
    else:
        answer = real_llm_direct_answer(state["query"])

    return {
        "answer": answer,
        "sources": [],
        "confidence": 1.0 if MOCK_LLM else 0.9,
    }

def call_real_llm(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    last_error = None
    for attempt in range(3):
        try:
            corrective = ""
            if attempt > 0:
                corrective = (
                    "\nPrevious output was invalid. Return only a concise answer. "
                    "Do not add unsupported policy facts."
                )
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt + corrective},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"LLM generation failed after 3 attempts: {last_error}")

def real_llm_answer(query: str, retrieved: list) -> str:
    context = "\n\n".join(
        f"[{item['id']}] {item['text']}" for item in retrieved
    )
    prompt = PROMPT_TEMPLATE.format(context=context)
    prompt += f"\n\nUser question: {query}"
    return call_real_llm(prompt)

def real_llm_direct_answer(query: str) -> str:
    prompt = """ROLE:
You are a Zepto support assistant.

CONTEXT:
No Zepto policy retrieval was requested.

TASK:
Answer the user's general question.

FORMAT:
Give a concise answer.

LENGTH:
Keep it concise.

NEGATIVE CONSTRAINT:
Do not invent Zepto policy information.

FEW-SHOT EXAMPLE:
User: What is Python?
Answer: Python is a programming language.
"""
    return call_real_llm(prompt + f"\n\nUser question: {query}")

def route(state: GraphState):
    return state["intent"]

builder = StateGraph(GraphState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)
builder.set_entry_point("classify_intent")
builder.add_conditional_edges(
    "classify_intent",
    route,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer",
    },
)
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)
graph = builder.compile()

app = FastAPI(
    title="Zepto Support Assistant",
    description="Offline-first RAG support assistant using LangGraph and ChromaDB.",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"message": "Zepto Support Assistant is running."}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = graph.invoke({"query": request.query})
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )
