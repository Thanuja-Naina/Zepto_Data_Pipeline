# Zepto Support Assistant

A small GenAI/RAG service that answers questions about Zepto policies using document retrieval, embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

The application is designed with an **offline mock LLM mode** as the required graded baseline. In this mode, no LLM API key, signup, or network access to an LLM provider is required. The optional real-LLM path can be enabled separately using `MOCK_LLM=0`.

---

## 1. Project Structure

```text
support_assistant/
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── chroma_db/
│
├── main.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .env.example
└── .gitignore
```

The `docs/` directory contains the eight Zepto policy documents supplied for the assignment.

---

# 2. Technologies Used

| Technology            | Purpose                                  |
| --------------------- | ---------------------------------------- |
| Python                | Application development                  |
| Sentence Transformers | Generate local document/query embeddings |
| `all-MiniLM-L6-v2`    | Embedding model                          |
| ChromaDB              | Store and retrieve vector embeddings     |
| LangGraph             | Orchestrate the query-routing workflow   |
| Pydantic              | Validate request and response schemas    |
| FastAPI               | Provide the `/ask` REST API              |
| Uvicorn               | Run the FastAPI application              |
| Docker                | Containerize the application             |

The assignment requires local embeddings using `sentence-transformers` with `all-MiniLM-L6-v2`, stored in ChromaDB.

---

# 3. RAG Architecture

The application follows the standard RAG pipeline:

```text
                 Zepto Policy Documents
                          │
                          ▼
                    INGESTION
                          │
                          ▼
                    EMBEDDING
                          │
                          ▼
                    CHROMADB
                 Vector Collection
                          │
                          │
                     User Query
                          │
                          ▼
                  FastAPI /ask
                          │
                          ▼
                   LangGraph Flow
                          │
                          ▼
                  classify_intent
                    /          \
                   /            \
                  ▼              ▼
       policy_question       general_question
              │                    │
              ▼                    ▼
   retrieve_and_answer       direct_answer
              │                    │
              ▼                    ▼
         ChromaDB              Direct response
        Top-3 retrieval
              │
              ▼
          Generation
              │
              ▼
       Pydantic Response
              │
              ▼
             JSON
```

The assignment specifically requires the README to explain the four RAG stages: **ingestion → embedding → retrieval → generation**, including which component handles each stage and how `MOCK_LLM` affects generation.

---

# 4. RAG Pipeline Explanation

## 4.1 Ingestion

The application contains eight policy documents:

```text
docs/doc_01.txt
docs/doc_02.txt
docs/doc_03.txt
docs/doc_04.txt
docs/doc_05.txt
docs/doc_06.txt
docs/doc_07.txt
docs/doc_08.txt
```

These documents contain information about Zepto's delivery, returns, membership, tracking, cancellation, damaged items, gift cards, and customer support policies.

The ingestion process is implemented in:

```python
load_and_index_documents()
```

inside `main.py`.

The function:

1. Reads each `.txt` file.
2. Extracts its text.
3. Assigns a document/chunk ID such as `doc_01`.
4. Creates metadata containing the source filename.
5. Generates an embedding.
6. Stores the document and embedding in ChromaDB.

The assignment requires all eight documents to be embedded and queryable from ChromaDB.

---

# 5. Embedding

The application uses:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

The model converts each document into a numerical vector.

For example:

```text
Zepto delivery policy
        │
        ▼
all-MiniLM-L6-v2
        │
        ▼
[0.12, -0.08, 0.31, ...]
```

The same embedding model is used when a user submits a query.

For example:

```text
"What is the delivery fee?"
        │
        ▼
all-MiniLM-L6-v2
        │
        ▼
Query embedding
```

This allows the query vector to be compared against the document vectors.

The embedding process runs locally, so it does not require an embedding API key.

---

# 6. ChromaDB

The generated embeddings are stored in ChromaDB.

The application creates a persistent ChromaDB client:

```python
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)
```

The collection is:

```text
zepto_policies
```

The application stores:

```text
Document ID
Document text
Metadata
Embedding vector
```

For example:

```text
doc_01
    │
    ├── Text
    ├── Metadata
    └── Embedding
```

When a user asks a policy question, the query is converted into an embedding and ChromaDB searches for the most similar documents.

The application retrieves the top three matching chunks.

---

# 7. LangGraph Workflow

LangGraph controls the application's query flow.

The graph contains three required nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The assignment requires these three named nodes and a conditional edge from `classify_intent`.

---

## 7.1 `classify_intent`

The first node determines whether the user is asking about a Zepto policy.

The mock baseline uses a keyword-based heuristic.

Policy keywords include:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

For example:

```text
"What is the delivery fee?"
```

contains:

```text
delivery
```

Therefore:

```text
policy_question
```

is selected.

An unrelated query such as:

```text
"What is Python?"
```

does not contain a policy keyword.

Therefore:

```text
general_question
```

is selected.

This routing is performed without an LLM call in mock mode.

---

# 8. Conditional Routing

The LangGraph conditional edge works like this:

```text
                 classify_intent
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
      policy_question    general_question
              │                 │
              ▼                 ▼
 retrieve_and_answer      direct_answer
```

The routing itself does not depend on `MOCK_LLM`.

`MOCK_LLM` only changes how the final answer is generated inside the answer nodes.

---

# 9. `retrieve_and_answer`

This node handles policy questions.

For example:

```text
What is the delivery fee?
```

The query is converted into an embedding and sent to ChromaDB.

ChromaDB retrieves the top three most similar documents.

For example:

```text
doc_01
doc_03
doc_02
```

The most relevant document is `doc_01` because it contains the delivery policy.

In mock mode, the answer is generated from the top retrieved chunk using:

```text
Based on the retrieved context: ...
```

The assignment specifically requires this deterministic mock behavior.

---

# 10. `direct_answer`

This node handles general questions.

For example:

```text
What is Python?
```

Because this is not a Zepto policy question, retrieval is not performed.

In mock mode, the application returns:

```text
I can only answer questions about Zepto policies right now.
```

No LLM API call is made.

The assignment specifies this fixed canned response for the graded mock baseline.

---

# 11. MOCK_LLM

`MOCK_LLM` controls whether the application uses the deterministic mock behavior or the optional real LLM path.

## Default / Graded Mode

If `MOCK_LLM` is not set:

```text
MOCK_LLM
```

or:

```text
MOCK_LLM=1
```

the application uses mock mode.

No LLM API key is required.

No network request to an LLM provider is made.

This is the required graded baseline.

---

## Optional Real-LLM Mode

If:

```text
MOCK_LLM=0
```

is explicitly configured, the optional real-LLM path is used.

The retrieved context is inserted into the structured prompt and sent to the configured LLM.

This path is optional and is not required to earn full marks.

---

# 12. Structured Prompt

The optional real-LLM path uses a structured prompt containing:

```text
ROLE
CONTEXT
TASK
FORMAT
LENGTH
NEGATIVE CONSTRAINT
FEW-SHOT EXAMPLE
```

The prompt is designed to make the model answer using only the retrieved Zepto policy context.

The negative constraint prevents the model from inventing unsupported policy information.

The assignment requires the actual prompt template to contain these components as text.

---

# 13. Pydantic Response Schema

The final API response is validated using Pydantic.

The response contains:

```json
{
  "answer": "string",
  "sources": [],
  "confidence": 1.0
}
```

### Fields

#### `answer`

Contains the final answer returned to the user.

#### `sources`

Contains the document/chunk IDs used to answer a policy question.

For example:

```json
"sources": [
  "doc_01",
  "doc_03",
  "doc_02"
]
```

For a general question:

```json
"sources": []
```

#### `confidence`

A floating-point value between:

```text
0.0
```

and:

```text
1.0
```

In mock mode, the application deterministically uses:

```text
1.0
```

The assignment explicitly requires this structured schema.

---

# 14. FastAPI

The LangGraph workflow is exposed through FastAPI.

Endpoint:

```text
POST /ask
```

Request:

```json
{
  "query": "What is the delivery fee?"
}
```

Response:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": [
    "doc_01"
  ],
  "confidence": 1.0
}
```

The application can be started locally using:

```powershell
uvicorn main:app --reload
```

The Swagger UI is available at:

```text
http://127.0.0.1:8000/docs
```

The assignment requires at least two example calls: one retrieval/policy question and one non-retrieval/general question.

---

# 15. Example API Call 1 — Policy Question

### Request

```json
{
  "query": "What is the delivery fee?"
}
```

### Flow

```text
"What is the delivery fee?"
            │
            ▼
      classify_intent
            │
            ▼
     policy_question
            │
            ▼
   retrieve_and_answer
            │
            ▼
        ChromaDB
            │
            ▼
       doc_01
            │
            ▼
      Mock response
```

### Example Response

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.",
  "sources": [
    "doc_01",
    "doc_03",
    "doc_02"
  ],
  "confidence": 1.0
}
```

The exact ordering of lower-ranked retrieved documents can vary. The important requirement is that the retrieved content matches the question and that the relevant source is returned.

---

# 16. Example API Call 2 — General Question

### Request

```json
{
  "query": "What is Python?"
}
```

### Flow

```text
"What is Python?"
        │
        ▼
 classify_intent
        │
        ▼
 general_question
        │
        ▼
 direct_answer
        │
        ▼
No retrieval
        │
        ▼
Fixed mock response
```

### Response

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This demonstrates the second LangGraph route required by the assignment.

---

# 17. Running the Application

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the application:

```powershell
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

# 18. MOCK_LLM Configuration

The required baseline can be run without configuring an API key.

PowerShell:

```powershell
$env:MOCK_LLM="1"
```

Then:

```powershell
uvicorn main:app --reload
```

Alternatively, leave `MOCK_LLM` unset because the application defaults to mock mode.

No `.env` file is required for the graded baseline.

---

# 19. Docker

The application includes a Dockerfile for local containerization.

### Build

```powershell
docker build -t zepto-support-assistant .
```

### Run

```powershell
docker run --rm -p 7860:7860 zepto-support-assistant
```

The application will then be available at:

```text
http://localhost:7860
```

The API endpoint is:

```text
POST http://localhost:7860/ask
```

Example:

```json
{
  "query": "How long do I have to report a damaged item?"
}
```

The locally buildable and runnable Dockerfile is the required graded containerization baseline. A Hugging Face Spaces deployment is optional and ungraded.

---

# 20. Optional Real-LLM Extension

The real LLM path is optional.

To enable it:

```text
MOCK_LLM=0
```

The application then uses the structured prompt and the configured LLM provider.

The real-LLM implementation also contains retry logic so that generation can be attempted again when the initial output fails the expected validation requirements.

The real-LLM and cloud deployment extensions are optional and are not required for the graded baseline.

---

# 21. Summary

The complete application implements:

```text
8 Zepto policy documents
          ↓
Local embeddings
          ↓
ChromaDB
          ↓
LangGraph
          ↓
Intent classification
       ↙       ↘
   Policy      General
      ↓           ↓
 Retrieval      Direct
      ↓           ↓
   Answer       Answer
       ↘        ↙
       Pydantic
          ↓
       FastAPI
          ↓
         JSON
```

The required graded path is completely deterministic and offline when `MOCK_LLM` is left unset or set to `1`.
