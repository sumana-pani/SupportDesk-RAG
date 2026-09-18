# SupportDesk-RAG

### AI-Powered Support Ticket Retrieval & Troubleshooting Assistant

SupportDesk-RAG is an end-to-end **Retrieval-Augmented Generation (RAG)** application that uses semantic search and large language models to retrieve relevant historical support tickets and generate grounded troubleshooting responses.

The project demonstrates the core components required to build a production-oriented RAG system, including **document ingestion, chunking, embeddings, vector search, retrieval evaluation, grounded generation, and agentic workflows**.

---

## Architecture

### RAG Pipeline

```text
                         ┌──────────────────────┐
                         │  Support Tickets     │
                         │  / Knowledge Base    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Document Ingestion  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Chunking        │
                         │ Fixed / Recursive /  │
                         │ Semantic Strategies  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Embedding Model    │
                         │ text-embedding-3-    │
                         │ small / large        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Vector Store      │
                         │       Chroma         │
                         └──────────┬───────────┘
                                    │
                                    │
                ┌───────────────────┘
                │
                ▼
        ┌──────────────────┐
        │   User Query     │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Query Embedding  │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Semantic Search  │
        │    / Retrieval   │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Relevant Tickets │
        │    / Context     │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Prompt + Context │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │       LLM        │
        │    GPT-4o-mini   │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Grounded Answer  │
        └──────────────────┘
```

---

## Key Features

### 1. Semantic Embeddings

Support tickets are converted into numerical vector representations using OpenAI embedding models.

This enables semantic matching between a user's issue and historically similar tickets even when the wording is different.

Example:

```text
Query:
"Users are unable to login after changing their password."

Retrieved ticket:
"Authentication fails following credential reset."
```

Keyword matching may consider these different, while embedding-based retrieval can identify their semantic similarity.

---

### 2. Multiple Chunking Strategies

The project explores different approaches to breaking documents into retrieval-friendly chunks:

* Fixed-size chunking
* Recursive chunking
* Semantic chunking
* Structure-aware chunking

Chunk size and overlap can significantly affect retrieval quality, so the project provides a way to experiment with these strategies.

---

### 3. Vector Search

Embedded support-ticket chunks are stored in a vector store and retrieved using semantic similarity.

The retrieval flow is:

```text
User Query
    ↓
Query Embedding
    ↓
Vector Similarity Search
    ↓
Top-K Relevant Chunks
```

---

### 4. Multiple Retrieval / Indexing Strategies

The project compares different approaches to information retrieval, including:

* Vector / semantic retrieval
* Keyword-based retrieval
* Summary-based retrieval
* Hierarchical retrieval
* Hybrid retrieval

This provides a practical comparison between semantic and traditional information retrieval techniques.

---

### 5. RAG-Based Response Generation

Retrieved support-ticket context is passed to the LLM along with the user's question.

```text
User Question
      +
Retrieved Context
      ↓
Prompt Construction
      ↓
LLM
      ↓
Grounded Response
```

The system is designed to answer using retrieved evidence rather than relying solely on the model's internal knowledge.

---

### 6. Anti-Hallucination / Grounding

The RAG pipeline includes safeguards intended to reduce unsupported responses.

The system is designed to:

* Ground answers in retrieved ticket context
* Avoid inventing troubleshooting information
* Return an appropriate response when relevant context cannot be retrieved
* Separate retrieved evidence from generated reasoning

---

### 7. RAG Evaluation

The project evaluates the pipeline at two levels.

#### Retrieval Evaluation

Measures whether the correct information was retrieved.

Metrics include:

```text
Precision@K
Recall@K
F1
```

#### Generation Evaluation

Measures the quality of the generated response.

Examples include:

```text
Groundedness
Completeness
```

The project also demonstrates **LLM-as-a-judge** evaluation for generated responses.

---

### 8. Agentic RAG

The project includes an agentic RAG workflow where an LLM can select and invoke tools based on the user's request.

Conceptually:

```text
                     User Query
                          │
                          ▼
                       Agent
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          Search       Retrieval     Other
           Tool          Tool         Tools
             │            │            │
             └────────────┼────────────┘
                          ▼
                         LLM
                          │
                          ▼
                       Answer
```

This demonstrates the difference between a deterministic RAG pipeline and a tool-using agentic workflow.

---

## Technology Stack

| Component            | Technology                      |
| -------------------- | ------------------------------- |
| Language             | Python 3.12                     |
| LLM                  | OpenAI GPT-4o-mini              |
| Embeddings           | OpenAI `text-embedding-3-small` |
| RAG Framework        | LangChain                       |
| Indexing / Retrieval | LlamaIndex                      |
| Vector Store         | Chroma                          |
| Evaluation           | FAISS + LLM-as-a-Judge          |
| Configuration        | Environment variables           |


---

# Getting Started

## Prerequisites

* Python 3.12+
* OpenAI API key
* Basic Python knowledge

> The current dependency set is tested with Python 3.12. If you encounter compatibility issues with Chroma/Pydantic on newer Python versions, use Python 3.12.

---

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd SupportDesk-RAG
```

---

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Configure the OpenAI API

Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Configure:

```env
OPENAI_API_KEY=your_api_key_here

OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

**Never commit `.env` or API keys to Git.**

---

# Running the Project

## Embeddings

```bash
cd modules/1_embeddings
python demo.py
```

Demonstrates:

* Embedding generation
* Semantic similarity
* Vector representations

---

## Chunking

```bash
cd modules/2_chunking
python demo.py
```

Demonstrates:

* Fixed-size chunking
* Recursive chunking
* Semantic chunking
* Structure-aware splitting

---

## Indexing

```bash
cd modules/3_indexing
python demo.py
```

Demonstrates different indexing and retrieval approaches.

---

## RAG Pipeline

```bash
cd modules/4_rag_pipeline
python demo.py
```

Demonstrates the complete:

```text
Query
 ↓
Retrieval
 ↓
Context
 ↓
Prompt
 ↓
LLM
 ↓
Response
```

pipeline.

---

## Evaluation

```bash
cd modules/5_evaluation
python demo.py
```

Evaluates retrieval and generated responses using retrieval metrics and LLM-based evaluation.

---

## Agentic RAG

```bash
cd modules/6_agentic_rag
python demo.py
```

Demonstrates tool-based retrieval and agentic workflows.

---

# Design Considerations

This project focuses on several practical RAG engineering trade-offs.

### Chunk Size

Larger chunks provide more context but may reduce retrieval precision and increase LLM token usage.

Smaller chunks improve retrieval granularity but can lose important surrounding context.

---

### Top-K Retrieval

Increasing `K` provides more context to the LLM but also:

* increases prompt size
* increases latency
* increases token cost
* may introduce irrelevant information

Therefore, retrieval quality should be evaluated rather than choosing `K` arbitrarily.

---

### Semantic vs Keyword Search

Semantic search handles differences in wording and meaning.

Keyword search can perform better when exact identifiers, ticket IDs, error messages, or technical terms are important.

A production system may therefore benefit from **hybrid retrieval**.

---

### Retrieval vs Generation Quality

A poor answer does not necessarily mean the LLM is the problem.

The failure could occur earlier:

```text
Bad Answer
   │
   ├── Poor Retrieval
   │
   ├── Incorrect Context
   │
   ├── Poor Prompt
   │
   └── LLM Generation
```

This is why the project evaluates retrieval and generation separately.
