Architecture
RAG Pipeline
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