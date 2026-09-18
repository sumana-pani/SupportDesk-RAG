import json
import os
import httpx
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex,
    Document,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

os.environ["HTTPX_TIMEOUT"] = "300"

Settings.embed_model = OpenAIEmbedding(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=120,
    max_retries=5,
)
Settings.llm = OpenAI(
    model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=300,
    max_retries=5,
)

print("=" * 80)
print("MODULE 3: INDEXING STRATEGIES FOR RAG")
print("=" * 80)
print("\nThis demo compares 5 different indexing approaches:")
print("1. Vector Index - Semantic similarity search")
print("2. Summary Index - Search through full documents with LLM")
print("3. Tree Index - Hierarchical retrieval")
print("4. Keyword Table Index - Traditional keyword matching")
print("5. Hybrid Retrieval - Combine multiple strategies")

print("\n" + "=" * 80)
print("Loading Support Tickets")
print("=" * 80)

with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

documents = []
for ticket in tickets:
    content = f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}"""

    doc = Document(
        text=content,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority'],
            'title': ticket['title'],
        },
    )
    documents.append(doc)

print(f"✓ Loaded {len(documents)} support tickets")

query = "How do I fix authentication issues after password reset?"
print(f"\nTest Query: '{query}'")

print("\n" + "=" * 80)
print("PART 1: Vector Index (Flat Index)")
print("=" * 80)

vector_index = VectorStoreIndex.from_documents(documents)
vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)

print("✓ Created vector index")
print(f"\nQuery: '{query}'")
vector_response = vector_query_engine.query(query)

print("\nVector Index Results:")
print(f"Answer: {vector_response.response}\n")
print("Source Documents:")
for i, node in enumerate(vector_response.source_nodes, 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   Score: {node.score:.4f}")
    print(f"   {node.text[:150]}...")

print("\n" + "=" * 80)
print("PART 2: Summary Index")
print("=" * 80)

summary_index = SummaryIndex.from_documents(documents)
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")

print("✓ Created summary index")
print(f"\nQuery: '{query}'")
summary_response = summary_query_engine.query(query)

print("\nSummary Index Results:")
print(f"Answer: {summary_response.response}\n")
print("Source Documents:")
for i, node in enumerate(summary_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

print("\n" + "=" * 80)
print("PART 3: Tree Index (Hierarchical Retrieval)")
print("=" * 80)

print("\nTree indexing builds a hierarchical structure from leaf to root.")
print("Using the full document set, which may take time to build.\n")

tree_index = TreeIndex.from_documents(documents)
tree_query_engine = tree_index.as_query_engine(child_branch_factor=2)

print("✓ Created tree index with hierarchical structure")
print(f"\nQuery: '{query}'")
tree_response = tree_query_engine.query(query)

print("\nTree Index Results:")
print(f"Answer: {tree_response.response}\n")
print("Source Documents:")
for i, node in enumerate(tree_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

print("\n" + "=" * 80)
print("PART 4: Keyword Table Index")
print("=" * 80)

keyword_index = KeywordTableIndex.from_documents(documents)
keyword_table = keyword_index.index_struct.table
print(f"\n✓ Extracted {len(keyword_table)} unique keywords:")
for keyword, node_ids in sorted(keyword_table.items()):
    print(f"  '{keyword}' → {len(node_ids)} document(s)")

keyword_query_engine = keyword_index.as_query_engine()

print("\n✓ Created keyword table index")
print(f"\nQuery: '{query}'")
keyword_response = keyword_query_engine.query(query)

print("\nKeyword Index Results:")
print(f"Answer: {keyword_response.response}\n")
print("Source Documents:")
for i, node in enumerate(keyword_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

print("\n" + "=" * 80)
print("PART 5: Hybrid Retrieval")
print("=" * 80)

vector_nodes = vector_index.as_retriever(similarity_top_k=5).retrieve(query)
keyword_nodes = keyword_index.as_retriever().retrieve(query)

seen_ids = set()
hybrid_nodes = []

for node in vector_nodes + keyword_nodes:
    node_id = node.metadata.get('ticket_id', node.node_id)
    if node_id not in seen_ids:
        seen_ids.add(node_id)
        hybrid_nodes.append(node)

print("✓ Using Vector + Keyword hybrid approach")
print(f"\nQuery: '{query}'")
print("\nHybrid Retrieval Results (Combined):")
for i, node in enumerate(hybrid_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    if hasattr(node, 'score') and node.score:
        print(f"   Score: {node.score:.4f}")
    print(f"   {node.text[:150]}...")

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

print("""
Vector Index: Best default semantic search
Summary Index: Good for small collections, slower at scale
Tree Index: Good for large hierarchical data
Keyword Index: Best for exact matches and IDs
Hybrid Retrieval: Best production choice
""")

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)

print("""
Key Takeaways:
1. Vector Index is the default choice.
2. Keyword Index is a useful complement.
3. Hybrid retrieval gives the best accuracy.
4. Tree Index is useful for large or structured collections.
5. Measurable evaluation is the next step.
""")
