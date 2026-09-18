import json
import os
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("MODULE 2: CHUNKING & VECTOR STORES")
print("=" * 80)

with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"\nLoaded {len(tickets)} support tickets")

print("\n" + "=" * 80)
print("PART 1: Chunking Strategies")
print("=" * 80)

documents = []
for ticket in tickets:
    full_text = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
    """.strip()

    doc = Document(
        page_content=full_text,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority'],
        },
    )
    documents.append(doc)

print(f"Created {len(documents)} documents")
print(f"\nSample document length: {len(documents[0].page_content)} characters")

print("\n--- Strategy 1: Fixed-Size Chunking ---")

fixed_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator="\n",
)
fixed_chunks = fixed_splitter.split_documents(documents)

print(f"✓ Created {len(fixed_chunks)} chunks")
print(f"  Chunk size: 200 chars, Overlap: 20 chars")
print(f"  Sample chunk: {fixed_chunks[0].page_content[:100]}...")

print("\n--- Strategy 2: Recursive Character Splitting ---")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)
recursive_chunks = recursive_splitter.split_documents(documents)

print(f"✓ Created {len(recursive_chunks)} chunks")
print(f"  Sample chunk: {recursive_chunks[0].page_content[:100]}...")

print("\n--- Strategy 3: Semantic Chunking ---")
print("  Note: Semantic chunking uses embeddings to detect topic shifts")

embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)

demo_paragraph = """
The Mars rover collected soil samples from the Jezero crater last week. Scientists believe these rocks may contain signs of ancient microbial life. NASA plans to retrieve these samples in a future mission. The discovery could reshape our understanding of life in the solar system.

Grandma's apple pie recipe starts with peeling six large Granny Smith apples. Mix flour, sugar, and cinnamon for the filling. Roll the dough thin and crimp the edges carefully. Bake at 375 degrees for 45 minutes until golden brown.

The defendant was charged with breach of contract under Section 12. The plaintiff seeks damages of fifty thousand dollars plus legal fees. Both parties agreed to mediation before proceeding to trial. The judge scheduled the preliminary hearing for next month.
"""

semantic_splitter = SemanticChunker(
    embeddings=embeddings_model,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.0,
)

demo_doc = Document(page_content=demo_paragraph.strip())
semantic_chunks = semantic_splitter.split_documents([demo_doc])
print(f"\n✓ Created {len(semantic_chunks)} chunks")

for i, chunk in enumerate(semantic_chunks):
    print(f"\n  Chunk {i + 1} ({len(chunk.page_content)} chars):")
    print("  " + "~" * 60)
    for line in chunk.page_content.strip().split('\n'):
        if line.strip():
            print(f"    {line.strip()}")
    print("  " + "~" * 60)

print("\n--- Strategy 4: Markdown Header Splitting ---")

markdown_doc = """
# Database Troubleshooting Guide

## Connection Issues

### Timeout Errors
If you encounter timeout errors, check the connection string and ensure the database server is reachable.
Increase the connection timeout value in your configuration.

### Authentication Failures
Verify your credentials are correct. Check for expired passwords or locked accounts.
Ensure the user has proper permissions on the database.

## Performance Problems

### Slow Queries
Analyze query execution plans using EXPLAIN.
Consider adding indexes on frequently queried columns.
Review and optimize JOIN operations.

### High CPU Usage
Monitor long-running queries.
Check for missing indexes causing table scans.
"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,
)
md_chunks = markdown_splitter.split_text(markdown_doc)

print(f"✓ Created {len(md_chunks)} chunks from markdown")
if md_chunks:
    print(f"  Sample chunk with metadata:")
    print(f"    Content: {md_chunks[0].page_content[:80]}...")
    print(f"    Metadata: {md_chunks[0].metadata}")

print("\n--- Strategy 5: HTML Header Splitting ---")

html_doc = """
<!DOCTYPE html>
<html>
<body>
    <h1>Email Configuration Guide</h1>
    
    <h2>SMTP Settings</h2>
    <p>Configure your SMTP server settings in the admin panel. Use port 587 for TLS or port 465 for SSL.</p>
    
    <h3>Common SMTP Servers</h3>
    <p>Gmail: smtp.gmail.com, Outlook: smtp.office365.com, Yahoo: smtp.mail.yahoo.com</p>
    
    <h2>IMAP Configuration</h2>
    <p>Set up IMAP to sync your emails across devices. Use port 993 for secure connections.</p>
    
    <h3>Folder Mapping</h3>
    <p>Map your email folders to the appropriate IMAP folders for proper synchronization.</p>
</body>
</html>
"""

headers_to_split_on_html = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on_html,
)
html_chunks = html_splitter.split_text(html_doc)

print(f"✓ Created {len(html_chunks)} chunks from HTML")
if html_chunks:
    print(f"  Sample chunk with metadata:")
    print(f"    Content: {html_chunks[0].page_content[:80]}...")
    print(f"    Metadata: {html_chunks[0].metadata}")

print("\n--- Strategy 6: Whole Documents (No Chunking) ---")
print(f"✓ Using {len(documents)} whole documents")
print(f"  Good for small documents like our tickets")

print("\n" + "=" * 80)
print("PART 2: Chroma Vector Store")
print("=" * 80)

embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)

query = "Authentication problems after password reset"

print("\nBuilding Chroma vector store...")

existing_store = Chroma(
    collection_name="support_tickets",
    persist_directory="./chroma_db",
)
existing_store.delete_collection()

chroma_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    collection_name="support_tickets",
    persist_directory="./chroma_db",
)
print("✓ Chroma store created and persisted")

print(f"\nSearching in Chroma: '{query}'")
chroma_results = chroma_store.similarity_search(query, k=3)

print(f"\nTop {len(chroma_results)} results:")
for i, doc in enumerate(chroma_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Category: {doc.metadata['category']}")

print("\n--- Using MMR for Diverse Results ---")
mmr_results = chroma_store.max_marginal_relevance_search(query, k=3)

print(f"\nMMR Results (more diverse):")
for i, doc in enumerate(mmr_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Title: {tickets[int(doc.metadata['ticket_id'].split('-')[1]) - 1]['title']}")

print("\n" + "=" * 80)
print("PART 3: Metadata Filtering")
print("=" * 80)

print("\nSearching only in 'Authentication' category:")
filtered_results = chroma_store.similarity_search(
    query,
    k=3,
    filter={"category": "Authentication"},
)

print(f"\nFiltered results ({len(filtered_results)}):")
for i, doc in enumerate(filtered_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Category: {doc.metadata['category']}")
    print(f"Content: {doc.page_content[:100]}...")

print("\n\nSearching only 'High' priority tickets:")
high_priority_results = chroma_store.similarity_search(
    "Database performance issues",
    k=3,
    filter={"priority": "High"},
)

print(f"\nHigh priority results ({len(high_priority_results)}):")
for i, doc in enumerate(high_priority_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Priority: {doc.metadata['priority']}")

print("\n" + "=" * 80)
print("PART 4: Evaluating Chunking Strategies")
print("=" * 80)

print("\nBuilding vector stores with different chunking strategies...")

store_whole = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    collection_name="whole_docs",
)

store_fixed = Chroma.from_documents(
    documents=fixed_chunks,
    embedding=embeddings_model,
    collection_name="fixed_chunks",
)

store_recursive = Chroma.from_documents(
    documents=recursive_chunks,
    embedding=embeddings_model,
    collection_name="recursive_chunks",
)

test_query = "Database connection failures"
print(f"\nTest query: '{test_query}'")

stores = [
    ("Whole Documents", store_whole),
    ("Fixed Chunks", store_fixed),
    ("Recursive Chunks", store_recursive),
]

for name, store in stores:
    results = store.similarity_search(test_query, k=1)
    print(f"\n{name}:")
    if results:
        print(f"  Top result: {results[0].page_content[:100]}...")
        print(f"  Length: {len(results[0].page_content)} chars")

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)
print("""
""")

