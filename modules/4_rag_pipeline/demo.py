# -*- coding: utf-8 -*-
"""
Building the Complete RAG Pipeline Demo
================================================

This demo teaches:
1. Complete RAG architecture: retrieve → inject → generate
2. LangChain components (retrievers, prompts, chains)
3. Anti-hallucination strategies
4. Building a production-ready Q&A system

LEARNING RESOURCES:
- RAG Paper (Lewis et al.): https://arxiv.org/abs/2005.11401
- LangChain Documentation: https://python.langchain.com/docs/get_started/introduction
- LCEL Guide: https://python.langchain.com/docs/expression_language/
- Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
- Chroma Vector DB: https://docs.trychroma.com/
"""

import json
import os
import time
from operator import itemgetter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("MODULE 4: RAG PIPELINE")
print("=" * 80)
print(
    """
STRUCTURE:

  PART 1: Data Ingestion & Vector Store
  PART 2: Retriever Setup
  PART 3: Prompt Engineering (Anti-Hallucination)
  PART 4: Language Model
  PART 5: LCEL Chain Assembly
  PART 6: Testing the RAG System
  PART 7: Validation & Fallback
  PART 8: Conversation with History (Multi-Turn RAG)
  PART 9: Interactive Demo
"""
)

print("\n" + "=" * 80)
print("PART 1: Data Ingestion Pipeline")
print("=" * 80)

with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"✓ Loaded {len(tickets)} support tickets")

documents = []
for ticket in tickets:
    content = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Date: {ticket['created_date']} to {ticket['resolved_date']}

Problem Description:
{ticket['description']}

Resolution:
{ticket['resolution']}
    """.strip()

    doc = Document(
        page_content=content,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'title': ticket['title'],
            'category': ticket['category'],
            'priority': ticket['priority'],
            'source': f"Ticket {ticket['ticket_id']}",
        },
    )
    documents.append(doc)

print(f"✓ Created {len(documents)} documents with metadata")

print("\nInitializing OpenAI embedding model...")
embeddings = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)
print("✓ OpenAI embedding model ready")

print("\nBuilding Chroma vector store...")
import shutil

persist_directory = "./rag_vectorstore"
if os.path.exists(persist_directory):
    try:
        shutil.rmtree(persist_directory)
    except PermissionError:
        persist_directory = f"./rag_vectorstore_{int(time.time())}"
        print(f"⚠ Vector store directory is locked; using {persist_directory} instead")
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="supportdesk_rag",
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"},
)
print("✓ Vector store created and persisted")

print("\n" + "=" * 80)
print("PART 2: Setting Up Retriever")
print("=" * 80)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

print("✓ Retriever configured:")
print(f"  - Search type: similarity")
print(f"  - Top-K results: 3")
print("\nTIP: k=3-5 is usually optimal. Too few → missing context, too many → noise")

test_query = "Users can't log in after changing passwords"
print(f"\nTest query: '{test_query}'")
retrieved_docs = retriever.invoke(test_query)

print(f"\nRetrieved {len(retrieved_docs)} documents:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n#{i} - {doc.metadata['ticket_id']}: {doc.metadata['title']}")
    print(f"  Category: {doc.metadata['category']}")

print("\n" + "=" * 80)
print("PART 3: Prompt Engineering for RAG")
print("=" * 80)

prompt_template = """You are SupportDesk AI, a technical support assistant that helps engineers troubleshoot issues using historical support ticket data.

CRITICAL RULES:
1. Answer using ONLY information from the provided context.
2. If the question is broad or underspecified, provide the best matching known issue(s) from context and state any assumptions.
3. If context is partially relevant, still provide the most likely troubleshooting guidance from relevant tickets.
4. If the answer is truly not present in context, say "I don't have enough information in the ticket history to answer that question."
5. DO NOT make up information or use external knowledge.
6. Always cite ticket IDs for every issue/resolution you mention.
7. If multiple tickets are relevant, summarize each briefly.

Context from support tickets:
{context}

Question: {question}

Helpful Answer (with ticket citations):"""

PROMPT = ChatPromptTemplate.from_template(prompt_template)

print("✓ Prompt template created with anti-hallucination rules:")
print("\n" + "-" * 80)
print(prompt_template)
print("-" * 80)

print("\n" + "=" * 80)
print("PART 4: Initializing Language Model")
print("=" * 80)

if os.getenv("OPENAI_API_KEY"):
    print("✓ OpenAI API key found")
    llm = ChatOpenAI(
        model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
        temperature=0,
        timeout=120,
        max_retries=3,
    )
    print(f"✓ Using {os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')}")
else:
    print("⚠ OpenAI API key not found!")
    print("  Please set OPENAI_API_KEY environment variable")
    print("  Or use Ollama: ollama pull llama2")
    print("\nFor this demo, we'll show the prompt without generating answers.")
    llm = None

print("\n" + "=" * 80)
print("PART 5: Assembling RAG Chain")
print("=" * 80)

def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

if llm:
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    print("✓ RAG chain assembled:")
    print("  Retriever → Context Injection → LLM → Answer")
    print("\nThis is the complete RAG pipeline! Query in → Answer out")
else:
    qa_chain = None
    print("⚠ LLM not available, showing architecture only")

print("\n" + "=" * 80)
print("PART 6: Testing the RAG System")
print("=" * 80)

test_queries = [
    "How do I fix authentication failures after password reset?",
    "What causes database connection timeouts?",
    "Why are emails not being delivered?",
    "How do I make the perfect pizza?",
]

for query in test_queries:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    docs = retriever.invoke(query)
    print(f"\nRetrieved {len(docs)} relevant tickets:")
    for i, doc in enumerate(docs, 1):
        print(f"\n  [{i}] {doc.metadata['ticket_id']}: {doc.metadata['title']}")

    if qa_chain:
        print("\nGenerating answer...")
        result = qa_chain.invoke(query)

        print("\n" + "-" * 80)
        print("ANSWER:")
        print("-" * 80)
        print(result)

        print("\n" + "-" * 80)
        print("SOURCE DOCUMENTS:")
        print("-" * 80)
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.metadata['source']}")
    else:
        print("\n(LLM not configured - would generate answer here)")

print("\n" + "=" * 80)
print("PART 7: Enhanced RAG with Answer Validation")
print("=" * 80)

def rag_with_validation(query, retriever, llm, min_similarity_score=0.5):
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=3)

    print(f"\nQuery: {query}")
    print(f"\nRelevance scores (cosine similarity: 0=no match, 1=identical):")
    for doc, score in docs_with_scores:
        print(f"  - {doc.metadata['ticket_id']}: {score:.4f}")

    best_score = docs_with_scores[0][1]

    if best_score < min_similarity_score:
        print(f"\n⚠ Best match relevance ({best_score:.4f}) is below threshold ({min_similarity_score}) — too dissimilar to answer confidently")
        return "I don't have enough relevant information in the ticket history to answer that question confidently."

    docs = [doc for doc, score in docs_with_scores]
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    prompt = f"""{prompt_template.replace('{context}', context).replace('{question}', query)}"""

    if llm:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    return "(LLM not configured)"

print("\nTesting validation logic:")
print("\n1. Relevant query (should answer):")
rag_with_validation(
    "How to fix database connection timeouts?",
    retriever,
    llm,
    min_similarity_score=0.5,
)

print("\n2. Irrelevant query (should refuse):")
rag_with_validation(
    "What is the capital of France?",
    retriever,
    llm,
    min_similarity_score=0.5,
)

print("\n" + "=" * 80)
print("PART 8: Conversation with History (Multi-Turn RAG)")
print("=" * 80)
print(
    """
Problem with single-turn RAG:
  Turn 1: "How do I fix authentication failures?"  → good answer
  Turn 2: "How long did it take to resolve?"       → loses context! "it" = ???

Solution: Two-part fix:
  1. MessagesPlaceholder injects prior HumanMessage / AIMessage objects into the prompt
     so the LLM can understand references like "that issue" or "it".
  2. Query reformulation rewrites follow-up questions into standalone queries
     BEFORE retrieval, so the retriever searches for the right documents.
     e.g. "How do I fix it?" + history → "How do I fix authentication failures?"
"""
)

if llm:
    condense_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given the chat history and a follow-up question, rephrase the "
            "follow-up as a standalone question that includes all necessary "
            "context from the history. If the question is already standalone, "
            "return it unchanged.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    condense_chain = condense_prompt | llm | StrOutputParser()

    conv_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are SupportDesk AI. Answer using the ticket context below and the chat history.
Use chat history to resolve references like "that issue" or "that ticket".
For factual claims, prioritize the retrieved context.
If information is not available in context or history, say "I don't have that information."
Always cite ticket IDs when available.

Context:
{context}""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def ask_with_history(question, history):
        if not history:
            standalone = question
        else:
            standalone = condense_chain.invoke({
                "question": question,
                "chat_history": history,
            })

        context = format_docs(retriever.invoke(standalone))
        answer = (conv_prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "chat_history": history,
            "question": question,
        })

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        return answer

    print("Multi-turn conversation demo:\n")
    history = []

    q1 = "How do I fix authentication failures after a password reset?"
    print(f"Turn 1 — User: {q1}")
    a1 = ask_with_history(q1, history)
    print(f"         Assistant: {a1[:300]}{'...' if len(a1) > 300 else ''}")
    print(f"         [chat_history now has {len(history)} messages]")

    q2 = "What was the ticket ID for that issue?"
    print(f"\nTurn 2 — User: {q2}")
    a2 = ask_with_history(q2, history)
    print(f"         Assistant: {a2[:300]}{'...' if len(a2) > 300 else ''}")
    print(f"         [chat_history now has {len(history)} messages]")

    q3 = "What was the resolution for that ticket?"
    print(f"\nTurn 3 — User: {q3}")
    a3 = ask_with_history(q3, history)
    print(f"         Assistant: {a3[:300]}{'...' if len(a3) > 300 else ''}")
    print(f"         [chat_history now has {len(history)} messages]")

    print(f"\n✓ History contains {len(history)} messages ({len(history) // 2} complete turns)")
    print("TIP: In production, cap history to avoid token bloat:")
    print("       history = history[-6:]  # keep last 3 turns")
else:
    print("(LLM not configured — would run multi-turn conversation here)")
    print("\nKey pattern:\n")
    print("  history = []")
    print()
    print("  if history:")
    print("      standalone = condense_chain.invoke({'question': q, 'chat_history': history})")
    print("  else:")
    print("      standalone = q")
    print()
    print("  context = format_docs(retriever.invoke(standalone))")
    print("  answer = (conv_prompt | llm | StrOutputParser()).invoke({")
    print("      'context': context, 'chat_history': history, 'question': q")
    print("  })")
    print()
    print("  history.append(HumanMessage(content=q))")
    print("  history.append(AIMessage(content=answer))")
    print()
    print("Query reformulation ensures the retriever gets meaningful queries")
    print("instead of vague pronouns like 'it' or 'that ticket'.")

print("\n" + "=" * 80)
print("PART 9: Interactive SupportDesk Assistant")
print("=" * 80)

if qa_chain:
    print("\nSupportDesk RAG Assistant Ready!")
    print("Ask questions about support ticket history.")
    print("Type 'quit' to exit.\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_query:
            continue

        print("\nAssistant: ", end="")
        answer = qa_chain.invoke(user_query)
        print(answer)

        docs = retriever.invoke(user_query)
        print(f"\n📎 Sources: {', '.join([doc.metadata['ticket_id'] for doc in docs])}")
        print()
else:
    print("\n⚠ Interactive mode requires OpenAI API key")
    print("Set OPENAI_API_KEY to try the interactive assistant!")

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)
print("\nKey Takeaways:")
print("1. RAG pipeline: Retrieve → Inject Context → Generate")
print("2. Strict prompt engineering prevents hallucinations")
print("3. Always return source documents for verification")
print("4. Implement fallbacks for low-confidence matches")
print("5. Temperature=0 for deterministic, grounded answers")
print("\nNext: Evaluation & Metrics")
