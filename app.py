from fastapi import FastAPI
from collections import defaultdict
import os

from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    deployment_name="gpt-4o",
    api_version="2025-01-01-preview",
    temperature=0
)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

prompt = ChatPromptTemplate.from_template("""
You are a Gapblue policy assistant.
Answer using the provided context.
The context may contain policy sections and rules.
Extract the most relevant information and answer clearly.
Specially note that: 
Return answer in bullet points using simple text like:
- point 1
- point 2
Do not use markdown symbols.                                       

If no relevant information exists, say "I don't know".
                                          
Chat History:
{history}

Context:
{context}

Question:
{question}

Answer:
""")

parser = StrOutputParser()

chat_memory = {}

def get_history(conversation_id):
    return chat_memory.get(conversation_id, [])

def update_history(conversation_id, question, answer):
    if conversation_id not in chat_memory:
        chat_memory[conversation_id] = []
    chat_memory[conversation_id].append((question, answer))


app = FastAPI()


@app.post("/chat")
def chat(query: str, conversation_id: str = "default"):

    
    results = vectordb.similarity_search_with_score(query, k=5)
    

    print("\n=== RETRIEVED ===")
    for r in results:
        print("Score:", r[1])
        print("Source:", r[0].metadata["source"])
        print("Page:", r[0].metadata["page"])
        print("Text:", r[0].page_content[:200])
        print("------")

    docs = [r[0] for r in results]

    context = "\n\n".join(d.page_content for d in docs)

    
    history_list = get_history(conversation_id)
    history_text = "\n".join(
        [f"Q: {q}\nA: {a}" for q, a in history_list]
    )

    # Chain (NO retriever inside)
    chain = prompt | llm | parser

    try:
        answer = chain.invoke({
            "context": context,
            "question": query,
            "history": history_text
        })
    except Exception as e:
        return {
            "answer": "I don't know",
            "sources": [],
            "confidence": "low",
            "error": str(e)
        }

    update_history(conversation_id, query, answer)

    
    sources = [
        {
            "document": r[0].metadata["source"],
            "page": r[0].metadata["page"],
            "relevance_score": round(r[1], 3)
        }
        for r in results
    ]

    confidence = "high" if len(docs) >= 2 else "low"

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }


@app.get("/sources")
def sources():
    all_data = vectordb.get()

    counts = defaultdict(int)

    for meta in all_data["metadatas"]:
        counts[meta["source"]] += 1

    return dict(counts)