from fastapi import FastAPI
from collections import defaultdict

from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



llm = AzureChatOpenAI(
    azure_endpoint="YOUR_ENDPOINT",
    api_key="YOUR_API_KEY",
    deployment_name="YOUR_DEPLOYMENT",
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

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

prompt = ChatPromptTemplate.from_template("""
You are a Gapblue policy assistant.

Answer ONLY from the provided context.
If the answer is not found, say "I don't know".

Chat History:
{history}

Context:
{context}

Question:
{question}

Answer:
""")


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

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
    
    
    docs = retriever.get_relevant_documents(query)

   
    history_list = get_history(conversation_id)
    history_text = "\n".join(
        [f"Q: {q}\nA: {a}" for q, a in history_list]
    )

    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "history": lambda x: history_text
        }
        | prompt
        | llm
        | StrOutputParser()
    )

   
    answer = rag_chain.invoke(query)

    
    update_history(conversation_id, query, answer)

    
    sources = [
        {
            "document": d.metadata["source"],
            "page": d.metadata["page"],
            "relevance_score": "N/A"
        }
        for d in docs
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
        doc_name = meta["source"]
        counts[doc_name] += 1

    return dict(counts)