import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re

def clean_text(text):
    
    text = re.sub(r'\b(?:[A-Za-z]\s){2,}[A-Za-z]\b',
                  lambda m: m.group(0).replace(" ", ""),
                  text)

   
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

DATA_PATH = "data/"
DB_PATH = "db/"


def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            pages = loader.load()

            for p in pages:
                p.metadata["source"] = file

               
                p.page_content = clean_text(p.page_content)

            docs.extend(pages)

    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

def enrich_metadata(chunks):
    for c in chunks:
        c.metadata["page"] = c.metadata.get("page", 0)
        c.metadata["category"] = c.metadata["source"].split("_")[0]
    return chunks

def create_db(chunks):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=DB_PATH
    )
    vectordb.persist()

if __name__ == "__main__":
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    docs = load_documents()
    chunks = split_docs(docs)
    chunks = enrich_metadata(chunks)
    create_db(chunks)

    print("Indexing complete!")