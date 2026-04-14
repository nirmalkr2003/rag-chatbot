from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(folder):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            pages = loader.load()
            for p in pages:
                p.metadata["source"] = file
            docs.extend(pages)
    return docs