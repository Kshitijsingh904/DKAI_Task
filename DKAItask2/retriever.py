import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from documents import load_pdfs,PDF_FOLDER
from chunking import chunker
from vectorstore import build_vectorstore

def retrieve(db, query, k=3):
    return db.similarity_search(query, k=k)

if __name__ == "__main__":
    persist_db_path = "./chroma_db"
    embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

    if os.path.exists(persist_db_path):
        print("vector store exist")
        db = Chroma(
            persist_directory=persist_db_path,
            embedding_function=embedding_model
        )
    else:
        print("Vector store not found. Building vector store")
        raw_documents = load_pdfs(PDF_FOLDER)
        chunks = chunker(raw_documents)
        db = build_vectorstore(chunks)


    query = "What is insulin resistance?"
    results = retrieve(db, query)

    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(doc.page_content)
        print()