import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from documents import load_pdfs,PDF_FOLDER
from chunking import chunker
from vectorstore import build_vectorstore
from retriever import retrieve
from generator import generate_answer

K = 3
persist_db_path = "./chroma_db"

def build_pipeline():
    if os.path.exists(persist_db_path):
        print("vector store found")
        embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        db = Chroma(
            persist_directory=persist_db_path,
            embedding_function=embedding_model
        )
        return db
    
    print("vector store not found. building new")
    raw_documents = load_pdfs(PDF_FOLDER)
    print(f"[1] Loaded {len(raw_documents)} PDF(s)")

    chunks = chunker(raw_documents)
    print(f"[2] Split into {len(chunks)} chunks")

    db = build_vectorstore(chunks)
    print(f"[3] Vector store built with {len(chunks)} embedded chunks")

    return db

def answer_query(db, query):
    docs = retrieve(db, query, k=K)
    print(f"[4] Retrieved top {len(docs)} chunks")

    context = "\n\n".join([doc.page_content for doc in docs])
    answer = generate_answer(context, query)
    print(f"[5] Answer generated")

    return answer, docs

def main():
    print("Initializing system...\n")
    db = build_pipeline()
    print("\nReady. Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        answer, sources = answer_query(db, query)

        print("\n--- ANSWER ---")
        print(answer)

        print("\n--- SOURCES USED ---")
        for i, doc in enumerate(sources):
            source_name = doc.metadata.get("source","unknown document")
            page_num = doc.metadata.get("page","N/A")
            print(f"[{i+1}] file: {source_name} | page: {page_num}")
        print("-" * 20 + "\n")

if __name__ == "__main__":
    main()