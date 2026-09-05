from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from documents import load_pdfs,PDF_FOLDER
from chunking import chunker

def build_vectorstore(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_db_path = "./chroma_db"
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory = persist_db_path)

    return db
if __name__ == "__main__":
    raw_documents = load_pdfs(PDF_FOLDER)
    chunks = chunker(raw_documents)
    db = build_vectorstore(chunks)
    print(f"Vector store created with {len(chunks)} chunks")