from langchain_text_splitters import RecursiveCharacterTextSplitter
from documents import load_pdfs,PDF_FOLDER

def chunker(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    raw_documents = load_pdfs(PDF_FOLDER)
    chunks = chunker(raw_documents)
    print(f"Total chunks: {len(chunks)}")
    print("\nExample chunk:\n", chunks[0])