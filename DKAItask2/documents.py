import os
from pypdf import PdfReader
from langchain_core.documents import Document

PDF_FOLDER = r"C:\Users\HP\Desktop\document"

def load_pdfs(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        print(f"error: the path {folder_path} does not exist")
        return documents
    
    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue

        file_path  = os.path.join(folder_path,filename)
        try:
            reader = PdfReader(file_path)


            
            for page_num,page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    doc = Document(
                        page_content = text,
                        metadata = {
                            "source": filename,
                            "page": page_num+1
                        }
                    )
                    documents.append(doc)
        except Exception as e:
            print(f"error redaing {filename} : {e}")
    return documents


if __name__ == "__main__":
    raw_documents = load_pdfs(PDF_FOLDER)
    print(f"successfully loaded {len(raw_documents)} pages across your pdf's")