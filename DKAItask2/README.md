# Local PDF Retrieval-Augmented Generation (RAG) System

A robust, production-ready local Retrieval-Augmented Generation (RAG) system built using **LangChain**, **ChromaDB**, and **Groq LLMs**. This pipeline targets a local folder containing 10-20 PDF documents, parses and chunks their semantic contents, indexes them within a persistent on-disk vector store cache, and exposes a command-line boundary interface tracking source paths and exact document pagination.

---

## 🌟 Key Functional Features

- **On-Disk Local Vector Caching:** Evaluates storage directories on program boot. Completely bypasses parsing pipelines if index states match, ensuring near-instantaneous query availability on hot reload.
- **Native Document Structuring:** Extracts data stream segments page by page utilizing `pypdf`, preserving structural layouts layout markers.
- **Granular Pagination Mapping:** Tracks source metadata footprints through processing operations, surfacing source names and precise page references during lookup evaluation.
- **Context-Safe Splitting Boundaries:** Utilizes a custom-configured `RecursiveCharacterTextSplitter` algorithm to balance structural text window lengths with overlapping semantic blocks.
- **High-Velocity Completion Generation:** Leverages dedicated inference endpoints powered via the Groq platform runtime architecture.

---

## 📂 System Module Architecture

```text
├── documents.py      # Scans the target workspace folder, extracts text pages via pypdf, and yields Document objects.
├── chunking.py       # Accepts document lists and divides characters into structured semantic shards.
├── vectorstore.py    # Manages disk folder pathways and coordinates initialization patterns for local DB storage.
├── retriever.py      # Executes mathematical cosine/similarity sorting routines matching target context blocks.
├── generator.py      # Handles prompt injection guardrails and processes request strings against the Groq model API.
├── main.py           # Core orchestrator script that controls startup checks, data validation, and CLI print layouts.
└── main.ipynb        # Fully converted Jupyter Notebook orchestration counterpart meeting task notebook delivery criteria.
```

---

## 🛠️ Required Dependencies & Frameworks

Ensure your runtime environment maps to **Python 3.10+**. The architecture depends on the following libraries:

```bash
pip install langchain langchain-community langchain-core langchain-chroma langchain-huggingface langchain-groq pypdf python-dotenv jupyter
```

---

## 🚀 Step-by-Step Setup & Deployment

### 1. Work Space Placement
Verify that all execution script modules (`documents.py`, `chunking.py`, `vectorstore.py`, `retriever.py`, `generator.py`, and `main.ipynb`) are saved together inside a flat root folder layout directory.

### 2. Configuration Matrix (`.env`)
Create an environment profile wrapper named `.env` right within your project root root file hierarchy and assign your authentication token credentials securely:

```env
GROQ_API_KEY=gsk_your_actual_secret_groq_api_key_goes_here
```

### 3. Populating Input Resources
Create the system absolute directory target variable path defined within your workspace rules inside `documents.py` (e.g., `C:\Users\HP\Desktop\document`) and paste your 10 to 20 reference manuals, textbooks, or guidelines within it.

---

## 💻 Running the Application

### Method A: Native Terminal Shell
Execute the central operational main pipeline through your command terminal terminal interface:
```bash
python main.py
```

### Method B: Jupyter Lab Notebook Workflow
If executing task steps through the structured notebook wrapper module:
1. Initialize the interactive service engine: `jupyter notebook`
2. Open up the `main.ipynb` asset block.
3. Click the **"Run All Cells"** command inside the primary top header control ribbon.

---

## ⚙️ Core Processing Flow Sequence

```text
[ Start Program ]
        │
        ▼
 Check Local DB Path? 
        ├──► (Folder Exists)  ──► [ Load Persistent Storage Caches Instantly ] ──┐
        │                                                                        │
        └──► (Folder Missing) ──► [ Call documents.py -> Ingest Folder PDFs ]     │
                                        │                                        │
                                        ▼                                        ▼
                                  [ Call chunking.py -> Split Overlap Chunks ] ──► [ Await User Question ]
                                        │                                                  │
                                        ▼                                                  ▼
                                  [ Embed Data and Save chroma_db Folder ]           [ Run Query Pipeline ]
```

1. **Scan Evaluation Phase:** The orchestrator inspects the hard drive layout directory tracking for an explicit `./chroma_db` folder structure.
2. **Dynamic Ingestion Routing:**
   - **Cache Match:** Instantly accesses persistent local database state allocations, slicing processing overhead completely.
   - **Fresh Compilation:** Loops through targets, builds vector representations via HuggingFace's model engine (`all-MiniLM-L6-v2`), and logs context indices down safely.
3. **Conversational Interface:** Presents a recurring user capture loop string: `Ask a question: `
4. **Response Compilation:** Matches structural matrix context coordinates, compiles relevant source lines, triggers the underlying LLM provider, and maps explicit citation lines indicating source filenames alongside exact page numbers below the generation section.