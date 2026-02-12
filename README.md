# Pravaah Legal AI

**Pravaah** is a comprehensive legal technology solution designed to bridge the gap between complex legal statutes and actionable insights. Built for both legal professionals and everyday citizens, Pravaah utilizes Retrieval-Augmented Generation (RAG) to provide accurate summaries, advisory analysis, and multilingual support for a wide array of Indian Acts and legal documents.

---

## 🚀 Features

* **Dual-Mode Intelligence:**
* **Human Mode:** Simplifies legal jargon into everyday language for citizens.
* **Professional Mode:** Provides technical, IRAC-structured (Issue, Rule, Application, Conclusion) analysis for lawyers.


* **Multilingual Support:** Seamlessly translates legal insights into Hindi, Kannada, Tamil, Malayalam, and Telugu.
* **Voice Integration:** Generates high-quality audio outputs of legal advice using Edge TTS.
* **Advanced OCR:** Intelligently processes digital PDFs, scanned documents, and images using Tesseract and PyMuPDF.
* **RAG-Powered:** Anchors all AI responses to a verified database of 20+ Indian Acts (BNS, Companies Act, Patent Act, etc.) to prevent hallucinations.

---

## 🏗️ Project Structure

```text
├── backend/
│   ├── PDFS/                # Knowledge base (Indian Acts & Statutes)
│   ├── routers/             # FastAPI endpoints (Human & Professional logic)
│   ├── chroma_langchain_db/ # Vector database for document retrieval
│   ├── llm.py               # RAG logic and prompt engineering
│   ├── vector.py            # Document processing and embedding pipeline
│   ├── ocr_processor.py     # OCR and PDF text extraction
│   └── multilingual.py      # Translation and TTS engine
├── frontend/
│   └── UI/                  # User Interface components
├── Dockerfile               # Containerization configuration
└── pyproject.toml           # Python dependencies

```

---

## 🛠️ Tech Stack

**Backend:**

* **Framework:** FastAPI
* **Orchestration:** LangChain
* **LLM:** Google Gemini 2.0 Flash
* **Vector DB:** ChromaDB
* **Embeddings:** HuggingFace (`hkunlp/instructor-large`)
* **OCR:** PyTesseract & PyMuPDF
* **TTS:** Edge-TTS

**Frontend:**

* Interactive Web UI for document uploads and querying.

---

## 🚦 Getting Started

### Prerequisites

* Python 3.13+
* Tesseract OCR installed on your system.
* Google API Key (for Gemini).

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/kritharth2005/Pravaah.git
cd Pravaah

```


2. **Setup Environment Variables:**
Create a `.env` file in the `backend/` directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here

```


3. **Install Dependencies:**
```bash
cd backend
pip install .

```


4. **Initialize Vector Database:**
Place your legal PDFs in `backend/PDFS/` and run:
```bash
python llm.py

```


5. **Run the Application:**
```bash
uvicorn app:app --reload

```



---

## 🐳 Docker Deployment

The application is containerized for easy deployment.

```bash
docker build -t pravaah-legal-ai .
docker run -p 8000:8000 --env-file .env pravaah-legal-ai

```

---

## ⚖️ Included Statutes

The system is pre-loaded with critical Indian legislation, including:

* Bhartiya Nyaya Sanhita (BNS)
* Companies Act 2013
* Consumer Protection Act 2019
* Indian Contract Act 1872
* Transfer of Property Act 1882
* ...and many others.

---

## 📄 Disclaimer

*Pravaah is an AI assistant and not a substitute for professional legal counsel. Always consult with a qualified legal professional for serious matters.*

Would you like me to add a section on how to contribute to the project or more detailed API documentation for the routers?
