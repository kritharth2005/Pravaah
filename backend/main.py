import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the OCR function and the single RAG system instance
# from ocr_processor import ocr_pdf_to_text
from llm import human_summarizer

# --- Environment and App Initialization ---
load_dotenv()

# Check for API Key
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")

app = FastAPI(
    title="Pravaah Legal AI",
    description="API for summarizing and advising on legal documents.",
    version="1.0.0"
)

# --- Directory Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIRECTORY = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIRECTORY = os.path.join(BASE_DIR, "processed_texts")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)


# --- Pydantic Models for API Request Bodies ---
class ProcessTextRequest(BaseModel):
    text: str
    source_filename: str

class QueryRequest(BaseModel):
    query: str


# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Pravaah Legal AI API"}

# @app.post("/ocr-upload/")
# async def create_ocr_upload(file: UploadFile = File(...)):
#     unique_id = uuid.uuid4().hex
#     pdf_filename = f"{unique_id}_{file.filename}"
#     txt_filename = f"{unique_id}_{os.path.splitext(file.filename)[0]}.txt"
#     pdf_path = os.path.join(UPLOAD_DIRECTORY, pdf_filename)
#     txt_path = os.path.join(PROCESSED_DIRECTORY, txt_filename)
    
#     try:
#         with open(pdf_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # success = ocr_pdf_to_text(pdf_path, txt_path)
        
#         if not success:
#             raise HTTPException(status_code=500, detail="OCR processing failed.")
            
#         with open(txt_path, 'r', encoding='utf-8') as f:
#             extracted_text = f.read()
            
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "message": "File processed successfully.",
#                 "filename": file.filename,
#                 "extracted_text": extracted_text
#             }
#         )
#     finally:
#         if os.path.exists(pdf_path): os.remove(pdf_path)
#         if os.path.exists(txt_path): os.remove(txt_path)

@app.get("/query")
def query_endpoint(q: str):
    try:
        response = human_summarizer(q)
        return JSONResponse(status_code=200, content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))