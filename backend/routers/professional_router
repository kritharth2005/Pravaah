from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import uuid
import shutil
from dotenv import load_dotenv
from models import QueryRequest, ResponseBody
from llm import professional_summarizer, professional_advisor
from ocr_processor import process_file_to_text

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIRECTORY = os.path.join(BASE_DIR, "uploads")
STATIC_DIRECTORY = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


router = APIRouter(prefix="/professional", tags=["Professional Summarizer and Advisor"])

@router.post("/upload-file-professional-summarizer/", response_model=ResponseBody)
async def handle_upload_file_professional_summarizer(language: str, file: UploadFile = File(...)):
    """
    Accepts a file (PDF, PNG, JPG), extracts text using the ocr_processor,
    and returns the text.
    """
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    temp_path = os.path.join(UPLOAD_DIRECTORY, f"{uuid.uuid4().hex}_{file.filename}")
    try:
        # Save the uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the saved file to extract text
        success, content = process_file_to_text(temp_path)

        if not success:
            # If processing fails, return the error message from the processor
            raise HTTPException(status_code=500, detail=content)

        return await handle_query_professional_summarizer(QueryRequest(query=content, language=language))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/upload-file-professional-advisor/", response_model=ResponseBody)
async def handle_upload_file_professional_advisor(language: str, file: UploadFile = File(...)):
    
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    temp_path = os.path.join(UPLOAD_DIRECTORY, f"{uuid.uuid4().hex}_{file.filename}")
    try:
        # Save the uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the saved file to extract text
        success, content = process_file_to_text(temp_path)

        if not success:
            # If processing fails, return the error message from the processor
            raise HTTPException(status_code=500, detail=content)

        return await handle_query_professional_advisor(QueryRequest(query=content, language=language))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)



@router.post("/query-professional-summarizer/", response_model=ResponseBody)
async def handle_query_professional_summarizer(data: QueryRequest):
    
    query = data.query
    if not query:
        raise HTTPException(
            status_code=400, detail="Query is required in the request body."
        )

    # Step 1: Get the text-based answer from the RAG system
    print(f"Received query: {query}")
    response_text, audio_url = await professional_summarizer(query, data.language)

    return ResponseBody(text=response_text, audio_path=audio_url)


@router.post("/query-professional-advisor/", response_model=ResponseBody)
async def handle_query_professional_advisor(data: QueryRequest):
    
    query = data.query
    if not query:
        raise HTTPException(
            status_code=400, detail="Query is required in the request body."
        )

    # Step 1: Get the text-based answer from the RAG system
    print(f"Received query: {query}")
    response_text, audio_url = await professional_advisor(query, data.language)

    return ResponseBody(text=response_text, audio_path=audio_url)


