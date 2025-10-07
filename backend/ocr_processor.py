import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import fitz  # PyMuPDF

# --- Configuration ---
# On Windows, you might need to uncomment and set this path if Tesseract is not in your system's PATH.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def process_file_to_text(input_path: str) -> tuple[bool, str]:
    """
    Processes a file (PDF, TXT, or Image) and extracts all text content.

    This function intelligently handles different file types:
    - For PDFs, it first attempts to extract digital text. If it fails or
      the text is minimal (indicating a scanned PDF), it falls back to OCR.
    - For images, it performs OCR directly.
    - For text files, it reads the content directly.

    Args:
        input_path (str): The full path to the file to be processed.

    Returns:
        tuple[bool, str]: A tuple containing:
                          - A boolean indicating success (True) or failure (False).
                          - The extracted text as a string, or an error message on failure.
    """
    if not os.path.exists(input_path):
        return False, "Error: File not found at the specified path."

    try:
        # CORRECTED: Changed 'splittext' to 'splitext'
        file_extension = os.path.splitext(input_path)[1].lower()

        if file_extension == '.pdf':
            return _process_pdf(input_path)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return _process_image(input_path)
        elif file_extension == '.txt':
            return _process_txt(input_path)
        else:
            return False, f"Unsupported file type: {file_extension}"
    except Exception as e:
        print(f"An unexpected error occurred in process_file_to_text: {e}")
        return False, f"An internal error occurred: {e}"


def _process_txt(file_path: str) -> tuple[bool, str]:
    """Reads text directly from a .txt file."""
    print(f"Processing .txt file: {os.path.basename(file_path)}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return True, text
    except Exception as e:
        print(f"Error reading .txt file {file_path}: {e}")
        return False, "Failed to read the text file."


def _process_image(file_path: str) -> tuple[bool, str]:
    """Performs OCR on a single image file."""
    print(f"Processing image file with OCR: {os.path.basename(file_path)}")
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        return True, text
    except Exception as e:
        print(f"Error performing OCR on image {file_path}: {e}")
        return False, "Failed to perform OCR on the image."


def _process_pdf(file_path: str) -> tuple[bool, str]:
    """
    Processes a PDF by first trying direct text extraction, then falling back to OCR.
    """
    # --- Step 1: Attempt to extract text directly (for digital PDFs) ---
    print(f"Attempting direct text extraction from PDF: {os.path.basename(file_path)}")
    try:
        with fitz.open(file_path) as doc:
            full_text = "".join(page.get_text() for page in doc)

        # Heuristic: If we get a reasonable amount of text, assume it's a digital PDF and we're done.
        if len(full_text.strip()) > 100:  # You can adjust this threshold
            print("Direct text extraction successful.")
            return True, full_text
        else:
            print("Direct text extraction yielded minimal text. Proceeding to OCR fallback.")
    except Exception as e:
        print(f"Direct text extraction failed: {e}. Proceeding to OCR fallback.")

    # --- Step 2: Fallback to OCR (for scanned/image-based PDFs) ---
    print(f"Processing PDF with OCR: {os.path.basename(file_path)}")
    try:
        images = convert_from_path(file_path)
        # CORRECTED: Initialized a new variable for OCR text to avoid the UnboundLocalError
        ocr_text = ""
        for i, image in enumerate(images):
            print(f"  - OCR on page {i + 1}/{len(images)}")
            text = pytesseract.image_to_string(image)
            ocr_text += text + "\n\n"  # Add page breaks for clarity

        return True, ocr_text
    except Exception as e:
        print(f"OCR processing for PDF {file_path} failed: {e}")
        return False, "Failed to perform OCR on the PDF file."

