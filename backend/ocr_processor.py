import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import fitz

def process_file_to_text(input_path:str) -> tuple[bool, str]:
    if not os.path.exists(input_path):
        return False, "File does not exist."
    
    try:
        file_extension = os.path.splittext(input_path)[1].lower()
        
        if file_extension == '.pdf':
            return _process_pdf(input_path)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return _process_image(input_path)
        else:
            return False, f"Unsupported file type: {file_extension}"
    except Exception as e:
        print(f"An unexpected error occured in process_file_to_text: {e}")
        return False, f"An internal error occured: {e}"
    

def _process_image(file_paath:str) -> tuple[bool, str]:
    print(f"Processing image fieel with OCR: {os.path.basename(file_paath)}")
    try:
        text = pytesseract.image_to_string(Image.open(file_paath))
        return True, text
    except Exception as e:
        print(f"Error processing image file {file_paath} with OCR: {e}")
        return False, "Failed to scan document!"
    
def _process_pdf(file_path:str) -> tuple[bool, str]:
    print(f"Processing PDF file with OCR: {os.path.basename(file_path)}")
    try:
        with fitz.open(file_path) as doc:
            full_text = "".join(page.get_text() for page in doc)
            
        if len(full_text.strip()) > 100:
            return True, full_text
        else:
            print("Direct text extraction yielded minimal text, falling back to OCR.")
        
    except Exception as e:
        print(f"Error extracting text directly from PDF {file_path}: {e}")
        
    print("Converting PDF pages to images for OCR...")
    try:
        images = convert_from_path(file_path)
        ocr_text = ""
        for i, image in enumerate(images):
            print(f"  - OCR on page{i+1}/{len(images)}")
            text = pytesseract.image_to_string(image)
            full_text += text + "\n\n"
            
        return True, full_text
    except Exception as e:
        print(f"OCR processing for PDF {file_path} failed: {e}")
        return False, "Failed to perform OCR on the PDF file."
