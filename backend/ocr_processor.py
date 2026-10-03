import logging
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image, UnidentifiedImageError

from config import MAX_OCR_PAGES

log = logging.getLogger(__name__)

# On Windows, set this if Tesseract is not on PATH:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Extension -> required leading bytes (None: plain text, no signature)
SIGNATURES = {
    ".pdf": (b"%PDF-",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".txt": None,
}
SUPPORTED_EXTENSIONS = tuple(SIGNATURES)
# Below this many characters a PDF's text layer is treated as missing (scanned document).
MIN_DIGITAL_TEXT_CHARS = 100


class UnsupportedFileError(Exception):
    """The file type is not accepted, or its contents don't match its extension."""


class ExtractionError(Exception):
    """No usable text could be extracted from the document."""


class OcrUnavailableError(Exception):
    """Tesseract or Poppler, needed for images and scanned PDFs, is not installed on the server."""


def extract_text(path: Path) -> tuple[str, list[str]]:
    """Returns the document's text and any notices about partial processing."""
    extension = path.suffix.lower()
    if extension not in SIGNATURES:
        raise UnsupportedFileError(f"Unsupported file type '{extension}'. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}")
    signatures = SIGNATURES[extension]
    if signatures:
        with path.open("rb") as f:
            if not f.read(16).startswith(signatures):
                raise UnsupportedFileError(f"The file's contents are not a valid {extension} file.")

    try:
        if extension == ".pdf":
            text, notices = _process_pdf(path)
        elif extension == ".txt":
            text, notices = path.read_text(encoding="utf-8", errors="replace"), []
        else:
            # Closed explicitly: an open handle blocks deleting the temp upload on Windows.
            with Image.open(path) as image:
                text, notices = pytesseract.image_to_string(image), []
    except (pytesseract.TesseractNotFoundError, PDFInfoNotInstalledError) as e:
        raise OcrUnavailableError("OCR is not available on the server (Tesseract/Poppler not installed).") from e
    except UnidentifiedImageError as e:
        raise ExtractionError("The image could not be read.") from e

    if not text.strip():
        raise ExtractionError("No readable text was found in the document.")
    return text, notices


def _process_pdf(path: Path) -> tuple[str, list[str]]:
    """Uses the PDF's text layer when present, otherwise OCRs it page by page."""
    try:
        with fitz.open(path) as doc:
            page_count = doc.page_count
            text = "".join(page.get_text() for page in doc)
    except Exception as e:
        raise ExtractionError("The PDF could not be opened.") from e

    if len(text.strip()) > MIN_DIGITAL_TEXT_CHARS:
        return text, []

    log.info("PDF has no usable text layer; running OCR on up to %d of %d pages", MAX_OCR_PAGES, page_count)
    pages_to_read = min(page_count, MAX_OCR_PAGES)
    # One page at a time so a long scan never holds every rendered page in memory.
    ocr_pages = [
        pytesseract.image_to_string(convert_from_path(path, first_page=number, last_page=number)[0])
        for number in range(1, pages_to_read + 1)
    ]
    notices = []
    if page_count > pages_to_read:
        notices.append(f"This scanned PDF has {page_count} pages; only the first {pages_to_read} were read.")
    return "\n\n".join(ocr_pages), notices
