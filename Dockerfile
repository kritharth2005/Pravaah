# ----------------------------
# Dockerfile for FastAPI backend
# ----------------------------

# Base image
FROM python:3.14-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Copy backend code into container
COPY backend /app/

# Install system dependencies for OCR and PDF processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Using pip for simplicity; if using pdm, adjust accordingly
RUN pip install --upgrade pip && \
    pip install "edge-tts>=7.2.3" \
                "fastapi>=0.118.0" \
                "langchain>=0.3.27" \
                "langchain-chroma>=0.2.6" \
                "langchain-community>=0.3.31" \
                "langchain-google-genai>=2.1.12" \
                "langchain-huggingface>=0.3.1" \
                "pdf2image>=1.17.0" \
                "pillow>=11.3.0" \
                "pymupdf>=1.26.4" \
                "pytesseract>=0.3.13" \
                "python-dotenv>=1.1.1" \
                "python-multipart>=0.0.20" \
                "sentence-transformers>=5.1.1" \
                "torch>=2.8.0" \
                "uvicorn>=0.37.0"

# Expose port for FastAPI
EXPOSE 8000

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
