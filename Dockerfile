# Use a specific version of the Python slim image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variable to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    git \
    git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment and upgrade pip
RUN python -m venv /opt/venv && \
    /opt/venv/bin/python -m ensurepip && \
    /opt/venv/bin/pip install --upgrade pip

# Add venv to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements file
COPY ./Backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY ./Backend .

# Expose the FastAPI port
EXPOSE 8000

# Run app using python -m uvicorn for reliability
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
