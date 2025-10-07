# ---- Builder Stage ----
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV_PATH=/opt/venv

# Create a non-root user for security
RUN addgroup --system nonroot && \
    adduser --system --ingroup nonroot --shell /bin/sh --no-create-home nonroot

# Install uv
RUN pip install uv

# Create the virtual environment
RUN python3 -m venv $VENV_PATH

# Set the PATH to include the venv
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY Backend/pyproject.toml Backend/uv.lock ./Backend/
WORKDIR /app/Backend

# Install dependencies using uv (respects lock file)
RUN uv sync --frozen --no-cache

# ---- Final Stage ----
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV_PATH=/opt/venv

ENV PATH="$VENV_PATH/bin:$PATH"

# Create non-root user
RUN addgroup --system nonroot && \
    adduser --system --ingroup nonroot --shell /bin/sh --no-create-home nonroot
USER nonroot

WORKDIR /app

# Copy the virtual environment and installed packages
COPY --from=builder --chown=nonroot:nonroot $VENV_PATH $VENV_PATH

# Copy application code
COPY --chown=nonroot:nonroot . .

EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
