FROM python:3.10-slim

# System deps needed for Prophet, TensorFlow, scipy
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements first (layer caching)
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all backend files
COPY --chown=user . .

# HF Spaces must listen on port 7860
ENV PORT=7860
ENV HOST=0.0.0.0
ENV DEBUG=False

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "main:app"]
