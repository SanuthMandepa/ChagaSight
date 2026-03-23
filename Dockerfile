FROM python:3.11-slim

# HuggingFace Spaces requires port 7860
ENV PORT=7860
ENV FLASK_DEBUG=false

WORKDIR /app

# Install system deps for wfdb / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy source
COPY app.py .
COPY src/ src/

# Copy models — real .pt files must be present in the repo
# (added via git lfs, see deployment guide)
COPY models/ models/

# Upload temp dir
RUN mkdir -p uploads

EXPOSE 7860

CMD ["python", "app.py"]
