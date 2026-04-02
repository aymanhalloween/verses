# ---- Stage 1: Build frontend ----
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --silent
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend + serve frontend ----
FROM python:3.11-slim
WORKDIR /app

# Install ffmpeg (required for audio processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Pre-download the Whisper model so it's baked into the image
# (avoids slow first-request download at runtime)
RUN python -c "\
from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
WhisperProcessor.from_pretrained('tarteel-ai/whisper-base-ar-quran'); \
WhisperForConditionalGeneration.from_pretrained('tarteel-ai/whisper-base-ar-quran'); \
print('Model cached successfully')"

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Expose port (Railway sets PORT env var)
EXPOSE 8000

# Start server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]
