FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies (no sentence-transformers needed at runtime)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY brute_force.py hnsw.py hnsw_instrumented.py server.py ./

# Copy pre-built index and data
COPY saved_index/ ./saved_index/
COPY positions.npy texts.json ./

# Copy pre-built frontend
COPY frontend/dist/ ./frontend/dist/

EXPOSE 8080

CMD ["python", "server.py"]
