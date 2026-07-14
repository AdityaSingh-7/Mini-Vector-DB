FROM python:3.11-slim

WORKDIR /app

# Install Node.js for frontend build
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY brute_force.py hnsw.py hnsw_instrumented.py server.py ./

# Copy pre-built index and data
COPY saved_index/ ./saved_index/
COPY positions.npy texts.json ./

# Build frontend
COPY frontend/package.json frontend/.npmrc frontend/
COPY frontend/src/ frontend/src/
COPY frontend/index.html frontend/tsconfig.json frontend/vite.config.ts frontend/
RUN cd frontend && npm install && npx vite build

EXPOSE 8080

CMD ["python", "server.py"]
