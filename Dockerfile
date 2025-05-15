FROM python:3.11.3
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose API port
EXPOSE 8081

# Run the server
# uvicorn app.main:app --host 0.0.0.0 --port 8081
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]