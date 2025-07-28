# Challenge 1B: Persona-Driven Document Intelligence System
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY main.py .
COPY process_pdfs.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set the default command
CMD ["python", "main.py"]
