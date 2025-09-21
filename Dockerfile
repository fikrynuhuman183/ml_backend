# Use Python 3.8.18 slim image as base
FROM python:3.8.18-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements_pip.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_pip.txt

# Copy application files
COPY app.py .
COPY util.py .
COPY *.h5 .

# Create directories for images and other data
RUN mkdir -p images

# Copy images directory if exists
COPY images/ ./images/

# Expose the port the app runs on
EXPOSE 5000

# Set default environment variables
ENV PORT=5000
ENV IMAGES_PATH=/app/images
ENV CSV_PATH=/app/data.csv

# Create a non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/hello || exit 1

# Run the application
CMD ["python", "app.py"]
