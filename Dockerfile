FROM python:3.11-slim

# Create non-root user for security
RUN groupadd --gid 1000 stockai \
    && useradd --uid 1000 --gid stockai --shell /bin/bash --create-home stockai

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc6-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and set ownership
COPY . .
RUN chown -R stockai:stockai /app

# Create necessary directories with correct permissions
RUN mkdir -p models cache logs \
    && chown -R stockai:stockai models cache logs

# Switch to non-root user
USER stockai

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV HOST=0.0.0.0

# Expose port
EXPOSE 5000

# Health check with proper curl installation
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# Run the application
CMD ["python", "main.py"]