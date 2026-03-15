FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install torch separately first (largest package)
RUN pip install --no-cache-dir --timeout=300 torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir --timeout=300 \
    streamlit==1.32.0 \
    psycopg2-binary==2.9.9 \
    pandas==2.2.1 \
    numpy==1.26.4 \
    matplotlib==3.8.3 \
    plotly==5.20.0 \
    sqlalchemy==2.0.28 \
    sentence-transformers==2.6.1

# Copy app files
COPY streamlit_app.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
