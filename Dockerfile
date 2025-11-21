# Use a lightweight Python image
FROM python:3.10-slim

# Install system dependencies for OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements FIRST (important for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- NEW: Create Streamlit Config to permanently fix 403/400 Errors ---
RUN mkdir -p /root/.streamlit
RUN echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > /root/.streamlit/config.toml

# Copy the rest of the project
COPY . .

# Expose Streamlit port
EXPOSE 8501

# CMD to run Streamlit (Simplified since config.toml handles flags now)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]