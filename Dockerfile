FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF/Word processing
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port Streamlit uses
EXPOSE 7860

# Start both the FastAPI backend and Streamlit UI together
RUN chmod +x start.sh
CMD ["./start.sh"]
