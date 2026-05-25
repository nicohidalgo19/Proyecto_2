FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY modelos/ ./modelos/

EXPOSE 8050

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--timeout", "120", "app:server"]