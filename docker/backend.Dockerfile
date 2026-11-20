FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --timeout 300 --retries 10 --no-cache-dir -r requirements.txt

COPY backend/ .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
