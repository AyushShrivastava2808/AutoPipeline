FROM python:3.11-slim

RUN pip install --timeout 300 --retries 10 --no-cache-dir \
    mlflow==2.17.2 \
    sqlalchemy==2.0.36 \
    psycopg2-binary==2.9.10

ENTRYPOINT ["mlflow"]
