# Tiny CPU-only image running the dual-stream annotation demo.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# Two worker threads would run two independent engines; the demo is a single
# in-process engine, so one uvicorn worker is intentional.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
