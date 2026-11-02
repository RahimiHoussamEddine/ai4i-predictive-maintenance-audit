FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.lock
COPY pyproject.toml README.md LICENSE NOTICE.md ./
COPY src ./src
COPY reports ./reports
COPY data ./data
RUN python -m pip install --no-deps --no-build-isolation .

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "maintenance_audit.api:app", "--host", "0.0.0.0", "--port", "8000"]
