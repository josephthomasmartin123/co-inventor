FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
# -m puts /app (cwd) on sys.path so `from app.api import ...` resolves
CMD ["python", "-m", "app.main"]
