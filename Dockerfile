FROM python:3.10-slim

WORKDIR /app

# Copie les fichiers nécessaires
COPY requirements.txt .
COPY app.py .
COPY le_queue.pkl .
COPY le_urgency.pkl .
COPY xgboost_queue_v2/ ./xgboost_queue_v2/
COPY xgboost_urgency_v2/ ./xgboost_urgency_v2/

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Expose le port
EXPOSE 8000

# Lance l'API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
