FROM python:3.13-slim
WORKDIR /app
ENV PYTHONPATH=/app/src:/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src ./src
COPY api ./api
COPY app ./app
COPY .streamlit ./.streamlit
COPY models ./models
COPY reports ./reports
COPY data/marts ./data/marts
COPY data/powerbi/*.csv ./data/powerbi/
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
