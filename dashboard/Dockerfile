FROM python:3.10-slim

WORKDIR /app

COPY . /app

ENV PIP_DEFAULT_TIMEOUT=100

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
