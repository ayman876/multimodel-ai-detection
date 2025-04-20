FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install wheel first to speed up installation of heavy packages
RUN pip install --upgrade pip \
 && pip install wheel \
 && pip install --prefer-binary -r requirements.txt

 CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]