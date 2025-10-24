FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py
COPY artifacts ./artifacts

ENV PORT=7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
