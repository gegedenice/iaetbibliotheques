# syntax=docker/dockerfile:1
FROM python:3.11.9-slim

RUN apt-get update && apt-get install -y git curl vim

WORKDIR /app
COPY --chmod=777 . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

VOLUME ["/app"]

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]