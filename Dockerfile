FROM python:3.12-slim
EXPOSE 8501

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
WORKDIR /app

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT [ "streamlit", "run", "Welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]
