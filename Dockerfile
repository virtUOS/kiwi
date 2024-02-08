FROM python:3.12-slim
EXPOSE 8501

COPY requirements.txt /ai-portal/requirements.txt
RUN pip install --no-cache-dir -r /ai-portal/requirements.txt

COPY . /ai-portal
WORKDIR /ai-portal

USER nobody
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT [ "streamlit", "run", "start.py", "--server.port=8501", "--server.address=0.0.0.0"]
