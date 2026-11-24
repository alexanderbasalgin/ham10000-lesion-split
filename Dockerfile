FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY skinlesion skinlesion
COPY app app
# weights are not in git: put finetune.pt into artifacts/ before building
COPY artifacts/finetune.pt artifacts/finetune.pt
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
