FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow psycopg2-binary boto3 gunicorn
ENV PORT=5000
EXPOSE 5000
CMD ["sh","-lc","mlflow server \
  --host 0.0.0.0 --port $PORT \
  --backend-store-uri \"$BACKEND_STORE_URI\" \
  --default-artifact-root \"$ARTIFACT_ROOT\""]
