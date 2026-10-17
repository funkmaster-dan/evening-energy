FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ENERGY_DATA_DIR=/data OMP_NUM_THREADS=2
ARG VCS_REF=unknown
LABEL org.opencontainers.image.revision=$VCS_REF
ENV ENERGY_RELEASE_COMMIT=$VCS_REF
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY scripts ./scripts/
EXPOSE 8099
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8099"]
