# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GATEWAY_CONFIG=/app/config/gateway.yaml
RUN useradd --uid 10001 --no-create-home gateway
WORKDIR /app
COPY --from=build /install /usr/local
COPY app ./app
COPY config ./config
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
# One worker per container: scale with replicas, not with processes (keeps metrics per pod simple).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
