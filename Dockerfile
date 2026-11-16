FROM node:22-bookworm-slim AS dashboard
WORKDIR /ui
COPY dashboard/package*.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FLWR_TELEMETRY_ENABLED=0
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-fl.txt requirements-dashboard.txt ./
RUN pip install --no-cache-dir -r requirements-dashboard.txt
COPY service/ service/
COPY fl/upi_fl/ fl/upi_fl/
COPY --from=dashboard /ui/dist/ dashboard/dist/
RUN mkdir -p /state && useradd --uid 10001 --create-home pos && chown pos /state
USER pos
ENTRYPOINT ["python", "-m", "service.flower_transport"]
