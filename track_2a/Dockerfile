# Tse-min — SwissGov-RSD cross-lingual semantic difference detection with Apertus (Hack Apertus, Track 2A / UZH).
# The pipeline is pure Python standard library: nothing to install.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
COPY . /work
RUN chmod +x /work/src/entrypoint.sh
ENTRYPOINT ["/work/src/entrypoint.sh"]
