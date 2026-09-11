# Single-stage on purpose: the image's size is dominated by torch and chromadb,
# which are runtime dependencies, so a builder stage would copy nearly all of it
# into the final layer anyway and buy complexity rather than megabytes.
FROM python:3.11-slim

# libgomp1 is required by onnxruntime, which chromadb pulls in unconditionally.
# Without it the image builds fine and then fails at first import, which is the
# worst time to find out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./

# ONE pip call, with PyTorch's CPU index primary and PyPI as the fallback.
#
# This is not a style preference. Installing CPU torch in a separate earlier
# command does NOT work: the second command is a fresh resolver with no knowledge
# of the CPU index, and it pulls the full CUDA-bundled torch straight back in
# from PyPI to satisfy a transitive constraint -- silently undoing the first
# command. That was diagnosed the expensive way against a real deploy; render.yaml
# carries the same fix and the same warning. A GPU-less container has no use for
# ~2GB of nvidia-* wheels.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
                   --extra-index-url https://pypi.org/simple \
                   -r requirements.txt

COPY . .

# Non-root. The app writes only to data/ (the chroma index and the audit log),
# so that is the only path needing ownership.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8000

# Bind the port immediately and construct the pipeline on the first real request.
# The pipeline is deliberately lazy (src/server.py) because eager construction
# once blew past a platform port-scan timeout on a low-CPU host; a HEALTHCHECK
# hitting /api/health does not force it, by design.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
