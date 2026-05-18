# Pinned by digest for reproducible builds (python:3.12-slim-bookworm multi-arch index).
FROM python:3.12-slim-bookworm@sha256:58525e1a8dada8e72d6f8a11a0ddff8d981fd888549108db52455d577f927f77

# Pin uv version for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /uvx /bin/

WORKDIR /app

# Non-root runtime user; explicit groupadd before adduser so --ingroup resolves.
RUN groupadd --system appgroup && \
    adduser --system --ingroup appgroup --home /app appuser && \
    chown -R appuser:appgroup /app

# Copy manifests first to leverage Docker layer caching for the dependency install.
COPY pyproject.toml uv.lock ./

# UV_LINK_MODE=copy avoids hardlink warnings on bind-mounted build contexts.
# UV_COMPILE_BYTECODE=1 trades a small build-time cost for faster container startup.
# UV_NO_CACHE=1 keeps uv's wheel cache (~2 GB for this dep set) out of the image —
# Docker's own layer cache already covers rebuilds when uv.lock is unchanged.
# --no-install-project: install deps only; the source tree is copied in the next layer.
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1
RUN uv sync --frozen --no-dev --no-install-project --extra cpu

COPY src/ src/

RUN uv sync --frozen --no-dev --extra cpu

ENV PATH="/app/.venv/bin:$PATH"
USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3).status == 200 else 1)"

# Default to the Gradio web UI; override CMD to run the headless CLI:
#   docker run … style-transfer src.cli -c … -s … -o … -m …
# `python -u` keeps stdout/stderr unbuffered so Docker logs are live.
ENTRYPOINT ["python", "-u", "-m"]
CMD ["src.app"]
