# Pinned by digest for reproducible builds (python:3.12-slim-bookworm multi-arch index).
FROM python:3.12-slim-bookworm@sha256:58525e1a8dada8e72d6f8a11a0ddff8d981fd888549108db52455d577f927f77

# Set the working directory inside the container
WORKDIR /app

# Non-root runtime user; explicit groupadd before adduser so --ingroup resolves.
RUN groupadd --system appgroup && \
    adduser --system --ingroup appgroup --home /app appuser && \
    chown -R appuser:appgroup /app


# Copy the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Reduces image size by not caching pip packages
# --upgrade pip: Ensures pip is up-to-date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Python application code into the container
COPY src/ src/

# Switch to the non-root user for runtime
USER appuser

# Expose the port Gradio typically runs on (default is 7860)
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3).status == 200 else 1)"

# Default to the Gradio web UI; override CMD to run the headless CLI:
#   docker run … style-transfer src.cli -c … -s … -o … -m …
# `python -u` keeps stdout/stderr unbuffered so Docker logs are live.
ENTRYPOINT ["python", "-u", "-m"]
CMD ["src.app"]
