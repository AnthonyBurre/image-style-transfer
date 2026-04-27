# Pinned by digest for reproducible builds (python:3.12-slim-bookworm multi-arch index).
FROM python:3.12-slim-bookworm@sha256:58525e1a8dada8e72d6f8a11a0ddff8d981fd888549108db52455d577f927f77

# Set the working directory inside the container
WORKDIR /app

# Explicitly create the group first (optional, adduser usually does this)
#    but being explicit ensures it's there.
RUN groupadd --system appgroup
# Create the user, associating them with the created group
#    `--ingroup appgroup` ensures the user's primary group is 'appgroup'
RUN adduser --system --ingroup appgroup --home /app appuser

# Ensure necessary directories are writable by the non-root user
RUN chown -R appuser:appgroup /app


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

# Set the entry point for the container.
# The `python -u` flag ensures unbuffered output, which is good for Docker logs.
ENTRYPOINT ["python", "-u", "-m", "src.app"]

# # Command to be executed by the entrypoint.
# CMD ["--host", "0.0.0.0"]
