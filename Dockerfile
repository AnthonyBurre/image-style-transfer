# Use a base image with Python and a recent Linux distribution
# debian:bookworm-slim is a good choice for smaller image size and includes Python 3.11
FROM python:3.12-slim-bookworm

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

# Set the entry point for the container.
# The `python -u` flag ensures unbuffered output, which is good for Docker logs.
ENTRYPOINT ["python", "-u", "-m", "src.app"]

# # Command to be executed by the entrypoint.
# CMD ["--host", "0.0.0.0"]
