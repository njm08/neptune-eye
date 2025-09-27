# Use Python 3.11 slim image as base
# TODO Use ultralytics for Jetson

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Copy the source code
COPY src/ src/

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set the default command to run the main Python file as a module
CMD ["python", "-m", "neptune_eye.neptune_eye"]