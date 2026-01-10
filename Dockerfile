# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install uv
RUN python -m pip install --no-cache-dir uv

# Set work directory
WORKDIR /code

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy project specification and lock files
COPY pyproject.toml uv.lock /code/

# Install project dependencies
ENV VIRTUAL_ENV=/code/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv sync --frozen --no-install-project

# Copy project files
COPY . /code/

# Run the application
CMD ["python", "whisper-simple-bot"]
