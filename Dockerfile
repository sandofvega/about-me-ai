FROM python:3.12-slim

WORKDIR /app

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application code.
COPY . ./

# Install the application dependencies.
RUN uv sync --frozen --no-cache --no-dev

# Expose the port that the application will run on.
EXPOSE 8000

# Run the application.
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "8000", "--host", "0.0.0.0"]