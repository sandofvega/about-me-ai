FROM python:3.12-alpine

WORKDIR /app

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application code.
COPY . .

# Install the application dependencies.
RUN uv sync --frozen --no-cache --no-dev

# Expose the port that the application will run on.
EXPOSE 8080

# Run the application.
CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]