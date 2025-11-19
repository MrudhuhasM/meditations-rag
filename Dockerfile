# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
# --frozen ensures we use the exact versions in uv.lock
# --no-dev excludes development dependencies
# --no-install-project installs only dependencies, not the package itself
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src ./src
COPY scripts ./scripts
COPY README.md ./

# Install the project itself
RUN uv sync --frozen --no-dev

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "meditations_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
