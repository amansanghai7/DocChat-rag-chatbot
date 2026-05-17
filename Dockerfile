# ── Stage 1: dependency installation ─────────────────────────────────────────
# Use a separate layer so Docker can cache it — re-runs only when
# requirements.txt changes, not every time source code changes.
FROM python:3.13-slim AS deps

WORKDIR /build

# System packages needed to compile any C extensions.
# libpq-dev is only needed if psycopg falls back to the source build;
# psycopg[binary] bundles its own libpq so this is a safety net only.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# Non-root user for security — running as root inside a container is risky
RUN useradd --create-home appuser

WORKDIR /code

# Copy only the installed site-packages from the build stage (not gcc etc.)
COPY --from=deps /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source code
COPY . .

# Ensure the uploads directory exists (runtime artefacts)
RUN mkdir -p uploads && chown -R appuser:appuser /code

USER appuser

# Expose ports:
#   8000 — FastAPI (uvicorn)
#   8501 — Streamlit
EXPOSE 8000 8501

# Default command runs the FastAPI server.
# Override with "streamlit run frontend_rag.py ..." in docker-compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
