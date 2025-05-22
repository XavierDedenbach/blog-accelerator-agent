# Stage 1: Builder stage
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install poetry (if you are using poetry, otherwise we adjust for pip)
# For now, assuming requirements.txt is used directly as per standard Python projects.
# If you use Poetry, we'll need to adjust this section.
# RUN pip install poetry
# COPY pyproject.toml poetry.lock* /app/
# RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi

# Install dependencies using requirements.txt
COPY requirements.txt /app/
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels uvicorn


# Stage 2: Runtime stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /wheels/*

# Make sure uvicorn is installed
RUN pip install --no-cache-dir uvicorn

# Copy the application code
# Assuming your FastAPI app is in the 'api' directory and other necessary code is in 'agents', 'templates', 'static'
COPY ./api /app/api
COPY ./agents /app/agents
COPY ./templates /app/templates
COPY ./static /app/static
COPY ./run_researcher_with_env.py /app/run_researcher_with_env.py
# Add any other necessary files or directories that your app needs at runtime

# Expose the port the app runs on
EXPOSE 8080

# Command to run the FastAPI application
# Make sure api.main:app points to your FastAPI app instance
# Use --host 0.0.0.0 to make it accessible from outside the container
# The port 8080 matches what's in your docker-compose.yml
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"] 

