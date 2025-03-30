FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1 \
    VIRTUAL_ENV=/opt/venv

# Add build argument
ARG ENVIRONMENT=development

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Create and activate virtual environment
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only pyproject.toml first
COPY pyproject.toml ./

# Install dependencies based on environment
RUN if [ "$ENVIRONMENT" = "production" ] ; then \
        poetry install --no-root --no-dev --no-cache ; \
    else \
        poetry install --no-root --no-cache ; \
    fi

# Copy application code
COPY . .

# Collect static files if in production
RUN if [ "$ENVIRONMENT" = "production" ] ; then \
        python manage.py collectstatic --noinput ; \
    fi

# Set the command based on environment
CMD if [ "$ENVIRONMENT" = "production" ] ; then \
        gunicorn --bind 0.0.0.0:8000 --workers 4 --threads 4 core.wsgi:application ; \
    else \
        python manage.py runserver 0.0.0.0:8000 ; \
    fi 