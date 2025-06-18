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
ARG OPENAI_API_KEY
ARG TAVILY_API_KEY
ARG LINKUP_API_KEY 
ARG GOOGLE_API_KEY
ARG LANGFUSE_SECRET_KEY
ARG LANGFUSE_PUBLIC_KEY
ARG LANGFUSE_HOST

# Set runtime environment variables from build args
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV TAVILY_API_KEY=${TAVILY_API_KEY}
ENV LINKUP_API_KEY=${LINKUP_API_KEY}
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}
ENV LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
ENV LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
ENV LANGFUSE_HOST=${LANGFUSE_HOST}
ENV REDIS_HOST=redis

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

# Copy only pyproject.toml and requirements.txt first
COPY pyproject.toml requirements.txt ./

# Install dependencies based on environment
RUN if [ "$ENVIRONMENT" = "production" ] ; then \
        poetry install --no-root --no-dev --no-cache ; \
    else \
        poetry install --no-root --no-cache ; \
    fi

# Explicitly install research requirements
RUN pip install open-deep-research>=0.0.10 langgraph>=0.2.55 langchain>=0.1.0 \
        langchain-core>=0.1.11 langchain-community>=0.3.9 langchain-openai>=0.3.7 \
        python-dotenv>=1.0.1 redis>=5.0.1 requests>=2.32.3 nest-asyncio>=1.5.8 asgiref>=3.7.2 \
        langchain-ollama==0.2.2 langchain_core>=0.1.20 pydantic>=1.10.8 \
        websockets>=12.0 httpx>=0.25.0

# Copy application code
COPY . .


# Collect static files if in production
RUN if [ "$ENVIRONMENT" = "production" ] ; then \
        python manage.py collectstatic --noinput ; \
    fi

# Set the command based on environment
CMD if [ "$ENVIRONMENT" = "production" ] ; then \
        gunicorn --bind 0.0.0.0:8000 --workers 4 --threads 4 --timeout 600 core.wsgi:application ; \
    else \
        python manage.py runserver 0.0.0.0:8000 ; \
    fi 