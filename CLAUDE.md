# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AllSides Next is a multi-tier web application that generates diverse political perspectives on user queries. It consists of:
- **Backend**: Django REST API with PostgreSQL database
- **Frontend**: Next.js React application
- **Infrastructure**: Docker Compose with Nginx reverse proxy, Redis cache, and pgAdmin
- **Research Module**: LangGraph integration for deep research reports

## Common Development Commands

### Running the Application

```bash
# Start all services
docker-compose up -d

# Rebuild and start (after code changes)
docker-compose up -d --build

# View logs
docker-compose logs -f [service_name]

# Stop all services
docker-compose down
```

### Backend Development

```bash
# Access backend container
docker-compose exec backend bash

# Run Django migrations
docker-compose exec backend python manage.py migrate

# Create superuser
docker-compose exec backend python manage.py createsuperuser

# Collect static files
docker-compose exec backend python manage.py collectstatic --noinput

# Run Django shell
docker-compose exec backend python manage.py shell

# Run tests
docker-compose exec backend python manage.py test
```

### Frontend Development

```bash
# Access frontend container
docker-compose exec frontend sh

# Install dependencies
docker-compose exec frontend npm install

# Run development server (locally)
cd allsides_next/frontend && npm run dev

# Build production
cd allsides_next/frontend && npm run build

# Run linting
cd allsides_next/frontend && npm run lint
```

### Database Management

```bash
# Access PostgreSQL
docker-compose exec db psql -U allsides_user -d allsides_db

# Database backup
docker-compose exec db pg_dump -U allsides_user allsides_db > backup.sql

# pgAdmin access
# URL: http://localhost:5050
# Default login: admin@admin.com / admin
```

## Architecture & Key Components

### Critical Infrastructure Files

1. **docker-compose.yml**: Orchestrates all services with environment variables and networking
2. **nginx/conf.d/default.conf**: Handles routing, CORS, SSE streaming, and port preservation (port 9000)
3. **Backend Dockerfile**: Python 3.11 with Poetry dependency management
4. **Frontend Dockerfile**: Node.js 18 Alpine with multi-stage build

### Backend Structure

- **API Models** (`api/models.py`):
  - `User`: Custom user with bias ratings, query limits, and AllStars tracking
  - `Query`: Stores user queries and AI responses  
  - `ArgumentRating` & `ThumbsRating`: User feedback on arguments
  - `ResearchReport`: Deep research reports with LangGraph integration

- **Key Settings** (`core/settings.py`):
  - JWT authentication with 1-day access tokens
  - CORS configuration for cross-origin requests
  - Custom middleware for port preservation in redirects
  - Redis caching configuration

- **Research Module** (`api/research/`):
  - Integrates Open Deep Research library
  - Requires API keys: OPENAI_API_KEY and search provider keys (TAVILY_API_KEY, etc.)
  - Provides SSE streaming for real-time research updates

### Frontend Structure

- **Pages** (Next.js App Router):
  - `/`: Main query interface with argument display
  - `/login`: Authentication page
  - `/dashboard`: User statistics and history
  - `/research/*`: Research report creation and viewing

- **Key Components**:
  - `ArgumentCard`: Displays political perspectives
  - `StreamingArgumentCard`: Real-time argument generation
  - `ResearchMarkdownViewer`: Renders research reports with syntax highlighting

- **API Service** (`services/api.ts`):
  - Axios-based HTTP client with JWT authentication
  - Handles token refresh and error responses

## Environment Variables

Required in `.env` file:

```bash
# Django
SECRET_KEY=your-secret-key
DEBUG=False
DATABASE_URL=postgres://allsides_user:password@db:5432/allsides_db

# API Keys
OPENAI_API_KEY=your-openai-key
TAVILY_API_KEY=your-tavily-key  # For research module
GOOGLE_API_KEY=your-google-key
LINKUP_API_KEY=your-linkup-key
LANGFUSE_SECRET_KEY=your-langfuse-secret
LANGFUSE_PUBLIC_KEY=your-langfuse-public
LANGFUSE_HOST=your-langfuse-host

# Frontend
NEXT_PUBLIC_API_URL=http://34.134.51.8:9000/api
```

## Important Considerations

1. **Port Configuration**: The application runs on port 9000, with complex Nginx configuration to preserve this port in all redirects and API calls.

2. **Authentication**: All API endpoints except auth routes require JWT authentication. Frontend stores tokens in localStorage.

3. **SSE Streaming**: Research endpoints use Server-Sent Events. Nginx is configured with `proxy_buffering off` for real-time streaming.

4. **Database Migrations**: Always run migrations after pulling changes that modify models.

5. **Static Files**: Run `collectstatic` after backend changes that affect admin or static assets.

6. **CORS**: Configured to allow requests from frontend origins. Add new origins to both Django and Nginx configs.

7. **Research Module**: Requires at least one LLM provider key (OPENAI_API_KEY) and one search provider key (TAVILY_API_KEY, etc.).

## Main Query Processing Engine (api/main_v3.py)

The `main_v3.py` file is the core engine that processes user queries and generates diverse political perspectives using GPT-4. Here's how it works:

### Key Components

1. **Redis Caching System**:
   - Uses Redis connection pool for efficient caching
   - Generates deterministic cache keys based on normalized query, diversity score, number of stances, and system prompt
   - Implements intelligent TTL (time-to-live) based on access frequency
   - Tracks cache hits/misses and access patterns
   - Includes cache maintenance tasks for optimization and consistency checks

2. **Langfuse Integration** (lines 188-246, 387-471):
   - Configured via `PROMPT_MANAGER="langfuse"` environment variable
   - Retrieves system prompts from Langfuse service (`AllStances_v1` prompt)
   - Creates traces for each completion request for monitoring
   - Falls back to default prompt if Langfuse is unavailable
   - Tracks model parameters like temperature and stance count

3. **OpenAI API Integration** (lines 257-268, 442-467):
   - Uses OpenAI client with Instructor library for structured output
   - Configured with 5-minute timeout for long requests
   - Makes calls to GPT-4 model with structured response format
   - Handles streaming and non-streaming responses

4. **Query Processing Flow**:
   ```
   User Query → Normalize Query → Generate Cache Key → Check Cache
     ↓ (cache miss)
   Get System Prompt from Langfuse → Add Stance Count → Call GPT-4
     ↓
   Parse Structured Response → Cache Result → Return to User
   ```

5. **Structured Response Models** (lines 569-606):
   - `Stance`: Contains stance name, core argument, and 1-3 supporting arguments
   - `ArgumentResponse`: Contains 2-7 stances and the model used
   - Uses Pydantic for validation and serialization

6. **Cache Key Generation** (lines 352-386):
   - Normalizes query text (lowercase, remove extra spaces, strip punctuation)
   - Includes cache version, normalized topic, diversity score, stance count, and system prompt
   - Creates SHA-256 hash for consistent, deterministic keys

7. **Performance Features**:
   - Query normalization ensures similar queries hit the same cache
   - Background cache maintenance tasks run every 6 hours
   - Cache warming capabilities for batch processing
   - Memory usage monitoring and automatic cleanup when >80% full

### Configuration Requirements

```bash
# Required Environment Variables
OPENAI_API_KEY=your-openai-key
LANGFUSE_HOST=https://vector.allsides.com
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
REDIS_HOST=redis  # or your Redis host
REDIS_PORT=6379
CACHE_TTL=3600  # 1 hour default
```

### Error Handling

- Graceful fallback to default prompts if Langfuse fails
- Returns structured error responses maintaining the expected format
- Comprehensive logging for debugging
- Redis connection pooling with retry logic