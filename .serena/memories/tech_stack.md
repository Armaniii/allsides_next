# Technology Stack

## Backend
- **Language**: Python 3.11+
- **Framework**: Django 4.2.17 with Django REST Framework 3.15.2
- **Database**: PostgreSQL 15 (with Aurora migration support)
- **Authentication**: JWT via djangorestframework-simplejwt
- **Web Server**: Gunicorn with 4 workers
- **Caching**: Redis with django-redis
- **AI/ML Libraries**:
  - OpenAI 1.84.0
  - Instructor 1.7.2 (structured outputs)
  - LangChain ecosystem (langgraph, langchain-core, langchain-openai)
  - Open-deep-research (for research module)
  - Ollama integration for local LLMs
- **Async Support**: asyncio, aioredis, nest-asyncio
- **Observability**: LangFuse, OpenTelemetry

## Frontend
- **Framework**: Next.js 14.2.23 (App Router)
- **Language**: TypeScript 5
- **UI Libraries**:
  - React 18
  - Tailwind CSS 3.3.0
  - Framer Motion 10.16.4 (animations)
  - Radix UI (components)
  - Heroicons & Lucide React (icons)
- **Data Fetching**: 
  - Axios 1.6.2
  - React Query (Tanstack Query 5.0.0)
- **Markdown**: react-markdown with syntax highlighting
- **Styling**: Tailwind with class-variance-authority

## Infrastructure
- **Containerization**: Docker & Docker Compose
- **Reverse Proxy**: Nginx
- **Process Management**: Gunicorn
- **Local LLM Serving**: 
  - Ollama (CPU-optimized)
  - vLLM (with CPU fallback)
- **Database Admin**: pgAdmin 4

## Development Tools
- **Python**: Poetry for dependency management, Black for formatting, Flake8 for linting, Pytest for testing
- **JavaScript**: ESLint (Next.js config), TypeScript compiler
- **Version Control**: Git (current branch: migration-cleanup)