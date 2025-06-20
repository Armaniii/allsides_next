# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AllSides Next is a multi-tier web application that generates diverse political perspectives on user queries using AI. It consists of:
- **Backend**: Django REST API with Gunicorn and PostgreSQL database
- **Frontend**: Next.js React application with TypeScript
- **Infrastructure**: Docker Compose orchestration with Nginx reverse proxy, Redis cache, and pgAdmin
- **AI/ML Services**: Ollama and vLLM for local LLM capabilities
- **Monitoring**: LangFuse to track API and LLM calls
- **Research Module**: LangGraph integration for deep research reports

## Key Features
- Local LLM enhancement (query formatting, follow-up questions, argument summarization, source analysis)
- Multi-perspective argument generation across political spectrum
- Source credibility analysis and bias detection
- User authentication and rating system


# Workflow Guidelines

## Code Quality
- Always run typechecking when you're done making code changes
- Prefer running single tests, not the whole test suite, for performance
- Use Black for Python formatting, ESLint for TypeScript
- Follow existing code patterns and conventions

## Docker Commands
- Always use "docker compose" without the hyphen
- Use `docker compose logs -f [service]` to check service status
- AI services (Ollama/vLLM) may take 1-5 minutes to start

## Task Management
- Update TODO.md when implementing complex features
- Mark progress and completion status for all tracked tasks
- Use TODO.md for features requiring deeper thought or multi-step implementation

## Development Process
- Test locally with `docker compose up -d` before committing
- Check AI service health: `curl localhost:11434/api/tags` (Ollama), `curl localhost:8001/v1/models` (vLLM)
- Monitor resource usage: `docker stats` for memory/CPU usage

## Important Files
- `allsides_next/backend/api/main_v3.py`: Core query processing logic
- `allsides_next/backend/api/llm_helpers.py`: Local LLM enhancement features
- `TODO.md`: Active task tracking and feature progress
- `LOCAL_LLM_INTEGRATION.md`: AI/ML service documentation

