# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AllSides Next is a multi-tier web application that generates diverse political perspectives on user queries. It consists of:
- **Backend**: Django REST API as WSGI with PostgreSQL database 
- **Frontend**: Next.js React application
- **Infrastructure**: Docker Compose with Nginx reverse proxy, Redis cache, and pgAdmin
- **Logging**: LangFuse to track API and LLM calls
- **Research Module**: LangGraph integration for deep research reports


# Workflow
- Be sure to typecheck when you’re done making a series of code changes
- Prefer running single tests, and not the whole test suite, for performance
## Memories

- Create a TODO.md where you will always update with features that you have yet to implement, updated progress for said features everytime you make progress on implementation and the final status (completed). You should update this when i instruct you too, or when you realize that your are implementing a feature that requires deeper thought.
- Always Use "docker compose" without the hyphen

