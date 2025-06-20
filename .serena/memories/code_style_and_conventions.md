# Code Style and Conventions

## General Principles
- **NO COMMENTS**: Do not add any comments unless explicitly requested by the user
- **File Management**: Always prefer editing existing files over creating new ones
- **Documentation**: Never proactively create documentation files (*.md) or README files unless explicitly requested

## Python (Backend)
- **Style Guide**: Follow PEP 8
- **Formatting**: Use Black formatter (configured in pyproject.toml)
- **Linting**: Flake8 for code quality
- **Type Hints**: Use type hints for function parameters and returns
- **Async Patterns**: Use asyncio.gather() for parallel operations, ThreadPoolExecutor for CPU-bound tasks
- **Error Handling**: Comprehensive try-except blocks with proper logging
- **Django Patterns**:
  - Class-based views for complex logic
  - Serializers for data validation
  - ViewSets for REST endpoints

## TypeScript/JavaScript (Frontend)
- **Style Guide**: Next.js ESLint configuration
- **Type Safety**: Strict TypeScript with proper interfaces
- **Component Structure**:
  - Functional components with hooks
  - Props interfaces defined above components
  - Proper export patterns
- **State Management**: React Context for auth, React Query for server state
- **Styling**: Tailwind utility classes, avoid inline styles
- **File Organization**:
  - Components in dedicated files
  - Shared types in lib/types.ts
  - API calls in services/api.ts

## Docker and Infrastructure
- **Docker Compose**: Use "docker compose" (without hyphen)
- **Environment Variables**: Use .env files, never commit secrets
- **Service Names**: Consistent naming (backend, frontend, db, redis, etc.)

## Git Workflow
- **Commit Messages**: Clear, concise descriptions
- **Branch Strategy**: Work on feature branches
- **Never Auto-commit**: Only commit when explicitly asked by user