# Task Completion Checklist

When completing any coding task, follow these steps:

## Backend (Python/Django)
1. **Run Type Checking**: Ensure all type hints are correct
2. **Format Code**: Run Black formatter
   ```bash
   docker compose exec backend black .
   ```
3. **Lint Code**: Run Flake8
   ```bash
   docker compose exec backend flake8
   ```
4. **Run Tests**: Execute relevant tests (prefer single tests over full suite)
   ```bash
   docker compose exec backend pytest path/to/test.py::TestClass::test_method
   ```
5. **Check Migrations**: If models changed, create and apply migrations

## Frontend (TypeScript/Next.js)
1. **Type Check**: Run TypeScript compiler
   ```bash
   cd allsides_next/frontend && npx tsc --noEmit
   ```
2. **Lint Code**: Run ESLint
   ```bash
   npm run lint
   ```
3. **Test Build**: Ensure production build works
   ```bash
   npm run build
   ```

## General Checks
1. **Update TODO.md**: If working on a tracked feature, update progress
2. **Check Docker Services**: Ensure all services are running properly
3. **Test End-to-End**: Verify the feature works in the browser
4. **Review Error Logs**: Check for any new errors in docker logs
5. **Verify No Secrets**: Ensure no API keys or secrets are committed

## Important Reminders
- Always run linting and type checking commands after code changes
- Use the TODO.md file for tracking complex features
- Never commit unless explicitly asked by the user
- Prefer running single tests for performance
- If lint/typecheck commands are unknown, ask the user and suggest adding to CLAUDE.md