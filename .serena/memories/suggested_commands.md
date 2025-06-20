# Suggested Commands for Development

## Docker Commands
```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f [service_name]

# Restart a specific service
docker compose restart [service_name]

# Stop all services
docker compose down

# Rebuild and start
docker compose up -d --build

# Access container shell
docker compose exec [service_name] bash
```

## Backend Development
```bash
# Run Django shell
docker compose exec backend python manage.py shell

# Make migrations
docker compose exec backend python manage.py makemigrations

# Apply migrations
docker compose exec backend python manage.py migrate

# Create superuser
docker compose exec backend python manage.py createsuperuser

# Collect static files
docker compose exec backend python manage.py collectstatic --noinput

# Run tests
docker compose exec backend pytest

# Format code
docker compose exec backend black .

# Lint code
docker compose exec backend flake8
```

## Frontend Development
```bash
# Install dependencies
cd allsides_next/frontend && npm install

# Run development server
npm run dev

# Build for production
npm run build

# Run linting
npm run lint

# Type checking (via TypeScript compiler)
npx tsc --noEmit
```

## Database Management
```bash
# Access PostgreSQL
docker compose exec db psql -U allsides_user -d allsides_db

# Backup database
docker compose exec db pg_dump -U allsides_user allsides_db > backup.sql

# Restore database
docker compose exec -T db psql -U allsides_user allsides_db < backup.sql
```

## System Utilities (Linux)
```bash
# Git commands
git status
git add .
git commit -m "message"
git push
git pull

# File operations
ls -la          # List files with details
cd [directory]  # Change directory
cat [file]      # View file contents
grep -r "pattern" .  # Search in files
find . -name "*.py"  # Find files by pattern

# Process management
ps aux | grep [process]
kill -9 [PID]
```

## Monitoring and Debugging
```bash
# View Redis keys
docker compose exec redis redis-cli

# Check service health
docker compose ps

# View Nginx logs
docker compose logs nginx

# Check disk usage
df -h

# Monitor system resources
htop
```