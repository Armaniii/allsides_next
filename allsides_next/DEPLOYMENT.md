# AllSides Next - Deployment Guide

## Overview

AllSides Next is a multi-tier web application that generates diverse political perspectives using AI. This guide covers deployment using Docker Compose with integrated AI/ML services.

## Architecture

The deployment consists of the following containerized services:

### Core Services
- **Nginx**: Reverse proxy and static file serving (ports 9000:80, 8443:443)
- **Frontend**: Next.js React application (port 3000)
- **Backend**: Django REST API with Gunicorn
- **PostgreSQL**: Primary database (internal port 5432)
- **Redis**: Caching layer (port 6379)

### AI/ML Services
- **Ollama**: Local LLM serving (port 11434)
- **vLLM**: OpenAI-compatible model serving (port 8001)

### Management Tools
- **pgAdmin**: Database administration UI (port 5050)

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 16GB+ RAM recommended (for AI services)
- 50GB+ free disk space

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd allsides_next_project
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   ```

3. **Configure environment variables** (see Environment Variables section)

4. **Start all services**
   ```bash
   docker compose up -d
   ```

5. **Verify deployment**
   ```bash
   docker compose ps
   ```

## Environment Variables

Create a `.env` file in the project root with the following variables:

### Database Configuration
```bash
POSTGRES_DB=allsides_db
POSTGRES_USER=allsides_user
POSTGRES_PASSWORD=<secure-password>
```

### Django Configuration
```bash
DEBUG=False
SECRET_KEY=<django-secret-key>
ALLOWED_HOSTS=*
NODE_ENV=production
```

### API Keys
```bash
OPENAI_API_KEY=<your-openai-key>
TAVILY_API_KEY=<your-tavily-key>
GOOGLE_API_KEY=<your-google-key>
LINKUP_API_KEY=<your-linkup-key>
```

### LangFuse Monitoring
```bash
LANGFUSE_SECRET_KEY=<langfuse-secret>
LANGFUSE_PUBLIC_KEY=<langfuse-public>
LANGFUSE_HOST=<langfuse-host-url>
```

### pgAdmin Configuration
```bash
PGADMIN_DEFAULT_EMAIL=admin@admin.com
PGADMIN_DEFAULT_PASSWORD=<pgadmin-password>
```

## Service Details

### Backend Service
- **Command**: Runs migrations, collects static files, starts Gunicorn
- **Workers**: 4 Gunicorn workers with 4 threads each
- **Timeout**: 600 seconds for long-running AI operations
- **Dependencies**: PostgreSQL, Redis, Ollama

### Frontend Service
- **Environment**: Next.js production build
- **API URL**: Configured via NEXT_PUBLIC_API_URL
- **Host**: Binds to 0.0.0.0 for container access

### Redis Service
- **Persistence**: AOF enabled with periodic saves
- **Data Volume**: redis_data for persistence

### Ollama Service
- **CPU Mode**: Optimized for 4-core systems
- **Memory**: 4-8GB allocated
- **Models**: Stored in ollama_data volume
- **Health Check**: Every 60 seconds

### vLLM Service
- **Model**: Microsoft DialoGPT-small (CPU mode)
- **Memory**: 8GB limit
- **Worker Method**: Ray with spawn
- **Health Check**: Every 60 seconds with 5-minute startup

## Deployment Commands

### Start Services
```bash
# Start all services
docker compose up -d

# Start specific service
docker compose up -d backend

# View logs
docker compose logs -f [service_name]
```

### Database Management
```bash
# Run migrations
docker compose exec backend python manage.py migrate

# Create superuser
docker compose exec backend python manage.py createsuperuser

# Access database shell
docker compose exec db psql -U allsides_user -d allsides_db

# Backup database
docker compose exec db pg_dump -U allsides_user allsides_db > backups/backup-$(date +%Y%m%d%H%M%S).sql

# Restore database
docker compose exec -T db psql -U allsides_user allsides_db < backups/backup.sql
```

### Static Files
```bash
# Collect static files
docker compose exec backend python manage.py collectstatic --noinput
```

### Service Management
```bash
# Restart service
docker compose restart [service_name]

# Stop all services
docker compose down

# Stop and remove volumes (CAUTION: deletes data)
docker compose down -v

# Rebuild service
docker compose build [service_name]

# Update and restart
docker compose up -d --build
```

## Accessing Services

- **Application**: http://localhost:9000
- **pgAdmin**: http://localhost:5050
- **Backend API**: http://localhost:9000/api (via Nginx)

### pgAdmin Database Connection
- Host: `db` (internal Docker network)
- Port: `5432`
- Username: `allsides_user`
- Password: (from POSTGRES_PASSWORD env var)

## Health Monitoring

### Check Service Health
```bash
# Overall status
docker compose ps

# Service logs
docker compose logs --tail=100 -f [service_name]

# Ollama health
curl http://localhost:11434/api/tags

# vLLM health
curl http://localhost:8001/v1/models
```

### Common Health Issues
- **Ollama**: May take 1-2 minutes to start
- **vLLM**: May take up to 5 minutes for initial model loading
- **Backend**: Check Redis and database connectivity

## Production Considerations

### SSL/TLS Configuration
1. Update Nginx configuration in `nginx/conf.d/`
2. Mount SSL certificates to Nginx container
3. Update ports to use 443 for HTTPS

### Resource Optimization
```yaml
# Add to docker-compose.yml for production
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

### Monitoring
- Use LangFuse dashboard for LLM usage tracking
- Monitor Docker resource usage: `docker stats`
- Set up log aggregation for production

## Troubleshooting

### Database Connection Issues
```bash
# Test database connection
docker compose exec backend python manage.py dbshell

# Check database logs
docker compose logs db
```

### Redis Connection Issues
```bash
# Test Redis connection
docker compose exec redis redis-cli ping

# Check Redis logs
docker compose logs redis
```

### AI Service Issues
```bash
# Restart Ollama
docker compose restart ollama

# Check Ollama models
docker compose exec ollama ollama list

# Restart vLLM
docker compose restart vllm
```

### Frontend Build Issues
```bash
# Rebuild frontend
docker compose build frontend

# Check frontend logs
docker compose logs frontend
```

## Backup and Recovery

### Automated Backups
Create a cron job for regular backups:
```bash
0 2 * * * cd /path/to/allsides_next_project && docker compose exec -T db pg_dump -U allsides_user allsides_db > backups/db_backup_$(date +\%Y\%m\%d).sql
```

### Backup All Data
```bash
# Database
docker compose exec db pg_dump -U allsides_user allsides_db > backups/db_backup.sql

# Redis
docker compose exec redis redis-cli SAVE
docker cp $(docker compose ps -q redis):/data/dump.rdb backups/redis_backup.rdb

# Media files
tar -czf backups/mediafiles_backup.tar.gz allsides_next/backend/mediafiles/
```

## AWS Aurora PostgreSQL Migration

For AWS Aurora deployment, use the alternative configuration:
```bash
docker compose -f docker-compose.aurora.yml up -d
```

See [AURORA_MIGRATION_GUIDE.md](../AURORA_MIGRATION_GUIDE.md) for detailed instructions.

## Maintenance

### Update Application
```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker compose build
docker compose up -d

# Run migrations
docker compose exec backend python manage.py migrate
```

### Clean Up
```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Complete cleanup (CAUTION)
docker system prune -a --volumes
```

## Security Best Practices

1. **Environment Variables**: Never commit `.env` files
2. **Database**: Use strong passwords, limit network exposure
3. **API Keys**: Rotate regularly, use least privilege
4. **Firewall**: Only expose necessary ports
5. **Updates**: Keep Docker images updated

## Support

For issues and questions:
- Check logs: `docker compose logs [service_name]`
- Review environment variables
- Ensure sufficient system resources
- Consult the [troubleshooting](#troubleshooting) section