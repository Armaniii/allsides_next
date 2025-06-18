# Aurora Migration Guide

This guide covers migrating from local PostgreSQL to AWS Aurora Serverless PostgreSQL.

## Pre-Migration Checklist

- [x] Database backup created: `db_backup_pre_migration_*.sql`
- [x] Cleanup branch created: `migration-cleanup`
- [x] Virtual environments and caches removed
- [x] Test files cleaned up

## Aurora Setup Steps

### 1. AWS Aurora Setup

```bash
# Create Aurora Serverless v2 cluster (via AWS CLI or Console)
aws rds create-db-cluster \
    --db-cluster-identifier allsides-aurora \
    --engine aurora-postgresql \
    --engine-version 15.4 \
    --master-username postgres \
    --master-user-password YOUR_SECURE_PASSWORD \
    --serverless-v2-scaling-configuration MinCapacity=0.5,MaxCapacity=4 \
    --storage-encrypted
```

### 2. EC2 Instance Setup

```bash
# On your new EC2 instance:
git clone https://github.com/your-repo/allsides_next_project.git
cd allsides_next_project
git checkout migration-cleanup

# Copy environment template
cp .env.aurora.example .env

# Update .env with your Aurora endpoint and credentials
nano .env
```

### 3. Database Migration

```bash
# Install PostgreSQL client on EC2
sudo apt update
sudo apt install postgresql-client

# Import your data to Aurora
psql -h your-aurora-endpoint.cluster-xxx.region.rds.amazonaws.com \
     -U postgres \
     -d allsides_db \
     -f db_backup_pre_migration_*.sql
```

### 4. Application Configuration

Update `allsides_next/backend/core/settings.py`:

```python
# Replace the DATABASES setting with:
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('AURORA_DB_NAME', 'allsides_db'),
        'USER': os.getenv('AURORA_USER', 'postgres'),
        'PASSWORD': os.getenv('AURORA_PASSWORD'),
        'HOST': os.getenv('AURORA_ENDPOINT'),
        'PORT': os.getenv('AURORA_PORT', '5432'),
        'OPTIONS': {
            'sslmode': os.getenv('AURORA_SSL_MODE', 'require'),
            'connect_timeout': 60,
            'options': '-c statement_timeout=30000'
        },
        'CONN_MAX_AGE': 60,
        'CONN_HEALTH_CHECKS': True,
    }
}

# Update for production
DEBUG = False
ALLOWED_HOSTS = ['your-ec2-public-ip', 'your-domain.com']
```

### 5. Docker Setup

```bash
# Use the Aurora-specific docker-compose file
cp docker-compose.aurora.yml docker-compose.yml

# Build and start services
docker compose build
docker compose up -d

# Run migrations
docker compose exec backend python manage.py migrate

# Create superuser
docker compose exec backend python manage.py createsuperuser
```

### 6. Security Group Configuration

Ensure your Aurora security group allows:
- Inbound PostgreSQL (5432) from your EC2 security group
- EC2 security group allows HTTP (80), HTTPS (443), and your app ports

### 7. Testing

```bash
# Test database connection
docker compose exec backend python manage.py dbshell

# Test application
curl http://your-ec2-public-ip:9000/api/

# Check logs
docker compose logs backend
```

## Optimization for Aurora Serverless

### Connection Pooling
Aurora Serverless benefits from connection pooling:

```python
# In settings.py
DATABASES['default']['CONN_MAX_AGE'] = 60
DATABASES['default']['CONN_HEALTH_CHECKS'] = True
```

### Query Optimization
- Use database indexes effectively
- Monitor Aurora performance insights
- Consider read replicas for heavy read workloads

### Cost Optimization
- Set appropriate min/max capacity
- Monitor usage patterns
- Use Aurora Data API for serverless functions

## Rollback Plan

If migration fails:

```bash
# Switch back to original branch
git checkout master

# Restore local environment
docker compose up -d

# Restore database if needed
docker compose exec db psql -U allsides_user -d allsides_db < db_backup_pre_migration_*.sql
```

## Environment Variables Reference

Required in `.env`:
- `AURORA_ENDPOINT`: Aurora cluster endpoint
- `AURORA_DB_NAME`: Database name
- `AURORA_USER`: Database user
- `AURORA_PASSWORD`: Database password
- `SECRET_KEY`: Django secret key
- API keys (OpenAI, Tavily, etc.)

## Monitoring

- Enable Aurora Performance Insights
- Set up CloudWatch alarms
- Monitor application logs
- Track connection counts and query performance