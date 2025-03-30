# AllSides Next - Deployment Guide

## Overview

This guide covers the deployment setup for the AllSides Next application using Docker, Docker Compose, and Nginx. The application is designed to run as a set of containerized services with SSL encryption.

## Architecture

The deployment uses the following components:

- **Docker Compose**: Orchestrates all services
- **Nginx**: Reverse proxy for routing and SSL termination
- **Certbot**: Automatic SSL certificate management
- **PostgreSQL**: Database for persistent storage
- **Backend Container**: Django API service
- **Frontend Container**: Next.js frontend service

## Docker Compose Configuration

The main `docker-compose.yml` file defines all the services and their dependencies:

### Services

1. **Backend (Django)**
   - Builds from the backend Dockerfile
   - Connected to the PostgreSQL database
   - Exposes API endpoints

2. **Frontend (Next.js)**
   - Builds from the frontend Dockerfile
   - Makes API calls to the backend
   - Serves the user interface

3. **Database (PostgreSQL)**
   - Stores all application data
   - Uses volumes for data persistence

4. **Nginx**
   - Routes traffic to the appropriate services
   - Handles SSL termination
   - Serves static files

5. **Certbot**
   - Manages SSL certificates
   - Performs automatic renewal

## Environment Variables

The deployment requires several environment variables to be configured:

### Backend Environment Variables

- `DEBUG`: Enable/disable debug mode
- `SECRET_KEY`: Django secret key
- `ALLOWED_HOSTS`: List of allowed hosts
- `DATABASE_URL`: PostgreSQL connection string
- `CORS_ALLOWED_ORIGINS`: CORS configuration

### Frontend Environment Variables

- `NEXT_PUBLIC_API_URL`: URL of the backend API
- `NODE_ENV`: Production/development environment

## Deployment Steps

1. **Initial Setup**

   Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd allsides_next
   ```

2. **Configure Environment Variables**

   Create `.env` files for each service:
   ```bash
   cp .env.example .env
   cp allsides_next/frontend/.env.example allsides_next/frontend/.env.local
   cp allsides_next/backend/.env.example allsides_next/backend/.env
   ```

   Edit these files with your specific configuration values.

3. **Start the Services**

   Launch all services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. **Initialize the Database**

   Run migrations and create a superuser:
   ```bash
   docker-compose exec backend python manage.py migrate
   docker-compose exec backend python manage.py createsuperuser
   ```

5. **Setup SSL Certificates**

   Initialize SSL certificates with Certbot:
   ```bash
   docker-compose run --rm certbot certonly --webroot -w /var/www/certbot -d yourdomain.com
   ```

## Maintenance

### Database Backups

Regular database backups are stored in the `backups/` directory. To create a manual backup:

```bash
docker-compose exec db pg_dump -U postgres -d allsides > backups/backup-$(date +%Y%m%d%H%M%S).sql
```

### SSL Certificate Renewal

SSL certificates are automatically renewed by the Certbot service. To force a renewal:

```bash
docker-compose run --rm certbot renew
```

### Updating the Application

To update the application to the latest version:

1. Pull the latest code:
   ```bash
   git pull
   ```

2. Rebuild and restart containers:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   - Check the database credentials in the `.env` file
   - Ensure the PostgreSQL service is running

2. **Nginx Configuration**
   - Verify the Nginx configuration in the `nginx/` directory
   - Check Nginx logs: `docker-compose logs nginx`

3. **SSL Certificate Problems**
   - Ensure domains are correctly configured in the Certbot command
   - Check Certbot logs: `docker-compose logs certbot`

### Logs

To view logs for a specific service:

```bash
docker-compose logs [service-name]
```

For example:
```bash
docker-compose logs backend
docker-compose logs frontend
``` 