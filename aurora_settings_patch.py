"""
Aurora PostgreSQL settings patch for Django
Apply this patch to your settings.py file when migrating to Aurora
"""

import os

# Aurora Database Configuration
AURORA_DATABASES = {
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
        'CONN_MAX_AGE': 60,  # Connection pooling for Aurora
        'CONN_HEALTH_CHECKS': True,
    }
}

# Aurora-specific settings
AURORA_SETTINGS = {
    # Connection pooling settings for Aurora Serverless
    'CONN_MAX_AGE': 60,
    'CONN_HEALTH_CHECKS': True,
    
    # Aurora-optimized database settings
    'DATABASE_OPTIONS': {
        'sslmode': 'require',
        'connect_timeout': 60,
        'options': '-c statement_timeout=30000'
    },
    
    # Security settings for production
    'SECURE_SSL_REDIRECT': True,
    'SESSION_COOKIE_SECURE': True,
    'CSRF_COOKIE_SECURE': True,
    'SECURE_BROWSER_XSS_FILTER': True,
    'SECURE_CONTENT_TYPE_NOSNIFF': True,
    'X_FRAME_OPTIONS': 'DENY',
}

# Instructions for applying this patch:
# 1. Replace the DATABASES setting in settings.py with AURORA_DATABASES
# 2. Add Aurora-specific configurations
# 3. Update ALLOWED_HOSTS for your EC2 instance
# 4. Set DEBUG=False for production
# 5. Update CORS settings if needed