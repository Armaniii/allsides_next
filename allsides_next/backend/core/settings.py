"""
Django settings for core project.

Generated by 'django-admin startproject' using Django 4.2.17.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from pathlib import Path
from datetime import timedelta
import os
from dotenv import load_dotenv
import logging
from urllib.parse import urlparse, urlunparse

# Setup logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-development-key')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True  # Temporarily enable debug mode to help diagnose the issue

# Allow all hosts for development, configure properly for production
ALLOWED_HOSTS = ['*']

# Settings to fix port preservation in redirects
USE_X_FORWARDED_HOST = True
USE_X_FORWARDED_PORT = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Force Django to detect and use the port in all requests
FORCE_PORT_IN_URL = True

# Force the admin to use the correct port in redirects
ADMIN_URL = 'admin/'
CSRF_TRUSTED_ORIGINS = [
    'http://34.134.51.8:9000',
    'http://localhost:9000',
    'http://34.134.51.8',  # Add without port for flexibility
    'http://localhost',
    # 'https://allstances.com'
]

# Additional settings to fix port preservation
SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False

# Ensures trailing slashes are handled consistently
APPEND_SLASH = True

# Reset any script name prefixes that might interfere with URLs
FORCE_SCRIPT_NAME = ''

# Explicitly set the port for redirects when needed
PORT_AWARE_REDIRECTS = True

# Set the default port to use in redirects if not specified
DEFAULT_HTTP_PORT = '9000'

# Override HttpRequest.get_host to always include port
def _custom_get_host(request):
    """Custom function to ensure get_host always includes port"""
    host = request.META.get('HTTP_HOST', '')
    if not host:
        host = request.META.get('SERVER_NAME', '')
        server_port = request.META.get('SERVER_PORT', '')
        if server_port not in ('80', '443'):
            host = '%s:%s' % (host, server_port)
    return host

# Patch Django's HttpRequest.get_host method
from django.http.request import HttpRequest
HttpRequest.get_host = _custom_get_host

# Override Django's HttpResponseRedirect classes to ensure ports are preserved
from django.http import HttpResponseRedirect, HttpResponsePermanentRedirect

# Store original __init__ methods
original_redirect_init = HttpResponseRedirect.__init__
original_permanent_redirect_init = HttpResponsePermanentRedirect.__init__

# Override HttpResponseRedirect.__init__
def port_aware_redirect_init(self, redirect_to, *args, **kwargs):
    """Override to ensure redirect URLs include port"""
    logger.debug(f"HttpResponseRedirect initialized with redirect_to: {redirect_to}")
    
    # Ensure redirect_to includes port if it's a domain without port
    if redirect_to and isinstance(redirect_to, str):
        if redirect_to.startswith('http'):
            parsed = urlparse(redirect_to)
            if ':' not in parsed.netloc and parsed.netloc:
                # Get default port from settings
                default_port = DEFAULT_HTTP_PORT
                # Only add non-standard ports
                if default_port not in ('80', '443'):
                    parts = list(parsed)
                    parts[1] = f"{parsed.netloc}:{default_port}"
                    new_redirect_to = urlunparse(parts)
                    logger.debug(f"Modified redirect URL: {redirect_to} -> {new_redirect_to}")
                    redirect_to = new_redirect_to
    
    # Call original init
    original_redirect_init(self, redirect_to, *args, **kwargs)
    logger.debug(f"Final redirect Location: {self.get('Location')}")

# Apply the patches
HttpResponseRedirect.__init__ = port_aware_redirect_init
HttpResponsePermanentRedirect.__init__ = port_aware_redirect_init
logger.debug("Applied patches to HttpResponseRedirect classes")

# Application definition

INSTALLED_APPS = [
    'corsheaders',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'core.middleware.PortPreservingMiddleware',
    'core.middleware.AdminPortPreservingMiddleware',  # Admin-specific middleware
    'api.research.middleware.SSEDebugMiddleware',  # SSE debugging middleware
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],  # Add our custom templates directory
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('POSTGRES_DB', 'allsides_db'),
        'USER': os.getenv('POSTGRES_USER', 'allsides_user'),
        'PASSWORD': os.getenv('POSTGRES_PASSWORD', 'allsides_password'),
        'HOST': 'db',  # This should match the service name in docker-compose.yml
        'PORT': '5432',
    }
}

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        'api': {  # This is your app name
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'core.middleware': {  # Add logging for our middleware
            'handlers': ['console'],
            'level': 'INFO',  # Changed from DEBUG to INFO
            'propagate': True,
        },
        'django.middleware': {  # Add specific logger for Django middleware
            'handlers': ['console'],
            'level': 'INFO',  # Set to INFO to reduce noise
            'propagate': True,
        },
        'corsheaders': {  # Add specific logger for CORS middleware
            'handlers': ['console'],
            'level': 'INFO',  # Set to INFO to reduce noise
            'propagate': True,
        },
    },
}


# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Only add STATICFILES_DIRS if the static directory exists
static_dir = os.path.join(BASE_DIR, 'static')
if os.path.exists(static_dir):
    STATICFILES_DIRS = [static_dir]

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'mediafiles')

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Custom user model
AUTH_USER_MODEL = 'api.User'

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_TIMEOUT': 300,  # 5 minutes timeout for API requests
}

# CORS settings
CORS_ORIGIN_ALLOW_ALL = True
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

CORS_EXPOSE_HEADERS = [
    'Content-Type', 
    'X-CSRFToken',
    'Authorization',
    'Access-Control-Allow-Origin',
    'Access-Control-Allow-Credentials'
]

CORS_PREFLIGHT_MAX_AGE = 86400

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://34.134.51.8:3000",
    "http://34.134.51.8:9000",  # Added port 9000
    "http://34.134.51.8",
    "http://34.134.51.8:80",
    "https://allstances.com",
    "https://www.allstances.com"
]

CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
    'access-control-allow-origin',
    'access-control-allow-credentials',
    'access-control-allow-headers',
    'access-control-allow-methods'
]

# Security settings
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://34.134.51.8:3001",
    "http://34.134.51.8",
    "http://34.134.51.8:80",
    "https://allstances.com",
    "https://www.allstances.com"
]

# Add CORS middleware settings
# CORS_REPLACE_HTTPS_REFERER = True
CORS_URLS_REGEX = r'^/api/.*$'

# JWT settings
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(days=1),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'UPDATE_LAST_LOGIN': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'VERIFYING_KEY': None,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
    'TOKEN_TYPE_CLAIM': 'token_type',
}

# Security settings when DEBUG is False
if not DEBUG:
    SECURE_HSTS_SECONDS = 31536000  # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
