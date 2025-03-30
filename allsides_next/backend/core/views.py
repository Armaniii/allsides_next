from django.http import HttpResponseNotFound, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from urllib.parse import urlparse, urlunparse
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def custom_page_not_found(request, exception=None):
    """
    Custom 404 handler that ensures port preservation in any URLs.
    This is especially important for admin redirects.
    """
    logger.debug(f"Custom 404 handler called for path: {request.path}")
    
    # Get the server port from various sources
    server_port = request.META.get('HTTP_X_FORWARDED_PORT', 
                  request.META.get('SERVER_PORT', 
                  getattr(settings, 'DEFAULT_HTTP_PORT', '9000')))
    
    # If this is an admin path, redirect to admin with port
    if request.path.startswith('/admin/'):
        # Get the scheme from request
        scheme = request.META.get('HTTP_X_FORWARDED_PROTO', 'http')
        
        # Get the host without port
        host = request.get_host().split(':')[0]
        
        # Construct the target URL with port
        target_url = f"{scheme}://{host}:{server_port}/admin/"
        logger.debug(f"404 for admin path, redirecting to: {target_url}")
        return HttpResponseRedirect(target_url)
    
    # For non-admin paths, render standard 404 template
    template = loader.get_template('404.html')
    context = {
        'path': request.path,
        'server_port': server_port,
    }
    return HttpResponseNotFound(template.render(context, request)) 