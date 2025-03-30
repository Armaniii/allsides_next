from django.contrib.admin import AdminSite
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.conf import settings
from django.shortcuts import redirect
import logging
import re
from urllib.parse import urlparse, urlunparse, urljoin
from django.contrib.admin.sites import site as admin_site
from django.contrib.auth import views as auth_views
from functools import wraps

logger = logging.getLogger(__name__)

# Monkey patch Django admin views to ensure port preservation
def patch_admin_views():
    """Monkey patch Django admin views to ensure port preservation in redirects"""
    # Store original login view
    original_login = auth_views.LoginView.as_view(
        template_name='admin/login.html',
        extra_context={'site_title': admin_site.site_title, 'site_header': admin_site.site_header}
    )

    @wraps(original_login)
    def port_aware_login(request, *args, **kwargs):
        """Wrap admin login view to ensure port preservation"""
        logger.debug("Using port-aware admin login view")
        response = original_login(request, *args, **kwargs)
        
        # Fix the response if it's a redirect
        if hasattr(response, 'status_code') and 300 <= response.status_code < 400:
            if 'Location' in response:
                location = response['Location']
                logger.debug(f"Admin login redirect detected: {location}")
                
                # Get the server port
                server_port = request.META.get('HTTP_X_FORWARDED_PORT', 
                             request.META.get('SERVER_PORT', 
                             getattr(settings, 'DEFAULT_HTTP_PORT', '9000')))
                
                # Fix absolute URLs to include port
                if location.startswith(('http://', 'https://')):
                    parsed = urlparse(location)
                    # Only add port if it's missing and not standard
                    if ':' not in parsed.netloc and server_port not in ('80', '443'):
                        parts = list(parsed)
                        parts[1] = f"{parsed.netloc}:{server_port}"
                        new_location = urlunparse(parts)
                        logger.debug(f"Fixed admin login redirect: {location} -> {new_location}")
                        response['Location'] = new_location
                
                # Convert relative URLs to absolute with port
                elif location.startswith('/'):
                    scheme = request.META.get('HTTP_X_FORWARDED_PROTO', 'http')
                    host = request.get_host().split(':')[0]  # Get host without port
                    new_location = f"{scheme}://{host}:{server_port}{location}"
                    logger.debug(f"Converted relative admin login redirect: {location} -> {new_location}")
                    response['Location'] = new_location
        
        return response
    
    # Get the admin login URL pattern and update its view
    from django.contrib import admin
    for pattern in admin.site.urls[0]:
        if hasattr(pattern, 'pattern') and 'login' in str(pattern.pattern):
            # Store original callback
            original_callback = pattern.callback
            # Replace with our port-aware version
            pattern.callback = port_aware_login
            logger.debug("Applied port-aware patch to admin login view")
            break
    
    # Store original logout view
    original_logout = auth_views.LogoutView.as_view(
        template_name='admin/logged_out.html',
        extra_context={'site_title': admin_site.site_title, 'site_header': admin_site.site_header}
    )

    @wraps(original_logout)
    def port_aware_logout(request, *args, **kwargs):
        """Wrap admin logout view to ensure port preservation"""
        logger.debug("Using port-aware admin logout view")
        response = original_logout(request, *args, **kwargs)
        
        # Fix the response if it's a redirect
        if hasattr(response, 'status_code') and 300 <= response.status_code < 400:
            if 'Location' in response:
                location = response['Location']
                logger.debug(f"Admin logout redirect detected: {location}")
                
                # Get the server port
                server_port = request.META.get('HTTP_X_FORWARDED_PORT', 
                             request.META.get('SERVER_PORT', 
                             getattr(settings, 'DEFAULT_HTTP_PORT', '9000')))
                
                # Fix absolute URLs to include port
                if location.startswith(('http://', 'https://')):
                    parsed = urlparse(location)
                    # Only add port if it's missing and not standard
                    if ':' not in parsed.netloc and server_port not in ('80', '443'):
                        parts = list(parsed)
                        parts[1] = f"{parsed.netloc}:{server_port}"
                        new_location = urlunparse(parts)
                        logger.debug(f"Fixed admin logout redirect: {location} -> {new_location}")
                        response['Location'] = new_location
                
                # Convert relative URLs to absolute with port
                elif location.startswith('/'):
                    scheme = request.META.get('HTTP_X_FORWARDED_PROTO', 'http')
                    host = request.get_host().split(':')[0]  # Get host without port
                    new_location = f"{scheme}://{host}:{server_port}{location}"
                    logger.debug(f"Converted relative admin logout redirect: {location} -> {new_location}")
                    response['Location'] = new_location
        
        return response
    
    # Get the admin logout URL pattern and update its view
    for pattern in admin.site.urls[0]:
        if hasattr(pattern, 'pattern') and 'logout' in str(pattern.pattern):
            # Store original callback
            original_callback = pattern.callback
            # Replace with our port-aware version
            pattern.callback = port_aware_logout
            logger.debug("Applied port-aware patch to admin logout view")
            break
    
    logger.debug("Finished patching admin views for port preservation")

try:
    # Execute the monkey patching
    patch_admin_views()
    logger.info("Successfully applied admin view patches for port preservation")
except Exception as e:
    logger.error(f"Failed to patch admin views: {str(e)}")

class PortPreservingAdminSite(AdminSite):
    """
    Custom admin site that ensures all URLs maintain the port number.
    This addresses the issue with admin redirects when behind a proxy.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("Initializing PortPreservingAdminSite")
        
        # Force port in all URLs generated by this admin site
        self.force_port_in_urls = True
    
    def get_app_list(self, request, app_label=None):
        """Ensure all admin URLs in the app list have the port included"""
        app_list = super().get_app_list(request, app_label)
        
        # Get server port from request or settings
        server_port = request.META.get('HTTP_X_FORWARDED_PORT',
                     request.META.get('SERVER_PORT', '9000'))
        
        # Always ensure all URLs in app_list have the port
        for app in app_list:
            # Fix app URL
            app['app_url'] = self._add_port_to_url(request, app['app_url'], server_port)
            
            # Fix model URLs
            for model in app.get('models', []):
                model['admin_url'] = self._add_port_to_url(request, model['admin_url'], server_port)
                model['add_url'] = self._add_port_to_url(request, model['add_url'], server_port)
        
        return app_list
    
    def _add_port_to_url(self, request, url, port):
        """Add port to URL if it's not already there"""
        if not url or not isinstance(url, str):
            return url
            
        # For relative URLs, convert to absolute with port
        if url.startswith('/'):
            scheme = request.META.get('HTTP_X_FORWARDED_PROTO', 
                      request.META.get('wsgi.url_scheme', 'http'))
            host = request.get_host().split(':')[0]  # Get host without port
            base_url = f"{scheme}://{host}:{port}" if port not in ('80', '443') else f"{scheme}://{host}"
            return urljoin(base_url, url)
            
        # For absolute URLs, ensure port is included
        if url.startswith(('http://', 'https://')):
            parsed_url = urlparse(url)
            netloc = parsed_url.netloc
            
            # If netloc doesn't have a port, add it
            if ':' not in netloc and port not in ('80', '443'):
                new_netloc = f"{netloc}:{port}"
                parts = list(parsed_url)
                parts[1] = new_netloc
                return urlunparse(parts)
            
        return url
    
    def login(self, request, extra_context=None):
        """Override login to ensure port is preserved in redirect URLs"""
        # Log all request headers for debugging
        logger.debug("Admin login request received")
        headers = {k: v for k, v in request.META.items() if k.startswith('HTTP_')}
        logger.debug(f"Admin login request headers: {headers}")
        
        # Extract the server port
        server_port = request.META.get('HTTP_X_FORWARDED_PORT',
                     request.META.get('SERVER_PORT', '9000'))
        logger.debug(f"Server port: {server_port}")
        
        # Get the host from forwarded headers or request
        if 'HTTP_X_FORWARDED_HOST' in request.META:
            host_with_port = request.META['HTTP_X_FORWARDED_HOST']
            logger.debug(f"Using X-Forwarded-Host: {host_with_port}")
        elif 'HTTP_HOST' in request.META:
            host_with_port = request.META['HTTP_HOST']
            logger.debug(f"Using HTTP_HOST: {host_with_port}")
        else:
            # Fallback to request host
            host = request.get_host().split(':')[0]  # Remove port if present
            host_with_port = f"{host}:{server_port}" if server_port not in ('80', '443') else host
            logger.debug(f"Using constructed host: {host_with_port}")
        
        # Always ensure host_with_port includes the port
        if ':' not in host_with_port and server_port not in ('80', '443'):
            host_with_port = f"{host_with_port}:{server_port}"
            logger.debug(f"Added port to host: {host_with_port}")
        
        # Add port information to extra_context
        extra_context = extra_context or {}
        extra_context.update({
            'server_port': server_port,
            'server_host': host_with_port.split(':')[0],
            'host_with_port': host_with_port,
            'current_path': request.get_full_path(),
            'absolute_uri': request.build_absolute_uri(),
        })
        
        # Call parent login method
        response = super().login(request, extra_context)
        
        # Fix all redirects to include port
        if hasattr(response, 'status_code') and 300 <= response.status_code < 400 and 'Location' in response:
            original_location = response['Location']
            logger.debug(f"Original login redirect location: {original_location}")
            
            # Convert all redirects to absolute URLs with port
            if original_location.startswith('/'):
                # Always make absolute
                scheme = request.META.get('HTTP_X_FORWARDED_PROTO', 'http')
                new_location = f"{scheme}://{host_with_port}{original_location}"
                logger.debug(f"Converting relative redirect to absolute: {new_location}")
                response['Location'] = new_location
            elif original_location.startswith(('http://', 'https://')):
                # Ensure port is in absolute URL
                parsed = urlparse(original_location)
                # Only add port if it's our domain and port is missing
                if ':' not in parsed.netloc and server_port not in ('80', '443'):
                    host = parsed.netloc
                    scheme = parsed.scheme
                    path = parsed.path
                    query = parsed.query
                    fragment = parsed.fragment
                    
                    # Build new URL with port
                    new_netloc = f"{host}:{server_port}"
                    parts = [scheme, new_netloc, path, '', query, fragment]
                    new_location = urlunparse(parts)
                    logger.debug(f"Adding port to absolute redirect: {new_location}")
                    response['Location'] = new_location
            
            logger.debug(f"Final login redirect location: {response['Location']}")
        
        return response
    
    def each_context(self, request):
        """Add custom context for admin templates"""
        context = super().each_context(request)
        
        # Add port information to the context
        server_port = request.META.get('HTTP_X_FORWARDED_PORT',
                     request.META.get('SERVER_PORT', '9000'))
        
        # Extract host without port
        host = request.get_host()
        if ':' in host:
            host, _ = host.split(':', 1)
        
        # Create host with port
        host_with_port = f"{host}:{server_port}" if server_port not in ('80', '443') else host
        
        # Add to context
        context.update({
            'server_port': server_port,
            'server_host': host,
            'host_with_port': host_with_port,
            'absolute_uri': request.build_absolute_uri(),
            'current_path': request.get_full_path(),
        })
        
        logger.debug(f"Admin context: port={server_port}, host={host}, host_with_port={host_with_port}")
        return context
    
    def _get_admin_url(self, request, url_name, args=None):
        """Override to ensure admin URLs always include port"""
        url = super()._get_admin_url(request, url_name, args)
        
        # Get the server port from the request
        server_port = request.META.get('HTTP_X_FORWARDED_PORT',
                     request.META.get('SERVER_PORT', '9000'))
        
        # Always make admin URLs absolute with port
        if url.startswith('/'):
            scheme = request.META.get('HTTP_X_FORWARDED_PROTO', 'http')
            host = request.get_host().split(':')[0]  # Get host without port
            base_url = f"{scheme}://{host}:{server_port}" if server_port not in ('80', '443') else f"{scheme}://{host}"
            url = urljoin(base_url, url)
            
        # For absolute URLs, ensure they have port
        elif url.startswith(('http://', 'https://')):
            parsed = urlparse(url)
            if ':' not in parsed.netloc and server_port not in ('80', '443'):
                parts = list(parsed)
                parts[1] = f"{parsed.netloc}:{server_port}"
                url = urlunparse(parts)
        
        logger.debug(f"Generated admin URL: {url}")
        return url
    
    def app_index(self, request, app_label, extra_context=None):
        """
        Override app_index to ensure port is preserved in redirect URLs
        """
        # Get the standard response first
        response = super().app_index(request, app_label, extra_context)
        
        # If response is a redirect, ensure it has the correct port
        if response.status_code in (301, 302, 303, 307, 308) and 'Location' in response:
            server_port = request.META.get('HTTP_X_FORWARDED_PORT',
                         request.META.get('SERVER_PORT', '9000'))
            
            original_location = response['Location']
            logger.debug(f"App index redirect: Original location: {original_location}")
            
            # Handle all URLs - always make absolute with port
            if original_location.startswith('/'):
                # Get the scheme and host
                scheme = request.META.get('HTTP_X_FORWARDED_PROTO', 'http')
                host = request.get_host().split(':')[0]  # Get host without port
                
                # Create absolute URL with port
                new_location = f"{scheme}://{host}:{server_port}{original_location}" if server_port not in ('80', '443') else f"{scheme}://{host}{original_location}"
                logger.debug(f"Converting app index relative redirect to absolute: {new_location}")
                response['Location'] = new_location
            # Handle absolute URLs
            elif original_location.startswith(('http://', 'https://')):
                parsed = urlparse(original_location)
                # Only add port if it's missing and not a standard port
                if ':' not in parsed.netloc and server_port not in ('80', '443'):
                    parts = list(parsed)
                    parts[1] = f"{parsed.netloc}:{server_port}"
                    new_location = urlunparse(parts)
                    logger.debug(f"Adding port to app index absolute redirect: {new_location}")
                    response['Location'] = new_location
            
            logger.debug(f"App index final redirect location: {response['Location']}")
            
        return response
        
    def response_post_save_add(self, request, obj):
        """Override to ensure port is preserved after adding objects"""
        response = super().response_post_save_add(request, obj)
        return self._fix_redirect_response(request, response)
        
    def response_post_save_change(self, request, obj):
        """Override to ensure port is preserved after editing objects"""
        response = super().response_post_save_change(request, obj)
        return self._fix_redirect_response(request, response)
        
    def response_delete(self, request, obj_display, obj_id):
        """Override to ensure port is preserved after deleting objects"""
        response = super().response_delete(request, obj_display, obj_id)
        return self._fix_redirect_response(request, response)
    
    def _fix_redirect_response(self, request, response):
        """Fix any redirect response to include port"""
        if hasattr(response, 'status_code') and 300 <= response.status_code < 400 and 'Location' in response:
            server_port = request.META.get('HTTP_X_FORWARDED_PORT',
                         request.META.get('SERVER_PORT', '9000'))
            
            original_location = response['Location']
            logger.debug(f"Fixing redirect response: {original_location}")
            
            # Force all redirects to be absolute with port
            if original_location.startswith('/'):
                scheme = request.META.get('HTTP_X_FORWARDED_PROTO', 'http')
                host = request.get_host().split(':')[0]
                new_location = f"{scheme}://{host}:{server_port}{original_location}" if server_port not in ('80', '443') else f"{scheme}://{host}{original_location}"
                logger.debug(f"Converting relative redirect to absolute: {new_location}")
                response['Location'] = new_location
            elif original_location.startswith(('http://', 'https://')):
                parsed = urlparse(original_location)
                if ':' not in parsed.netloc and server_port not in ('80', '443'):
                    parts = list(parsed)
                    parts[1] = f"{parsed.netloc}:{server_port}"
                    new_location = urlunparse(parts)
                    logger.debug(f"Adding port to absolute redirect: {new_location}")
                    response['Location'] = new_location
        
        return response
    
    def index(self, request, extra_context=None):
        """Override index view to ensure all URLs have port"""
        response = super().index(request, extra_context)
        if hasattr(response, 'render'):
            # Store original render method
            original_render = response.render
            
            # Override render to fix URLs in the rendered content
            def port_aware_render(content_only=False):
                response_content = original_render(content_only)
                
                # Only fix HTML content
                content_type = response.get('Content-Type', '')
                if 'text/html' in content_type:
                    # Get the port
                    server_port = request.META.get('HTTP_X_FORWARDED_PORT',
                                 request.META.get('SERVER_PORT', '9000'))
                    
                    # Fix URLs in content
                    if hasattr(response, 'content') and server_port not in ('80', '443'):
                        try:
                            content_str = response.content.decode('utf-8')
                            
                            # Fix URLs in href attributes - absolute URLs without port
                            pattern = r'href=["\'](https?://[^:/]+)(/[^"\']*)["\']'
                            replacement = f'href="\\1:{server_port}\\2"'
                            content_str = re.sub(pattern, replacement, content_str)
                            
                            # Fix URLs in src attributes
                            pattern = r'src=["\'](https?://[^:/]+)(/[^"\']*)["\']'
                            replacement = f'src="\\1:{server_port}\\2"'
                            content_str = re.sub(pattern, replacement, content_str)
                            
                            # Fix URLs in action attributes
                            pattern = r'action=["\'](https?://[^:/]+)(/[^"\']*)["\']'
                            replacement = f'action="\\1:{server_port}\\2"'
                            content_str = re.sub(pattern, replacement, content_str)
                            
                            # Update the response content
                            response.content = content_str.encode('utf-8')
                            logger.debug("Fixed URLs in admin index response content")
                        except Exception as e:
                            logger.error(f"Error fixing URLs in admin index content: {str(e)}")
                
                return response_content
            
            # Replace the render method
            response.render = port_aware_render
        
        return response 