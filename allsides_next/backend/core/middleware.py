import re
from django.conf import settings
from django.utils.deprecation import MiddlewareMixin
import logging
import json
from urllib.parse import urlparse, urlunparse, urljoin

logger = logging.getLogger(__name__)

class PortPreservingMiddleware(MiddlewareMixin):
    """
    Middleware to ensure port is preserved in all redirects.
    This is especially important for the admin interface.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        logger.debug("PortPreservingMiddleware initialized")
    
    def __call__(self, request):
        # Process request
        request = self.process_request(request)
        
        # Get response
        response = self.get_response(request)
        
        # Process response
        response = self.process_response(request, response)
        
        return response
    
    def process_request(self, request):
        """Log incoming request information for debugging"""
        host = request.get_host()
        path = request.path
        
        # Enhanced logging for debugging
        logger.debug(f"=====================================================")
        logger.debug(f"REQUEST: {request.method} {host}{path}")
        logger.debug(f"request.get_host() returns: {host}")
        logger.debug(f"request.build_absolute_uri() returns: {request.build_absolute_uri()}")
        logger.debug(f"X-Forwarded-Host: {request.META.get('HTTP_X_FORWARDED_HOST', 'Not set')}")
        logger.debug(f"X-Forwarded-Port: {request.META.get('HTTP_X_FORWARDED_PORT', 'Not set')}")
        logger.debug(f"Host header: {request.META.get('HTTP_HOST', 'Not set')}")
        logger.debug(f"Server Port: {request.META.get('SERVER_PORT', 'Not set')}")
        
        # Always ensure server port is available in the request
        # This fixes issues with various Django functions that use get_host()
        self._ensure_server_port(request)
        
        # If we're accessing the admin, ensure all necessary headers are set
        if path.startswith('/admin/'):
            logger.debug("Admin request detected, ensuring all necessary headers are set")
            
            # Get the server port from all possible sources
            server_port = request.META.get('HTTP_X_FORWARDED_PORT', 
                          request.META.get('SERVER_PORT',
                          getattr(settings, 'DEFAULT_HTTP_PORT', '9000')))
            
            # Extract host and fix port if needed
            host_header = request.META.get('HTTP_HOST', request.get_host())
            
            # If host has a port, extract it
            if ':' in host_header:
                host, port = host_header.split(':')
                # Use this port if available
                server_port = port
            else:
                host = host_header
                
            # Always set the forwarded headers
            request.META['HTTP_X_FORWARDED_HOST'] = f"{host}:{server_port}"
            request.META['HTTP_X_FORWARDED_PORT'] = server_port
            # Also set standard HTTP_HOST header
            request.META['HTTP_HOST'] = f"{host}:{server_port}"
            
            logger.debug(f"Set forwarded headers: X-Forwarded-Host={request.META['HTTP_X_FORWARDED_HOST']}, X-Forwarded-Port={server_port}")
        
        # Log all headers for debugging
        headers = {k: v for k, v in request.META.items() if k.startswith('HTTP_')}
        logger.debug(f"All request headers: {json.dumps(headers, indent=2)}")
        logger.debug(f"After processing, request.get_host() returns: {request.get_host()}")
        logger.debug(f"=====================================================")
        
        return request
    
    def _ensure_server_port(self, request):
        """Ensure server port is available and consistent in request"""
        # Get the server port from all possible sources
        server_port = request.META.get('HTTP_X_FORWARDED_PORT', 
                      request.META.get('SERVER_PORT',
                      getattr(settings, 'DEFAULT_HTTP_PORT', '9000')))
        
        # Ensure HTTP_HOST has port if needed
        host_header = request.META.get('HTTP_HOST', request.get_host())
        if ':' not in host_header and server_port not in ('80', '443'):
            request.META['HTTP_HOST'] = f"{host_header}:{server_port}"
            
        # Always set X_FORWARDED_PORT for consistent behavior
        request.META['HTTP_X_FORWARDED_PORT'] = server_port
        
        # Set SERVER_PORT for Django internal functions
        request.META['SERVER_PORT'] = server_port
    
    def _fix_absolute_url(self, url, port):
        """Fix absolute URL to include the port"""
        if not url or not isinstance(url, str):
            return url
            
        # Skip URLs that aren't HTTP/HTTPS
        if not url.startswith(('http://', 'https://')):
            return url
            
        # Parse the URL
        parsed_url = urlparse(url)
        netloc = parsed_url.netloc
        
        # If there's already a port in the netloc, don't change it
        if ':' in netloc:
            return url
            
        # Skip standard ports (80 for HTTP, 443 for HTTPS) if scheme matches
        if (parsed_url.scheme == 'http' and port == '80') or (parsed_url.scheme == 'https' and port == '443'):
            return url
            
        # Add the port to the netloc
        new_netloc = f"{netloc}:{port}"
        
        # Rebuild the URL with the new netloc
        parts = list(parsed_url)
        parts[1] = new_netloc
        return urlunparse(parts)
    
    def _make_absolute_url(self, request, path, port):
        """Convert a relative path to an absolute URL with port"""
        if not path:
            return path
            
        # If it's already an absolute URL, just ensure port is present
        if path.startswith(('http://', 'https://')):
            return self._fix_absolute_url(path, port)
            
        # Get scheme from request
        scheme = request.META.get('HTTP_X_FORWARDED_PROTO', 
                 request.META.get('wsgi.url_scheme', 'http'))
        
        # Get host from request, without port
        host_with_port = request.get_host()
        if ':' in host_with_port:
            host = host_with_port.split(':')[0]
        else:
            host = host_with_port
        
        # Build absolute URL with port
        base_url = f"{scheme}://{host}:{port}" if port not in ('80', '443') else f"{scheme}://{host}"
        return urljoin(base_url, path)
    
    def process_response(self, request, response):
        """Ensure redirects preserve the port number"""
        # Get the server port from various sources
        server_port = request.META.get('HTTP_X_FORWARDED_PORT', 
                      request.META.get('SERVER_PORT', 
                      getattr(settings, 'DEFAULT_HTTP_PORT', '9000')))
        
        # Log all responses for debugging
        if hasattr(response, 'status_code'):
            logger.debug(f"=====================================================")
            logger.debug(f"RESPONSE: Status {response.status_code} for {request.method} {request.get_host()}{request.path}")
            
            # Log all headers for debugging
            if hasattr(response, 'items'):
                headers = {k: v for k, v in response.items()}
                logger.debug(f"Response headers: {json.dumps(headers, indent=2)}")
        
        # Only process redirect responses
        if hasattr(response, 'status_code') and 300 <= response.status_code < 400:
            logger.debug(f"Detected redirect response with status {response.status_code}")
            # Fix Location header if present
            if 'Location' in response:
                original_location = response['Location']
                logger.debug(f"Processing redirect Location: {original_location}")
                
                # Get the host from various sources
                if 'HTTP_X_FORWARDED_HOST' in request.META:
                    host_with_port = request.META['HTTP_X_FORWARDED_HOST']
                    logger.debug(f"Using X-Forwarded-Host: {host_with_port}")
                elif 'HTTP_HOST' in request.META:
                    host_with_port = request.META['HTTP_HOST']
                    logger.debug(f"Using HTTP_HOST: {host_with_port}")
                else:
                    host = request.get_host().split(':')[0]  # Remove port if present
                    host_with_port = f"{host}:{server_port}" if server_port not in ('80', '443') else host
                    logger.debug(f"Using constructed host: {host_with_port}")
                
                # Ensure host_with_port includes the port if it's not already there
                if ':' not in host_with_port and server_port not in ('80', '443'):
                    host_with_port = f"{host_with_port}:{server_port}"
                    logger.debug(f"Added port to host: {host_with_port}")
                
                # First strategy: For absolute URLs including http/https
                if original_location.startswith(('http://', 'https://')):
                    fixed_location = self._fix_absolute_url(original_location, server_port)
                    if fixed_location != original_location:
                        logger.debug(f"Fixed absolute URL from {original_location} to {fixed_location}")
                        response['Location'] = fixed_location
                
                # Second strategy: For URLs that are relative to the host (start with /)
                elif original_location.startswith('/'):
                    scheme = request.META.get('HTTP_X_FORWARDED_PROTO', 
                             request.META.get('wsgi.url_scheme', 'http'))
                    
                    # Ensure we preserve any query params or fragments
                    parsed_url = urlparse(original_location)
                    path = parsed_url.path
                    query = parsed_url.query
                    fragment = parsed_url.fragment
                    
                    # Build the absolute URL with port
                    if query:
                        path = f"{path}?{query}"
                    if fragment:
                        path = f"{path}#{fragment}"
                    
                    new_location = f"{scheme}://{host_with_port}{path}"
                    logger.debug(f"Converting relative redirect from {original_location} to {new_location}")
                    response['Location'] = new_location
                
                # Log the final redirect location
                logger.debug(f"Final redirect location: {response['Location']}")
                
                # Super extra check for admin redirects
                if '/admin/' in response['Location']:
                    final_location = response['Location']
                    parsed = urlparse(final_location)
                    
                    # One more time, make sure the port is included if not on standard ports
                    if ':' not in parsed.netloc and server_port not in ('80', '443'):
                        # Explicitly add port to netloc
                        parts = list(parsed)
                        parts[1] = f"{parsed.netloc}:{server_port}"
                        final_location = urlunparse(parts)
                        logger.debug(f"Fixed admin URL: {response['Location']} -> {final_location}")
                        response['Location'] = final_location
                
                # Log the final-final redirect location
                logger.debug(f"Final-final redirect location: {response['Location']}")
        
        # Fix URL patterns in response content
        if hasattr(response, 'content') and isinstance(response.content, bytes):
            try:
                # Only process HTML responses
                content_type = response.get('Content-Type', '')
                if 'text/html' in content_type:
                    content_str = response.content.decode('utf-8')
                    
                    # Match absolute URLs in HTML attributes
                    pattern = r'(href|src|action)=["\']((https?://[^:/]+)/[^"\']*)["\']'
                    
                    def replace_url(match):
                        attr = match.group(1)
                        url = match.group(2)
                        domain = match.group(3)
                        
                        # Only fix URLs for our domain
                        request_host = request.get_host().split(':')[0]
                        url_host = urlparse(url).netloc.split(':')[0]
                        
                        if url_host == request_host:
                            fixed_url = self._fix_absolute_url(url, server_port)
                            return f'{attr}="{fixed_url}"'
                        return match.group(0)
                    
                    # Replace matching URLs
                    modified_content = re.sub(pattern, replace_url, content_str)
                    
                    # If content was modified, update the response
                    if modified_content != content_str:
                        response.content = modified_content.encode('utf-8')
                        logger.debug(f"Fixed URLs in HTML content")
            
            except Exception as e:
                logger.error(f"Error fixing URLs in content: {str(e)}")
        
        logger.debug(f"=====================================================")
        return response

class AdminPortPreservingMiddleware(MiddlewareMixin):
    """
    Middleware specifically for handling admin redirects to ensure port preservation.
    This middleware should be placed after PortPreservingMiddleware in the MIDDLEWARE
    settings to provide additional handling for admin-specific redirects.
    """
    
    def process_request(self, request):
        """Process admin requests to ensure port preservation"""
        # Only process admin requests
        if request.path.startswith('/admin/'):
            logger.debug(f"AdminPortPreservingMiddleware processing: {request.path}")
            
            # Set port information in request
            server_port = request.META.get('HTTP_X_FORWARDED_PORT', 
                          request.META.get('SERVER_PORT', '9000'))
                
            # Store for later use
            request._admin_server_port = server_port
            logger.debug(f"Set request._admin_server_port to {server_port}")
            
            # Ensure host header includes port
            host_header = request.META.get('HTTP_HOST', '')
            if host_header and ':' not in host_header and server_port not in ('80', '443'):
                request.META['HTTP_HOST'] = f"{host_header}:{server_port}"
                logger.debug(f"Set HTTP_HOST to {request.META['HTTP_HOST']}")
            
            # Set X-Forwarded headers
            host = request.get_host().split(':')[0]  # Get host without port
            request.META['HTTP_X_FORWARDED_HOST'] = f"{host}:{server_port}"
            request.META['HTTP_X_FORWARDED_PORT'] = server_port
            logger.debug(f"Set forwarded headers: Host={request.META['HTTP_X_FORWARDED_HOST']}, Port={server_port}")
        
        return None
    
    def process_response(self, request, response):
        """Special handling for admin redirects to ensure port preservation"""
        # Only process admin requests or redirects to admin
        if not (request.path.startswith('/admin/') or 
                (hasattr(response, 'status_code') and 
                 300 <= response.status_code < 400 and 
                 'Location' in response and 
                 '/admin/' in response['Location'])):
            return response
            
        logger.debug(f"AdminPortPreservingMiddleware processing response for {request.path}")
        
        if not hasattr(response, 'status_code'):
            return response
            
        # Only process redirects
        if not (300 <= response.status_code < 400 and 'Location' in response):
            return response
            
        location = response['Location']
        logger.debug(f"AdminPortPreservingMiddleware: Processing redirect: {location}")
        
        # Get the server port
        server_port = getattr(request, '_admin_server_port', 
                     request.META.get('HTTP_X_FORWARDED_PORT', 
                     request.META.get('SERVER_PORT', '9000')))
        
        # Handle absolute URLs (starting with http:// or https://)
        if location.startswith(('http://', 'https://')):
            parsed = urlparse(location)
            # Only add port if it's missing and not standard
            if ':' not in parsed.netloc and server_port not in ('80', '443'):
                parts = list(parsed)
                parts[1] = f"{parsed.netloc}:{server_port}"
                new_location = urlunparse(parts)
                logger.debug(f"Modified absolute admin redirect: {location} -> {new_location}")
                response['Location'] = new_location
        
        # Handle relative URLs (starting with /)
        elif location.startswith('/'):
            # If it's an admin URL, convert to absolute
            if location.startswith('/admin/'):
                # Get scheme and host from request
                scheme = request.META.get('HTTP_X_FORWARDED_PROTO', 'http')
                host = request.get_host().split(':')[0]  # Get host without port
                
                # Build absolute URL with port
                new_location = f"{scheme}://{host}:{server_port}{location}"
                logger.debug(f"Converted relative admin redirect: {location} -> {new_location}")
                response['Location'] = new_location
        
        return response 