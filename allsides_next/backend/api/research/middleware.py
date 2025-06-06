import logging
import time
import json
from django.utils.deprecation import MiddlewareMixin
from django.http import StreamingHttpResponse

logger = logging.getLogger(__name__)

class SSEDebugMiddleware(MiddlewareMixin):
    """Middleware to handle SSE connections properly."""
    
    def process_request(self, request):
        """Process request and ensure SSE connections are handled properly."""
        # Add request timestamp
        request.start_time = time.time()
        
        # Only log minimal info for SSE endpoints
        if 'stream' in request.path or 'start' in request.path or 'health' in request.path or 'approve' in request.path or 'feedback' in request.path:
            logger.info(f"SSE Request: {request.method} {request.path}")
            
            # Check if this is an EventSource request and add appropriate headers
            if 'HTTP_ACCEPT' in request.META and 'text/event-stream' in request.META['HTTP_ACCEPT']:
                request.is_sse = True
                
                # Set special attribute to ensure connection is not closed prematurely
                request._dont_enforce_csrf_checks = True
                
                # Log SSE-specific headers for debugging
                logger.debug(f"SSE Headers: {dict(request.META)}")
        
        return None
    
    def process_response(self, request, response):
        """Process response and add essential SSE headers if needed."""
        if isinstance(response, StreamingHttpResponse) and 'text/event-stream' in response.get('Content-Type', ''):
            # Only add essential headers
            response['Cache-Control'] = 'no-cache'
            response['X-Accel-Buffering'] = 'no'
            
            # Add CORS headers only if needed
            if 'HTTP_ORIGIN' in request.META:
                response['Access-Control-Allow-Origin'] = request.META['HTTP_ORIGIN']
                response['Access-Control-Allow-Credentials'] = 'true'
        
        return response 