import time
from django.core.cache import cache
from django.conf import settings
import logging
from .models import CacheStatistics
from django.utils import timezone
from datetime import timedelta

logger = logging.getLogger(__name__)

class CacheMonitoringMiddleware:
    """Middleware to monitor cache performance."""
    
    def __init__(self, get_response):
        self.get_response = get_response
        self._last_stats_update = timezone.now()
        self._stats_update_interval = timedelta(minutes=5)  # Update stats every 5 minutes
        self._hits = 0
        self._misses = 0

    def __call__(self, request):
        # Only monitor cache for API requests
        if request.path.startswith('/api/'):
            start_time = time.time()
            
            # Get initial cache stats
            initial_hits = cache.get('cache_hits', 0)
            initial_misses = cache.get('cache_misses', 0)
            
            # Process the request
            response = self.get_response(request)
            
            # Get final cache stats
            final_hits = cache.get('cache_hits', 0)
            final_misses = cache.get('cache_misses', 0)
            
            # Update counters
            self._hits += final_hits - initial_hits
            self._misses += final_misses - initial_misses
            
            # Calculate request timing
            duration = time.time() - start_time
            
            # Add cache performance headers
            response['X-Cache-Time'] = str(duration)
            response['X-Cache-Hits'] = str(final_hits - initial_hits)
            response['X-Cache-Misses'] = str(final_misses - initial_misses)
            
            # Update statistics periodically
            self._update_statistics()
            
            return response
        return self.get_response(request)
    
    def _update_statistics(self):
        """Update cache statistics in the database."""
        now = timezone.now()
        if now - self._last_stats_update >= self._stats_update_interval:
            try:
                total_requests = self._hits + self._misses
                hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
                
                CacheStatistics.objects.create(
                    cache_hits=self._hits,
                    cache_misses=self._misses,
                    hit_rate=hit_rate,
                    total_entries=cache.get('total_cached_entries', 0),
                    memory_usage=cache.get('memory_usage', 0)
                )
                
                # Reset counters
                self._hits = 0
                self._misses = 0
                self._last_stats_update = now
                
                logger.info(f"Cache statistics updated: hit_rate={hit_rate:.2f}%")
            except Exception as e:
                logger.error(f"Error updating cache statistics: {str(e)}") 