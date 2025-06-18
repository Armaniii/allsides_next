"""
Simple queueing mechanism for LLM requests to prevent overwhelming Ollama.
This provides visibility into queue depth and can return early if overloaded.
"""
import redis
import json
import logging
import time
from django.conf import settings
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class LLMQueueManager:
    """Manages a queue for LLM requests to ensure serialized access to Ollama"""
    
    def __init__(self):
        self.redis_client = redis.from_url(
            f"redis://{settings.REDIS_HOST}:6379/0",
            decode_responses=True
        )
        self.queue_key = "llm_request_queue"
        self.processing_key = "llm_processing_count"
        self.max_queue_size = 10  # Reject requests if queue gets too long
        
    def get_queue_depth(self) -> int:
        """Get current number of requests waiting in queue"""
        return self.redis_client.llen(self.queue_key)
    
    def get_processing_count(self) -> int:
        """Get number of requests currently being processed"""
        count = self.redis_client.get(self.processing_key)
        return int(count) if count else 0
    
    def can_accept_request(self) -> tuple[bool, str]:
        """Check if we can accept a new request"""
        queue_depth = self.get_queue_depth()
        processing = self.get_processing_count()
        
        # Total load = queued + processing
        total_load = queue_depth + processing
        
        if total_load >= self.max_queue_size:
            return False, f"System overloaded: {queue_depth} queued, {processing} processing"
        
        return True, f"Queue depth: {queue_depth}, Processing: {processing}"
    
    def acquire_processing_slot(self, timeout: float = 30.0) -> bool:
        """
        Try to acquire a processing slot. This implements a simple semaphore.
        Returns True if acquired, False if timeout.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Try to increment if we're below the limit (1 for Ollama)
            current = self.get_processing_count()
            if current < 1:  # Only allow 1 concurrent Ollama request
                # Use Redis transaction to avoid race conditions
                pipe = self.redis_client.pipeline()
                pipe.watch(self.processing_key)
                
                current = int(pipe.get(self.processing_key) or 0)
                if current < 1:
                    pipe.multi()
                    pipe.set(self.processing_key, current + 1)
                    pipe.expire(self.processing_key, 300)  # Auto-cleanup after 5 min
                    try:
                        pipe.execute()
                        logger.info(f"✅ Acquired LLM processing slot (was {current}, now {current + 1})")
                        return True
                    except redis.WatchError:
                        # Another process got there first, retry
                        pass
                
                pipe.reset()
            
            # Wait a bit before retrying
            time.sleep(0.1)
        
        logger.warning(f"⏱️ Timeout waiting for LLM processing slot after {timeout}s")
        return False
    
    def release_processing_slot(self):
        """Release the processing slot when done"""
        current = self.redis_client.decr(self.processing_key)
        logger.info(f"🔓 Released LLM processing slot (now {current})")
        
        # Ensure we don't go negative due to errors
        if current < 0:
            self.redis_client.set(self.processing_key, 0)

# Global instance
llm_queue = LLMQueueManager()

def with_llm_queue_protection(func: Callable) -> Callable:
    """
    Decorator to protect LLM calls with queue management.
    Use this around any function that calls Ollama.
    """
    def wrapper(*args, **kwargs):
        # Check if we can accept the request
        can_accept, status = llm_queue.can_accept_request()
        if not can_accept:
            logger.warning(f"🚫 Rejecting LLM request: {status}")
            raise Exception(f"Service temporarily unavailable: {status}")
        
        # Try to acquire a processing slot
        if not llm_queue.acquire_processing_slot(timeout=10.0):
            raise Exception("Service busy: Could not acquire processing slot")
        
        try:
            # Execute the actual LLM call
            return func(*args, **kwargs)
        finally:
            # Always release the slot
            llm_queue.release_processing_slot()
    
    return wrapper