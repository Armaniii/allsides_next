import json
import logging
import hashlib
import redis
import pickle
import os
from typing import Dict, Any, Optional, List
from functools import wraps
from datetime import timedelta
import time

from django.conf import settings

# Fix imports to work with current LangChain
try:
    # Try loading from langchain_core first (new style)
    from langchain_core.messages import HumanMessage, SystemMessage
    USING_LANGCHAIN_CORE = True
except ImportError:
    # Fall back to old location
    try:
        from langchain.schema.messages import HumanMessage, SystemMessage
        USING_LANGCHAIN_CORE = False
    except ImportError:
        # Last resort to very old location
        from langchain.chat_models.base import HumanMessage, SystemMessage
        USING_LANGCHAIN_CORE = False

# Import the correct chat model initializer
try:
    # First try the new modular imports
    from langchain_openai import ChatOpenAI
    from langchain.chat_models import init_chat_model
    USING_LANGCHAIN_OPENAI = True
except ImportError:
    # Fall back to old monolithic imports
    from langchain.chat_models import ChatOpenAI, init_chat_model
    USING_LANGCHAIN_OPENAI = False

# Log which imports we're using
logger = logging.getLogger(__name__)

# Redis key prefixes
THREAD_PREFIX = "research:thread:"
REPORT_PREFIX = "research:report:"
CACHE_PREFIX = "cache:"
CACHE_TTL = 60 * 60 * 24  # 24 hours

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")  # Default to docker service name
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

# Connection pool for Redis
REDIS_POOL = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=False,  # Keep as bytes for pickle serialization
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True
)

def get_redis_client() -> redis.Redis:
    """Get a Redis client with direct connection pool."""
    try:
        # Use our own connection pool instead of main_v3
        return redis.Redis(connection_pool=REDIS_POOL)
    except Exception as e:
        logger.error(f"Redis connection error: {str(e)}")
        # Return a dummy client that won't crash but will log errors
        return DummyRedisClient()

class DummyRedisClient:
    """Dummy Redis client that logs errors but doesn't crash."""
    
    def get(self, key):
        logger.error("Using dummy Redis client - no connection available")
        return None
        
    def set(self, key, value, ex=None):
        logger.error("Using dummy Redis client - no connection available")
        return False
        
    def delete(self, key):
        logger.error("Using dummy Redis client - no connection available")
        return 0

def with_redis_client(func):
    """Decorator to handle Redis client acquisition and error handling."""
    from ..main_v3 import with_redis_client as main_with_redis_client
    return main_with_redis_client(func)

def generate_cache_key(topic: str, config: Dict[str, Any]) -> str:
    """Generate a cache key for a research request.
    
    Args:
        topic: The research topic
        config: The configuration settings
        
    Returns:
        A unique cache key
    """
    # Create a deterministic representation of the config
    config_str = json.dumps(config, sort_keys=True)
    
    # Normalize the topic (lowercase, trim whitespace)
    normalized_topic = topic.lower().strip()
    
    # Create a hash combining the topic and config
    hash_input = f"{normalized_topic}:{config_str}"
    key_hash = hashlib.sha256(hash_input.encode()).hexdigest()
    
    return f"{REPORT_PREFIX}{key_hash}"

@with_redis_client
def get_cached_report(client: redis.Redis, cache_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve a cached report.
    
    Args:
        client: Redis client
        cache_key: The cache key
        
    Returns:
        The cached report or None if not found
    """
    try:
        cached_data = client.get(cache_key)
        
        if cached_data:
            logger.info(f"Cache HIT for report: {cache_key}")
            return pickle.loads(cached_data)
            
        logger.info(f"Cache MISS for report: {cache_key}")
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving cached report: {str(e)}")
        return None

@with_redis_client
def set_cached_report(client: redis.Redis, cache_key: str, 
                     report_data: Dict[str, Any], ttl: int = CACHE_TTL) -> bool:
    """Store a report in cache.
    
    Args:
        client: Redis client
        cache_key: The cache key
        report_data: The report data to cache
        ttl: Time to live in seconds
        
    Returns:
        True if caching was successful
    """
    try:
        serialized_data = pickle.dumps(report_data)
        success = client.setex(cache_key, ttl, serialized_data)
        
        if success:
            logger.info(f"Successfully cached report with key: {cache_key}")
        
        return bool(success)
    except Exception as e:
        logger.error(f"Error caching report: {str(e)}")
        return False

@with_redis_client
def save_thread_state(client: redis.Redis, thread_id: str, 
                     state: Dict[str, Any], ttl: int = CACHE_TTL) -> bool:
    """Save thread state to Redis."""
    try:
        key = f"{THREAD_PREFIX}{thread_id}"
        serialized_data = pickle.dumps(state)
        success = client.setex(key, ttl, serialized_data)
        
        if success:
            logger.info(f"Successfully saved thread state: {thread_id}")
        
        return bool(success)
    except Exception as e:
        logger.error(f"Error saving thread state: {str(e)}")
        return False

@with_redis_client
def get_thread_state(client: redis.Redis, thread_id: str) -> Optional[Dict[str, Any]]:
    """Get thread state from Redis."""
    try:
        key = f"{THREAD_PREFIX}{thread_id}"
        data = client.get(key)
        
        if data:
            return pickle.loads(data)
            
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving thread state: {str(e)}")
        return None

@with_redis_client
def debug_redis_thread_state(client: redis.Redis, thread_id: str) -> Dict[str, Any]:
    """Debug Redis thread state."""
    try:
        key = f"{THREAD_PREFIX}{thread_id}"
        exists = client.exists(key)
        ttl = client.ttl(key) if exists else -1
        size = len(client.get(key) or b'') if exists else 0
        
        return {
            "exists": bool(exists),
            "ttl": ttl,
            "size": size,
            "key": key
        }
    except Exception as e:
        logger.error(f"Error debugging thread state: {str(e)}")
        return {
            "exists": False,
            "error": str(e),
            "key": f"{THREAD_PREFIX}{thread_id}"
        }

def format_markdown_for_frontend(markdown: str) -> str:
    """Format markdown content for frontend rendering."""
    if not markdown:
        return ""
        
    # Replace newlines with <br/> to preserve formatting
    formatted = markdown.replace('\n', '\n\n')
    
    return formatted

def extract_sections_from_plan(plan_text: str) -> List[Dict[str, Any]]:
    """Extract section information from plan text."""
    if not plan_text:
        return []
        
    sections = []
    current_section = None
    
    # Process plan text line by line
    lines = plan_text.strip().split('\n')
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Check for section headers
        if line.strip().startswith('Section:'):
            # Save previous section if it exists
            if current_section and 'name' in current_section:
                sections.append(current_section)
                
            # Start a new section
            current_section = {
                'name': line.replace('Section:', '').strip(),
                'description': '',
                'research': True  # Default to requiring research
            }
            
        # Check for description
        elif current_section and line.strip().startswith('Description:'):
            current_section['description'] = line.replace('Description:', '').strip()
            
        # Check for research needed flag
        elif current_section and 'research needed' in line.lower():
            if 'no' in line.lower() or 'false' in line.lower():
                current_section['research'] = False
            else:
                current_section['research'] = True
                
        # Research flag in alternative format
        elif current_section and line.strip().startswith('Research:'):
            if 'no' in line.lower() or 'false' in line.lower():
                current_section['research'] = False
            else:
                current_section['research'] = True
                
        # Research needed line
        elif current_section and line.strip().startswith('Research needed:'):
            if 'no' in line.lower() or 'false' in line.lower():
                current_section['research'] = False
            else:
                current_section['research'] = True
    
    # Add the last section if it exists
    if current_section and 'name' in current_section:
        sections.append(current_section)
        
    return sections

def extract_interrupt_message(event: Dict[str, Any]) -> Optional[str]:
    """Extract the interrupt message from an event."""
    # Case 1: Check for standard __interrupt__ format
    if '__interrupt__' in event:
        interrupt_data = event['__interrupt__']
        
        # Handle list format
        if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
            item = interrupt_data[0]
            # Check if it has a 'value' attribute or is a dictionary with 'value' key
            if hasattr(item, 'value'):
                return item.value
            elif isinstance(item, dict) and 'value' in item:
                return item['value']
            # Otherwise, try to convert the item to a string
            return str(item)
        
        # Handle direct object with 'value' attribute
        elif hasattr(interrupt_data, 'value'):
            return interrupt_data.value
        
        # Handle dict format
        elif isinstance(interrupt_data, dict) and 'value' in interrupt_data:
            return interrupt_data['value']
        
        # Handle string format directly
        elif isinstance(interrupt_data, str):
            return interrupt_data
            
        # Handle any other type by converting to string
        return str(interrupt_data)
    
    # Case 2: Check for interrupt_message format (used in some versions)
    elif 'interrupt_message' in event:
        return event['interrupt_message']
    
    # Case 3: Check for message directly in human_feedback result
    elif 'human_feedback' in event and isinstance(event['human_feedback'], dict):
        if 'message' in event['human_feedback']:
            return event['human_feedback']['message']
        elif 'interrupt_message' in event['human_feedback']:
            return event['human_feedback']['interrupt_message']
        elif 'interrupt' in event['human_feedback']:
            return event['human_feedback']['interrupt']
    
    # No interrupt message found
    return None

def verify_openai_api_key():
    """
    Verify that the OpenAI API key is valid by making a simple API call.
    
    Returns:
        tuple: (is_valid, error_message, details)
    """
    import os
    
    # Check if key exists in environment
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        return False, "OPENAI_API_KEY not found in environment", {"env_vars": [k for k in os.environ.keys() if k.startswith('OPENAI')]}
    
    # Clean the API key (remove quotes if present)
    api_key = api_key.strip().strip('"\'')
    
    # Check if key looks like a valid OpenAI key format
    if not api_key.startswith('sk-'):
        logger.error(f"OPENAI_API_KEY has invalid format: {api_key[:5]}... (does not start with 'sk-')")
        return False, "OpenAI API key has invalid format", {"key_prefix": api_key[:5]}
        
    # Import here to avoid dependency issues if OpenAI is not installed
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Make a simple request to verify the key
        try:
            # Use models.list as a simple test call that uses minimal tokens
            models = client.models.list()
            models_list = [model.id for model in models.data]
            
            # Check if models contains the ones we need
            required_models = ['gpt-4', 'gpt-3.5-turbo']  # Updated model names
            missing_models = [model for model in required_models if model not in models_list]
            
            if missing_models:
                logger.warning(f"OpenAI API key is valid, but missing access to required models: {missing_models}")
                return True, f"OpenAI API key is valid, but missing access to required models: {missing_models}", {"available_models": models_list}
            
            logger.info(f"OpenAI API key is valid with access to required models")
            return True, "OpenAI API key is valid", {"available_models": models_list}
        except Exception as e:
            error_message = str(e)
            logger.error(f"OpenAI API key validation failed: {error_message}")
            
            # Add specific diagnostic details
            details = {"error": error_message}
            
            if "authentication" in error_message.lower() or "invalid api key" in error_message.lower():
                details["error_type"] = "auth_error"
                return False, f"Invalid OpenAI API key: {error_message}", details
            elif "rate limit" in error_message.lower():
                details["error_type"] = "rate_limit"
                return False, f"OpenAI API rate limit exceeded: {error_message}", details
            elif "insufficient_quota" in error_message.lower() or "exceeded your quota" in error_message.lower():
                details["error_type"] = "quota_error"
                return False, f"OpenAI API quota exceeded: {error_message}", details
            
            return False, f"OpenAI API key validation failed: {error_message}", details
            
    except ImportError:
        logger.error("OpenAI Python package not installed")
        return False, "OpenAI Python package not installed", {"error_type": "import_error"}

def ensure_thread_has_required_props(thread, topic, thread_id=None):
    """Ensure a thread has all required properties."""
    if not thread:
        thread = {}
        
    if 'configurable' not in thread:
        thread['configurable'] = {}
        
    # Use provided thread_id or generate a unique one
    if thread_id:
        thread['configurable']['thread_id'] = thread_id
        
    # Always ensure topic is set in configurable
    if topic:
        thread['configurable']['topic'] = topic
        
        # Also ensure topic is available at the root level (for LangGraph compatibility)
        thread['topic'] = topic
        
    # Set default values for required fields if missing
    defaults = {
        'search_api': os.getenv('DEFAULT_SEARCH_API', 'tavily'),
        'planner_provider': os.getenv('DEFAULT_PLANNER_PROVIDER', 'openai'),
        'planner_model': os.getenv('DEFAULT_PLANNER_MODEL', 'gpt-4'),
        'writer_provider': os.getenv('DEFAULT_WRITER_PROVIDER', 'openai'),
        'writer_model': os.getenv('DEFAULT_WRITER_MODEL', 'gpt-4'),
        'max_search_depth': int(os.getenv('MAX_SEARCH_DEPTH', 2)),
        'number_of_queries': int(os.getenv('NUMBER_OF_QUERIES', 2)),
        'openai_api_key': os.getenv('OPENAI_API_KEY')  # Add OpenAI API key to defaults
    }
    
    # Add default report structure
    from .graph_manager import ResearchManager
    defaults['report_structure'] = ResearchManager.REPORT_STRUCTURE
    
    # Apply defaults for missing values
    for key, value in defaults.items():
        if key not in thread['configurable']:
            thread['configurable'][key] = value
            
    # Special case for topic - ensure it's always present
    if 'topic' not in thread['configurable'] and topic:
        thread['configurable']['topic'] = topic
    
    # Ensure numeric values are integers to prevent type comparison issues
    if 'max_search_depth' in thread['configurable']:
        thread['configurable']['max_search_depth'] = int(thread['configurable']['max_search_depth'])
    if 'number_of_queries' in thread['configurable']:
        thread['configurable']['number_of_queries'] = int(thread['configurable']['number_of_queries'])
    
    # Log the state of thread
    logger.info(f"Thread has topic at root level: {thread.get('topic', None) is not None}")
    logger.info(f"Thread has topic in configurable: {thread.get('configurable', {}).get('topic', None) is not None}")
    
    return thread

def debug_langgraph_event(event: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Debug information about a LangGraph event.
    
    Provides detailed debug info about a LangGraph event.
    
    Args:
        event: The event to debug
        prefix: Optional prefix for logged messages
        
    Returns:
        Debug information
    """
    debug_info = {}
    
    if not isinstance(event, dict):
        logger.warning(f"{prefix}Event is not a dictionary, type: {type(event).__name__}")
        return {"error": "Not a dictionary", "type": type(event).__name__}
    
    # Get keys
    keys = list(event.keys())
    debug_info["keys"] = keys
    
    # Check for sections
    if "sections" in keys:
        sections = event["sections"]
        
        # Check if sections is a list
        if isinstance(sections, list):
            debug_info["sections_count"] = len(sections)
            
            # Check if there are sections
            if len(sections) > 0:
                first_section = sections[0]
                
                # Get section type
                section_type = type(first_section).__name__
                debug_info["section_type"] = section_type
                
                # Get section attributes
                if hasattr(first_section, "__dict__"):
                    debug_info["section_attributes"] = list(first_section.__dict__.keys())
                elif hasattr(first_section, "__slots__"):
                    debug_info["section_attributes"] = first_section.__slots__
                elif isinstance(first_section, dict):
                    debug_info["section_attributes"] = list(first_section.keys())
                else:
                    debug_info["section_attributes"] = "unknown"
                
                # Check for name and content
                if hasattr(first_section, "name"):
                    debug_info["has_name"] = True
                    debug_info["first_section_name"] = first_section.name
                
                if hasattr(first_section, "content"):
                    debug_info["has_content"] = True
                    content_length = len(first_section.content) if first_section.content else 0
                    debug_info["content_length"] = content_length
                
                if hasattr(first_section, "research"):
                    debug_info["has_research"] = True
                    debug_info["is_research"] = first_section.research
        else:
            debug_info["sections_error"] = f"Not a list, type: {type(sections).__name__}"
    
    # Check for interrupt message
    if "interrupt_message" in keys:
        msg = event["interrupt_message"]
        debug_info["interrupt_length"] = len(msg) if isinstance(msg, str) else "Not a string"
    
    # Check for errors
    if "error" in keys:
        debug_info["has_error"] = True
        debug_info["error"] = event["error"]
    
    return debug_info 

def setup_sse_response(response):
    """
    Set up a response with the necessary headers for SSE.
    
    Args:
        response: The response object to modify
        
    Returns:
        The modified response
    """
    # Add required SSE headers
    response['Cache-Control'] = 'no-cache, no-transform'
    response['X-Accel-Buffering'] = 'no'  # Important for Nginx
    response['Connection'] = 'keep-alive'
    
    # Ensure the Content-Type is set correctly
    response['Content-Type'] = 'text/event-stream'
    
    # Add specific retry directive (tells browser to reconnect after 3 seconds if connection is lost)
    # This is critical for SSE reconnection
    response['retry'] = '3000'
    
    # Disable chunked transfer encoding which can interfere with SSE
    response['Transfer-Encoding'] = 'identity'
    
    logger.debug(f"Setup SSE response headers: {dict(response.items())}")
    
    return response

def generate_sse_event(event_data):
    """
    Generate an SSE event string from event data.
    
    Args:
        event_data: Dictionary with event data
        
    Returns:
        Formatted SSE event string
    """
    try:
        return f"data: {json.dumps(event_data)}\n\n"
    except Exception as e:
        logger.error(f"Error generating SSE event: {str(e)}")
        return f"data: {json.dumps({'error': f'Error formatting event: {str(e)}'})}\n\n"

def generate_sse_keepalive():
    """
    Generate an SSE keep-alive comment.
    
    Returns:
        SSE keep-alive comment string
    """
    return ":\n\n"

def handle_sse_stream(async_gen, process_event_fn=None, metadata=None, keepalive_interval=10):
    """
    Handle an SSE stream with proper keep-alive and error handling.
    
    Args:
        async_gen: Async generator producing events
        process_event_fn: Optional function to process each event before sending
        metadata: Optional metadata to add to each event
        keepalive_interval: Seconds between keep-alive messages (default: 10 seconds)
        
    Returns:
        Async generator yielding formatted SSE events
    """
    async def stream_handler():
        # Start with multiple keep-alive comments to establish connection
        yield generate_sse_keepalive()
        yield generate_sse_keepalive()
        
        # Send connection established event
        yield generate_sse_event({
            'type': 'connection_established', 
            'message': 'SSE connection established',
            'metadata': metadata,
            'timestamp': time.time()
        })
        
        # Send another keep-alive
        yield generate_sse_keepalive()
        
        last_event_time = time.time()
        last_keepalive_time = time.time()
        error_count = 0
        max_errors = 3
        empty_event_count = 0
        max_empty_events = 5
        
        try:
            # Process each event from the generator
            async for event in async_gen:
                try:
                    # Always update last_event_time even for empty events
                    current_time = time.time()
                    
                    # Send a keepalive if needed
                    if current_time - last_keepalive_time > keepalive_interval:
                        logger.debug(f"Sending keepalive after {current_time - last_keepalive_time:.2f}s")
                        yield generate_sse_keepalive()
                        last_keepalive_time = current_time
                        
                    # Process the event if it's not empty
                    if event:
                        last_event_time = current_time
                        empty_event_count = 0  # Reset empty event counter
                        
                        # Process the event if a processing function is provided
                        if process_event_fn:
                            try:
                                processed_event = process_event_fn(event)
                                if not processed_event:
                                    logger.warning("Event processor returned empty result")
                                    continue
                                event = processed_event
                            except Exception as e:
                                logger.error(f"Error processing event: {str(e)}", exc_info=True)
                                yield generate_sse_event({'type': 'error', 'message': f'Error processing event: {str(e)}', 'timestamp': time.time()})
                                
                                # Track error count and continue if not too many errors
                                error_count += 1
                                if error_count >= max_errors:
                                    logger.error(f"Too many errors ({error_count}), stopping stream")
                                    break
                                continue
                        
                        # Add metadata if provided
                        if metadata and isinstance(event, dict):
                            if '__meta' not in event:
                                event['__meta'] = {}
                            event['__meta'].update(metadata)
                            
                            # Add timestamp if not present
                            if 'timestamp' not in event:
                                event['timestamp'] = time.time()
                        
                        # Send the event
                        yield generate_sse_event(event)
                        
                        # Send a keepalive after each substantial event
                        yield generate_sse_keepalive()
                        last_keepalive_time = time.time()

                        # Reset error count on successful event
                        error_count = 0
                    else:
                        # Handle empty events
                        empty_event_count += 1
                        if empty_event_count >= max_empty_events:
                            logger.warning(f"Received {empty_event_count} empty events, breaking stream")
                            # Send a keepalive and a warning
                            yield generate_sse_keepalive()
                            yield generate_sse_event({
                                'type': 'warning',
                                'message': 'Stream producing empty events',
                                'empty_count': empty_event_count,
                                'timestamp': time.time()
                            })
                            break
                        
                        # Send a keepalive on empty events to maintain connection
                        if current_time - last_keepalive_time > keepalive_interval / 2:  # More frequent keepalives for empty events
                            yield generate_sse_keepalive()
                            last_keepalive_time = current_time
                            
                except Exception as event_error:
                    logger.error(f"Error handling event in SSE stream: {str(event_error)}", exc_info=True)
                    yield generate_sse_event({'type': 'error', 'message': f'Error handling event: {str(event_error)}', 'timestamp': time.time()})
                    yield generate_sse_keepalive()  # Add keepalive even after error
                    last_keepalive_time = time.time()
                    
                    # Track error count and continue if not too many errors
                    error_count += 1
                    if error_count >= max_errors:
                        logger.error(f"Too many errors ({error_count}), stopping stream")
                        break
                
                # Force a keepalive if too much time has passed with no events
                if time.time() - last_event_time > keepalive_interval * 2:
                    logger.debug(f"No events for {time.time() - last_event_time:.2f}s, sending health check")
                    yield generate_sse_event({
                        'type': 'health_check',
                        'message': 'Connection active but no events received recently',
                        'seconds_since_last_event': round(time.time() - last_event_time, 1),
                        'timestamp': time.time()
                    })
                    yield generate_sse_keepalive()
                    last_event_time = time.time()  # Reset timer
                    last_keepalive_time = time.time()
                    
        except Exception as e:
            logger.error(f"Error in SSE stream: {str(e)}", exc_info=True)
            yield generate_sse_event({'type': 'error', 'message': f'Error in SSE stream: {str(e)}', 'timestamp': time.time()})
            yield generate_sse_keepalive()
        
        # Send stream complete event
        yield generate_sse_event({'type': 'stream_complete', 'message': 'Stream completed', 'timestamp': time.time()})
        
        # Final keep-alive
        yield generate_sse_keepalive()
    
    return stream_handler() 