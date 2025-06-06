import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import instructor
from django.http import StreamingHttpResponse
import redis
import hashlib
import pickle
import time
from functools import lru_cache, wraps
from django.core.cache import cache
from django.conf import settings

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # Default 1 hour cache
CACHE_VERSION = "v1"  # Increment this when making breaking changes to the response format

# Global connection pool
REDIS_POOL = redis.connection.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    max_connections=50,
    decode_responses=False,  # Keep as bytes for pickle serialization
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True,
    health_check_interval=30
)

# Clear cache on startup
try:
    logger.info("🧹 Clearing cache on startup...")
    clear_result = clear_cache()
    logger.info(f"Cache clear result: {clear_result}")
except Exception as e:
    logger.error(f"Failed to clear cache on startup: {str(e)}")

def get_redis_client() -> redis.Redis:
    """Get a Redis client from the connection pool."""
    try:
        client = redis.Redis(connection_pool=REDIS_POOL)
        client.ping()  # Test connection
        return client
    except redis.ConnectionError as e:
        logger.error(f"Redis connection error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error getting Redis client: {str(e)}")
        raise

def with_redis_client(func):
    """Decorator to handle Redis client acquisition and error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            client = get_redis_client()
            return func(client, *args, **kwargs)
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error in {func.__name__}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

def calculate_ttl(cache_key: str) -> int:
    """Calculate TTL based on usage patterns."""
    try:
        redis_client = get_redis_client()
        access_count = int(redis_client.get(f"{cache_key}:access_count") or 0)
        
        # Base TTL is CACHE_TTL (1 hour by default)
        base_ttl = CACHE_TTL
        
        # Frequently accessed items (>10 accesses) get longer TTL (up to 24 hours)
        if access_count > 10:
            return min(base_ttl * 24, base_ttl * (access_count // 5))
        
        return base_ttl
    except Exception as e:
        logger.error(f"Error calculating TTL: {str(e)}")
        return CACHE_TTL

@with_redis_client
def get_cached_response(client: redis.Redis, cache_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve a cached response."""
    try:
        cached_data = client.get(cache_key)
        
        if cached_data:
            try:
                # Increment access counter
                client.incr(f"{cache_key}:access_count")
                
                # Update TTL based on new access count
                new_ttl = calculate_ttl(cache_key)
                client.expire(cache_key, new_ttl)
                client.expire(f"{cache_key}:access_count", new_ttl)
                
                # Update metadata
                metadata = {
                    'last_accessed': time.time(),
                    'ttl': new_ttl,
                    'access_count': int(client.get(f"{cache_key}:access_count") or 1)
                }
                client.setex(f"{cache_key}:metadata", new_ttl, json.dumps(metadata))
                
                # Increment hit counter
                client.incr('cache_hits')
                
                logger.info(f"✅ Cache HIT for key: {cache_key}")
                return pickle.loads(cached_data)
            except Exception as e:
                logger.warning(f"Non-critical error updating cache metadata: {str(e)}")
                return pickle.loads(cached_data)
        
        # Increment miss counter
        client.incr('cache_misses')
        logger.info(f"❌ Cache MISS for key: {cache_key}")
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving from cache: {str(e)}")
        return None

@with_redis_client
def set_cached_response(client: redis.Redis, cache_key: str, response_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
    """Store a response in cache."""
    try:
        # Calculate TTL if not provided
        if ttl is None:
            try:
                ttl = calculate_ttl(cache_key)
            except:
                ttl = CACHE_TTL
        
        # Core caching operation
        serialized_data = pickle.dumps(response_data)
        success = client.setex(cache_key, ttl, serialized_data)
        
        if success:
            try:
                # Increment access counter
                client.incr(f"{cache_key}:access_count")
                client.expire(f"{cache_key}:access_count", ttl)
                
                # Store metadata
                metadata = {
                    'created_at': time.time(),
                    'ttl': ttl,
                    'access_count': 1
                }
                client.setex(f"{cache_key}:metadata", ttl, json.dumps(metadata))
                
                # Increment total entries counter
                client.incr('total_cached_entries')
                
                logger.info(f"💾 Successfully cached response with key: {cache_key}")
            except Exception as e:
                logger.warning(f"Non-critical error setting cache metadata: {str(e)}")
        
        return bool(success)
    except Exception as e:
        logger.error(f"Error setting cache: {str(e)}")
        return False

# Prompt management configuration
PROMPT_MANAGER ="langfuse"# os.getenv("PROMPT_MANAGER", "default")

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant that helps analyze topics from multiple perspectives. For each query, provide a balanced analysis with different viewpoints, core arguments, and supporting points. Focus on factual, well-reasoned arguments rather than emotional appeals.

For each stance:
1. Start with "stance:" followed by a clear position
2. Then provide "core argument:" with the main reasoning
3. Finally, list "supporting arguments:" with specific points

Example format:
stance: [Position]
core argument: [Main reasoning]
supporting arguments: [Specific point 1]
supporting arguments: [Specific point 2]
supporting arguments: [Specific point 3]

Repeat this format for each different perspective on the topic."""

# Log environment variables (excluding sensitive data)
logger.debug("Environment variables loaded:")
logger.debug(f"PROMPT_MANAGER: {PROMPT_MANAGER}")

# Initialize Langfuse only if selected
langfuse = None
if PROMPT_MANAGER == "langfuse":
    try:
        from langfuse import Langfuse
        # Get Langfuse configuration
        langfuse_host = os.getenv("LANGFUSE_HOST", "https://vector.allsides.com")
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        
        if not all([langfuse_host, langfuse_public_key, langfuse_secret_key]):
            logger.warning("Missing Langfuse credentials, falling back to default prompt")
            PROMPT_MANAGER = "default"
        else:
            # Initialize Langfuse client with proper configuration
            langfuse = Langfuse(
                public_key=langfuse_public_key,
                secret_key=langfuse_secret_key,
                host=langfuse_host,
                debug=False,  # Set to True for debugging
                enabled=True,
                flush_at=10,  # Flush after 10 events
                flush_interval=1,  # Flush every 1 second
                max_retries=3,
                timeout=30  # 30 second timeout
            )
            
            # Verify connection
            if langfuse.auth_check():
                logger.info("Langfuse initialized and authenticated successfully")
                
                # Test prompt retrieval
                try:
                    langfuse_prompt = langfuse.get_prompt("AllStances_v1")
                    if langfuse_prompt:
                        logger.info(f"Successfully retrieved prompt: AllStances_v1")
                    else:
                        logger.warning("Prompt 'AllStances_v1' not found in Langfuse")
                except Exception as prompt_error:
                    logger.warning(f"Error retrieving prompt: {str(prompt_error)}")
            else:
                logger.error("Langfuse authentication failed")
                langfuse = None
                PROMPT_MANAGER = "default"

    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {str(e)}")
        langfuse = None
        PROMPT_MANAGER = "default"
        logger.info("Falling back to default prompt management")

# Get API keys with proper error handling
def get_api_keys():
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
    return openai_key.strip().strip('"\'')

# Initialize OpenAI client
try:
    openai_key = get_api_keys()
    client = OpenAI(
        api_key=openai_key,
        timeout=300.0  # Set 5 minute timeout for OpenAI requests
    )
    # Add instructor for response parsing
    client = instructor.patch(client)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

# Get system configuration based on selected prompt manager
def get_config():
    try:
        if PROMPT_MANAGER == "langfuse" and langfuse:
            try:
                # Get the prompt from Langfuse
                langfuse_prompt = langfuse.get_prompt("AllStances_v1")
                if not langfuse_prompt:
                    logger.warning("Prompt not found in Langfuse, using default")
                    return {'system_prompt': DEFAULT_SYSTEM_PROMPT}
                
                return {
                    'system_prompt': langfuse_prompt.prompt,
                }
            except Exception as prompt_error:
                logger.warning(f"Error fetching Langfuse prompt: {str(prompt_error)}")
                return {'system_prompt': DEFAULT_SYSTEM_PROMPT}
        else:
            logger.info("Using default prompt management")
            return {'system_prompt': DEFAULT_SYSTEM_PROMPT}
    except Exception as e:
        logger.warning(f"Error in get_config: {str(e)}, using default")
        return {'system_prompt': DEFAULT_SYSTEM_PROMPT}

def normalize_query(query: str) -> str:
    """Normalize query text for consistent caching."""
    if not query:
        return ""
    # Convert to lowercase and strip whitespace
    normalized = query.lower().strip()
    # Remove extra whitespace between words and normalize internal spaces
    normalized = ' '.join(normalized.split())
    # Remove punctuation that doesn't affect meaning
    normalized = normalized.replace("'s", "s")  # normalize possessives
    normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
    return normalized

def verify_cache_key_consistency(topic: str, diversity: float, num_stances: int, system_content: str) -> Dict[str, Any]:
    """Verify that cache keys are being generated consistently."""
    try:
        # Normalize topic and system content
        normalized_topic = normalize_query(topic)
        normalized_system_content = normalize_query(system_content)
        
        # Generate the key twice with the same parameters
        params1 = {
            "topic": normalized_topic,
            "diversity": diversity,
            "num_stances": num_stances,
            "system_content": normalized_system_content
        }
        
        params2 = params1.copy()
        
        key1 = generate_cache_key(**params1)
        key2 = generate_cache_key(**params2)
        
        are_equal = key1 == key2
        
        result = {
            "are_equal": are_equal,
            "key1": key1,
            "key2": key2,
            "params": params1,
            "normalized_topic": normalized_topic,
            "original_topic": topic,
            "normalized_system_content": normalized_system_content
        }
        
        if not are_equal:
            logger.error(f"Cache key inconsistency detected: {json.dumps(result, indent=2)}")
        else:
            logger.debug(f"Cache key consistency verified: {key1}")
        
        return result
    except Exception as e:
        logger.error(f"Error verifying cache key consistency: {str(e)}")
        return {
            "error": str(e),
            "are_equal": False
        }

def generate_cache_key(topic: str, diversity: float, num_stances: int, system_content: str) -> str:
    """Generate a deterministic cache key based on essential request parameters."""
    try:
        # Validate inputs
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Topic must be a non-empty string")
        if not isinstance(diversity, (int, float)) or not 0 <= diversity <= 1:
            raise ValueError("Diversity must be a float between 0 and 1")
        if not isinstance(num_stances, int) or not 2 <= num_stances <= 7:
            raise ValueError("Number of stances must be between 2 and 7")
        if not isinstance(system_content, str) or not system_content.strip():
            raise ValueError("System content must be a non-empty string")
        
        # Normalize inputs - ensure consistent formatting
        normalized_topic = normalize_query(topic)
        # Extract core system prompt without dynamic parts
        base_system_content = system_content.split("\n\nPlease attempt to provide")[0]
        normalized_system_content = normalize_query(base_system_content)
        rounded_diversity = round(float(diversity), 3)
        
        # Create a string containing all parameters in a consistent order
        param_string = f"{CACHE_VERSION}|{normalized_topic}|{rounded_diversity:.3f}|{num_stances}|{normalized_system_content}"
        logger.debug(f"Cache key param string: {param_string}")
        
        # Create SHA-256 hash of the parameter string
        hash_object = hashlib.sha256(param_string.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        
        # Return a prefixed cache key
        cache_key = f"allstances:{CACHE_VERSION}:response:{hash_hex}"
        logger.debug(f"Generated cache key: {cache_key}")
        return cache_key
    except Exception as e:
        logger.error(f"Error generating cache key: {str(e)}")
        raise
from langfuse.decorators import langfuse_context, observe


# GPT-4 pricing (as of 2024)
GPT4_INPUT_COST_PER_1K_TOKENS = 0.03
GPT4_OUTPUT_COST_PER_1K_TOKENS = 0.06

def calculate_openai_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4") -> float:
    """Calculate the cost of an OpenAI API call based on token usage."""
    try:
        if model.startswith("gpt-4"):
            input_cost = (input_tokens / 1000) * GPT4_INPUT_COST_PER_1K_TOKENS
            output_cost = (output_tokens / 1000) * GPT4_OUTPUT_COST_PER_1K_TOKENS
            return round(input_cost + output_cost, 6)
        else:
            # Default to GPT-4 pricing for unknown models
            input_cost = (input_tokens / 1000) * GPT4_INPUT_COST_PER_1K_TOKENS
            output_cost = (output_tokens / 1000) * GPT4_OUTPUT_COST_PER_1K_TOKENS
            return round(input_cost + output_cost, 6)
    except Exception as e:
        logger.error(f"Error calculating cost: {str(e)}")
        return 0.0


def complete(topic: str, diversity: float, num_stances: int = 3, user_id: Optional[str] = None, session_id: Optional[str] = None, request_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get completion from OpenAI with structured output and caching."""
    trace = None
    generation = None
    
    try:
        # Normalize the topic first
        normalized_topic = normalize_query(topic)
        logger.info(f"Normalized topic: '{normalized_topic}' (original: '{topic}')")
        
        # Prepare trace metadata
        trace_metadata = {
            "topic": topic,
            "normalized_topic": normalized_topic,
            "diversity": diversity,
            "num_stances": num_stances,
            "cache_version": CACHE_VERSION,
            "prompt_manager": PROMPT_MANAGER
        }
        
        # Add request metadata if provided
        if request_metadata:
            trace_metadata.update(request_metadata)
        
        # Start Langfuse Trace with proper context
        if langfuse:
            trace = langfuse.trace(
                name="allstances_completion",
                user_id=user_id,
                session_id=session_id,
                metadata=trace_metadata,
                tags=["allstances", "argument_generation", f"stances_{num_stances}"]
            )
            logger.info(f"Started Langfuse trace: {trace.id if trace else 'None'}")
        else:
            logger.warning("Langfuse not available, skipping trace creation")

    
    # Get system configuration first to ensure consistent cache key generation
    config = get_config()
    system_message = config['system_prompt']
    if isinstance(system_message, list):
        system_message = system_message[0]
    
    # Get base system content
    if isinstance(system_message, dict):
        system_content = system_message.get('content', DEFAULT_SYSTEM_PROMPT)
    else:
        system_content = str(system_message)
    
    # Add stance count to system prompt
    system_content = system_content + f"\n\nPlease attempt to provide at most or equal to {num_stances} different perspectives on this topic."
    
    # Verify cache key consistency with the complete system prompt
    consistency_check = verify_cache_key_consistency(normalized_topic, diversity, num_stances, system_content)
    if not consistency_check.get("are_equal", False):
        logger.error("Cache key consistency check failed!")
    
    try:
        # Generate cache key using normalized parameters and complete system prompt
        cache_key = generate_cache_key(normalized_topic, diversity, num_stances, system_content)
        logger.info(f"🔑 Generated cache key: {cache_key}")

        # Try to get cached response
        cached_result = get_cached_response(cache_key)
        if cached_result:
            logger.info(f"✅ Cache HIT for topic: '{normalized_topic}'")
            
            # Create a span for cache hit if trace exists
            if trace:
                cache_span = trace.span(
                    name="cache_hit",
                    metadata={
                        "cache_key": cache_key,
                        "cache_hit": True,
                        "response_source": "redis_cache"
                    }
                )
                cache_span.end(output=cached_result)
                
                # Update trace metadata
                trace.update(
                    metadata={
                        **trace_metadata,
                        "cache_hit": True,
                        "response_source": "cache"
                    }
                )
                
                # Flush events for cache hits too
                try:
                    langfuse.flush()
                except Exception as flush_error:
                    logger.warning(f"Failed to flush Langfuse events on cache hit: {str(flush_error)}")
            
            return cached_result

        logger.info(f"❌ Cache MISS for topic: '{normalized_topic}'")
        
        # Prepare final system message format for API call
        if isinstance(system_message, dict):
            system_message['content'] = system_content
        else:
            system_message = {"role": "system", "content": system_content}

        # Prepare messages for API call
        messages = [
            system_message,
            {"role": "user", "content": normalized_topic}
        ]
        
        # Create generation in trace if available
        if trace:
            generation = trace.generation(
                name="allstances_output",
                model="gpt-4",
                model_parameters={
                    "temperature": diversity,
                    "max_tokens": None,
                    "top_p": 1.0,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "stances": num_stances
                },
                input=messages,
                metadata={
                    "cache_key": cache_key,
                    "instructor_enabled": True,
                    "response_model": "ArgumentResponse"
                }
            )
            logger.info(f"Created generation: {generation.id if generation else 'None'}")

        # Call OpenAI API with instructor to get structured output
        try:
            start_time = time.time()
            
            response = client.chat.completions.create(
                model='gpt-4',
                messages=messages,
                temperature=diversity,
                stream=False,
                response_model=ArgumentResponse
            )
            
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            # Convert response to dict
            result = response.to_dict()
            
            # Extract usage information if available
            usage_data = None
            cost = 0.0
            
            if hasattr(response, '_raw_response') and hasattr(response._raw_response, 'usage'):
                usage = response._raw_response.usage
                usage_data = {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
                cost = calculate_openai_cost(usage.prompt_tokens, usage.completion_tokens, "gpt-4")
                logger.info(f"Usage: {usage_data}, Cost: ${cost}")
            else:
                logger.warning("Usage information not available from response")
            
            # End generation with complete information
            if generation:
                generation_end_data = {
                    "output": result,
                    "end_time": end_time,
                    "metadata": {
                        "latency_ms": latency_ms,
                        "cache_miss": True,
                        "cost_usd": cost
                    }
                }
                
                if usage_data:
                    generation_end_data["usage"] = usage_data
                
                generation.end(**generation_end_data)
                logger.info(f"Ended generation with usage data: {usage_data}")

            # Cache the result
            cache_success = set_cached_response(cache_key, result)
            if cache_success:
                logger.info(f"💾 Successfully cached response for topic: '{normalized_topic}' with key: {cache_key}")
            else:
                logger.warning(f"⚠️ Failed to cache response for topic: '{normalized_topic}' with key: {cache_key}")
            
            # Flush Langfuse events
            if langfuse:
                try:
                    langfuse.flush()
                    logger.debug("Flushed Langfuse events")
                except Exception as flush_error:
                    logger.warning(f"Failed to flush Langfuse events: {str(flush_error)}")
            
            return result

        except Exception as api_error:
            logger.error(f"API Error: {str(api_error)}")
            
            # Track error in generation if available
            if generation:
                try:
                    generation.end(
                        output=None,
                        metadata={
                            "error": str(api_error),
                            "error_type": type(api_error).__name__,
                            "cache_miss": True,
                            "latency_ms": int((time.time() - start_time) * 1000) if 'start_time' in locals() else None
                        }
                    )
                    logger.info("Recorded API error in generation")
                except Exception as gen_error:
                    logger.warning(f"Failed to record error in generation: {str(gen_error)}")
            
            # Flush Langfuse events even on error
            if langfuse:
                try:
                    langfuse.flush()
                except Exception as flush_error:
                    logger.warning(f"Failed to flush Langfuse events on error: {str(flush_error)}")
            
            error_response = ArgumentResponse(
                arguments=[
                    Stance(
                        stance="Error Processing Request",
                        core_argument="An error occurred while processing your request",
                        supporting_arguments=["Please try again later"]
                    ),
                    Stance(
                        stance="Technical Details",
                        core_argument="API Error Information",
                        supporting_arguments=[str(api_error)]
                    )
                ],
                model="gpt-4"
            )
            return error_response.to_dict()

    except Exception as e:
        logger.error(f"Error in complete: {str(e)}")
        
        # Track error in trace if available
        if trace:
            try:
                trace.update(
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "function": "complete"
                    }
                )
                logger.info("Recorded system error in trace")
            except Exception as trace_error:
                logger.warning(f"Failed to record error in trace: {str(trace_error)}")
        
        # Flush Langfuse events even on error
        if langfuse:
            try:
                langfuse.flush()
            except Exception as flush_error:
                logger.warning(f"Failed to flush Langfuse events on system error: {str(flush_error)}")
        
        error_response = ArgumentResponse(
            arguments=[
                Stance(
                    stance="System Error",
                    core_argument="A system error occurred",
                    supporting_arguments=[str(e)]
                ),
                Stance(
                    stance="Technical Details",
                    core_argument="Additional error information",
                    supporting_arguments=["Please try again later", "If the issue persists, contact support"]
                )
            ],
            model="gpt-4"
        )
        return error_response.to_dict()

def process_streaming_response(response):
    """Process streaming response and return structured output."""
    current_stance = None
    current_core_argument = None
    current_supporting_arguments = []
    arguments = []
    
    for chunk in response:
        if not chunk.choices or not chunk.choices[0].delta or not chunk.choices[0].delta.content:
            continue
        
        content = chunk.choices[0].delta.content
        
        # Parse the streaming content and update the response structure
        if "stance:" in content.lower():
            if current_stance:
                arguments.append({
                    "stance": current_stance,
                    "core_argument": current_core_argument or "",
                    "supporting_arguments": current_supporting_arguments
                })
            current_stance = content.split("stance:", 1)[1].strip()
            current_core_argument = None
            current_supporting_arguments = []
        elif "core argument:" in content.lower():
            current_core_argument = content.split("core argument:", 1)[1].strip()
        elif "supporting arguments:" in content.lower():
            arg = content.split("supporting arguments:", 1)[1].strip()
            if arg:
                current_supporting_arguments.append(arg)
        elif current_supporting_arguments:
            current_supporting_arguments[-1] += content

    # Add the last stance if exists
    if current_stance:
        arguments.append({
            "stance": current_stance,
            "core_argument": current_core_argument or "",
            "supporting_arguments": current_supporting_arguments
        })

    return {"arguments": arguments}



        
class Stance(BaseModel):
    stance: str = Field(description="The stance or perspective on the topic")
    core_argument: str = Field(description="The main argument supporting this stance", alias="core_argument")
    supporting_arguments: List[str] = Field(
        description="Supporting points for the core argument",
        min_items=1,
        max_items=3,
        alias="supporting_arguments"
    )

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stance": self.stance,
            "core_argument": self.core_argument,
            "supporting_arguments": self.supporting_arguments
        }

class ArgumentResponse(BaseModel):
    arguments: List[Stance] = Field(
        description="List of stances and their arguments",
        min_items=2,
        max_items=7  # Updated to allow up to 7 stances
    )
    model: str = Field(default="gpt-4", description="The model used for generating the response")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arguments": [arg.to_dict() for arg in self.arguments],
            "model": self.model
        }

# Add cache management endpoints
async def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the cache."""
    try:
        redis_client = get_redis_client()
        keys = redis_client.keys("allstances:response:*")
        stats = {
            "total_cached_entries": len(keys),
            "cache_size_bytes": sum(asyncio.gather(*[redis_client.memory_usage(key) for key in keys])) if keys else 0,
            "cache_keys": keys
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        return {"error": str(e)}

# Performance optimizations and utilities

async def warm_cache(topics: List[str], diversity: float, num_stances: int) -> Dict[str, Any]:
    """Pre-warm cache with a batch of topics."""
    results = {}
    async with asyncio.TaskGroup() as group:
        for topic in topics:
            task = group.create_task(complete(topic, diversity, num_stances))
            results[topic] = task
    
    return {topic: task.result() for topic, task in results.items()}

async def get_cache_health() -> Dict[str, Any]:
    """Get detailed cache health metrics."""
    try:
        redis_client = get_redis_client()
        info = redis_client.info()
        
        # Get cache statistics
        keys = redis_client.keys("allstances:response:*")
        total_size = sum(asyncio.gather(*[redis_client.memory_usage(key) for key in keys])) if keys else 0
        
        # Calculate hit rate if available
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        hit_rate = (hits / (hits + misses)) * 100 if (hits + misses) > 0 else 0
        
        return {
            "status": "healthy" if redis_client.ping() else "unhealthy",
            "total_entries": len(keys),
            "total_size_bytes": total_size,
            "memory_usage_percent": info.get('used_memory_peak_perc', 0),
            "hit_rate_percent": hit_rate,
            "uptime_seconds": info.get('uptime_in_seconds', 0),
            "connected_clients": info.get('connected_clients', 0)
        }
    except Exception as e:
        logger.error(f"Error getting cache health: {str(e)}")
        return {"status": "error", "error": str(e)}

async def optimize_cache() -> Dict[str, Any]:
    """Optimize cache by removing least used entries when approaching memory limits."""
    try:
        redis_client = get_redis_client()
        info = redis_client.info()
        
        # If memory usage is above 80%, remove least recently used entries
        if float(info.get('used_memory_peak_perc', 0)) > 80:
            keys = redis_client.keys("allstances:response:*")
            # Get access times for all keys
            access_times = []
            for key in keys:
                # Get last access time using OBJECT IDLETIME
                idle_time = redis_client.execute_command('OBJECT', 'IDLETIME', key)
                access_times.append((key, idle_time))
            
            # Sort by idle time (descending) and remove oldest 20%
            access_times.sort(key=lambda x: x[1], reverse=True)
            keys_to_remove = access_times[:int(len(access_times) * 0.2)]
            
            if keys_to_remove:
                redis_client.delete(*[key for key, _ in keys_to_remove])
                
            return {
                "status": "optimized",
                "entries_removed": len(keys_to_remove),
                "memory_usage_before": info.get('used_memory_peak_perc', 0)
            }
        
        return {"status": "optimization_not_needed"}
    except Exception as e:
        logger.error(f"Error optimizing cache: {str(e)}")
        return {"status": "error", "error": str(e)}

# Periodic cache maintenance task
async def verify_cache_consistency() -> Dict[str, Any]:
    """Verify cache consistency and repair if needed."""
    try:
        redis_client = get_redis_client()
        issues = []
        fixes = []
        
        # Get all cache keys
        cursor = 0
        all_keys = set()
        while True:
            cursor, keys = redis_client.scan(cursor, match="allstances:*")
            all_keys.update(keys)
            if cursor == 0:
                break
        
        # Check each key
        for key in all_keys:
            try:
                # Check if key is a main data key
                if key.endswith(":metadata") or key.endswith(":access_count"):
                    continue
                
                # Verify data exists and is valid
                data = redis_client.get(key)
                if not data:
                    issues.append(f"Missing data for key: {key}")
                    continue
                
                try:
                    pickle.loads(data)
                except:
                    issues.append(f"Corrupted data for key: {key}")
                    redis_client.delete(key)
                    fixes.append(f"Removed corrupted data for key: {key}")
                    continue
                
                # Check associated metadata and counters exist
                metadata_key = f"{key}:metadata"
                counter_key = f"{key}:access_count"
                
                if not redis_client.exists(metadata_key):
                    issues.append(f"Missing metadata for key: {key}")
                    # Create metadata
                    metadata = {
                        'created_at': time.time(),
                        'ttl': CACHE_TTL,
                        'access_count': 1
                    }
                    redis_client.setex(metadata_key, CACHE_TTL, json.dumps(metadata))
                    fixes.append(f"Created metadata for key: {key}")
                
                if not redis_client.exists(counter_key):
                    issues.append(f"Missing access counter for key: {key}")
                    redis_client.set(counter_key, 1)
                    fixes.append(f"Created access counter for key: {key}")
                
            except Exception as e:
                issues.append(f"Error processing key {key}: {str(e)}")
        
        return {
            "status": "completed",
            "total_keys": len(all_keys),
            "issues_found": len(issues),
            "fixes_applied": len(fixes),
            "issues": issues,
            "fixes": fixes
        }
    except Exception as e:
        logger.error(f"Error verifying cache consistency: {str(e)}")
        return {"status": "error", "error": str(e)}

# Update cache_maintenance_task to include consistency check
async def cache_maintenance_task():
    """Background task to periodically optimize cache and check health."""
    while True:
        try:
            # Run optimization
            await optimize_cache()
            
            # Check health
            health = await get_cache_health()
            if health["status"] != "healthy":
                logger.warning(f"Cache health check failed: {health}")
            
            # Verify consistency
            consistency = await verify_cache_consistency()
            if consistency["issues_found"] > 0:
                logger.warning(f"Cache consistency check found issues: {consistency}")
            
            # Wait for next interval
            await asyncio.sleep(6 * 60 * 60)  # 6 hours
        except Exception as e:
            logger.error(f"Error in cache maintenance task: {str(e)}")
            await asyncio.sleep(300)  # 5 minutes

# Add after Redis Configuration section

async def initialize_cache_counters():
    """Initialize cache counters if they don't exist."""
    try:
        redis_client = get_redis_client()
        counters = ['cache_hits', 'cache_misses', 'total_cached_entries']
        for counter in counters:
            if not redis_client.exists(counter):
                redis_client.set(counter, 0)
        logger.info("✨ Cache counters initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing cache counters: {str(e)}")

# Update start_cache_maintenance to include initialization
async def start_cache_maintenance():
    """Start the cache maintenance task and initialize counters."""
    await initialize_cache_counters()
    asyncio.create_task(cache_maintenance_task())

@with_redis_client
def clear_cache(client: redis.Redis) -> Dict[str, Any]:
    """Clear all cached responses."""
    try:
        # Get all cache keys
        pattern = "allstances:*"
        keys = client.keys(pattern)
        
        if keys:
            # Delete all matching keys
            client.delete(*keys)
            logger.info(f"🧹 Cleared {len(keys)} cache entries")
            return {"status": "success", "cleared_entries": len(keys)}
        else:
            logger.info("No cache entries to clear")
            return {"status": "success", "cleared_entries": 0}
            
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return {"status": "error", "error": str(e)}