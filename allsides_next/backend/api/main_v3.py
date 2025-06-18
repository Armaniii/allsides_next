import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, RootModel
import instructor
from django.http import StreamingHttpResponse
import redis
import hashlib
import pickle
import time
from functools import lru_cache, wraps
from django.core.cache import cache
from django.conf import settings
import asyncio
import httpx
import websockets
import uuid
import re
from urllib.parse import urlparse

# Set up logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local LLM client and helpers
try:
    from .llm_client import llm_client
    from .llm_helpers import (
        QueryFormatter, 
        FollowUpQuestionGenerator,
        CoreArgumentSummarizer,
        DialecticalAnalyzer,
        enhance_response_with_llm,
        enhance_response_with_llm_sync
    )
    LOCAL_LLM_AVAILABLE = True
    logger.info("✅ Local LLM client and helpers imported successfully")
except ImportError as e:
    LOCAL_LLM_AVAILABLE = False
    logger.warning(f"⚠️ Local LLM client not available: {e}")

def extract_domain_from_url(url: str) -> str:
    """Extract domain name from URL for display purposes.
    Returns '' if invalid or missing.
    """
    try:
        if not url or not isinstance(url, str):
            return ""
        # Ensure protocol for parsing
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc.lower().rstrip(".")
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception as e:
        logger.warning(f"Error extracting domain from URL '{url}': {str(e)}")
        return ""

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # Default 1 hour cache
CACHE_VERSION = "v1"  # Increment this when making breaking changes to the response format

# Junto Position Generation API Configuration
JUNTO_API_URL = os.getenv("JUNTO_API_URL", "http://209.38.6.6:80")
JUNTO_API_KEY = os.getenv("JUNTO_API_KEY")
JUNTO_ENABLED = os.getenv("JUNTO_ENABLED", "true").lower() == "true"

# Junto API settings
JUNTO_MODEL = "gpt-4o-mini" #os.getenv("JUNTO_MODEL", "gpt-4o-mini")
JUNTO_MAX_ITERATIONS = 1 #int(os.getenv("JUNTO_MAX_ITERATIONS", "1"))
JUNTO_SIMILARITY_THRESHOLD = 0.85 #float(os.getenv("JUNTO_SIMILARITY_THRESHOLD", "0.85"))
JUNTO_TEMPERATURE_STEP = 0.1 #float(os.getenv("JUNTO_TEMPERATURE_STEP", "0.1"))
JUNTO_TIMEOUT = 600 #int(os.getenv("JUNTO_TIMEOUT", "300"))  # 5 minutes timeout

# Junto Evidence Finder API settings
JUNTO_EVIDENCE_MODEL = "gpt-4o-mini" #os.getenv("JUNTO_EVIDENCE_MODEL", "gpt-4o-mini")
JUNTO_EVIDENCE_ITERATIONS = 1 #int(os.getenv("JUNTO_EVIDENCE_ITERATIONS", "1"))
# HARDCODED FOR TESTING: Enable Junto Evidence Finder
JUNTO_EVIDENCE_ENABLED = True  # os.getenv("JUNTO_EVIDENCE_ENABLED", "false").lower() == "true"

# Pipeline selection configuration  
# HARDCODED FOR TESTING: Use Junto Evidence Pipeline
USE_JUNTO_EVIDENCE_PIPELINE = True  # os.getenv("USE_JUNTO_EVIDENCE_PIPELINE", "false").lower() == "true"

# Application-scoped Redis clients
redis_client = None  # Async client
sync_redis_client = None  # Sync client

def initialize_redis_client():
    """Initialize the application-scoped async Redis client."""
    global redis_client
    if redis_client is None:
        import redis.asyncio as async_redis
        redis_client = async_redis.Redis(
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
    return redis_client

def initialize_sync_redis_client():
    """Initialize the synchronous Redis client for sync operations."""
    global sync_redis_client
    if sync_redis_client is None:
        import redis
        sync_redis_client = redis.Redis(
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
    return sync_redis_client

def get_redis_client():
    """Get the application-scoped async Redis client."""
    global redis_client
    if redis_client is None:
        redis_client = initialize_redis_client()
    return redis_client

def get_sync_redis_client():
    """Get the application-scoped sync Redis client."""
    global sync_redis_client
    if sync_redis_client is None:
        sync_redis_client = initialize_sync_redis_client()
    return sync_redis_client

def with_redis_client(func):
    """Decorator to handle sync Redis client acquisition and error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            client = get_sync_redis_client()  # Use sync client for sync operations
            return func(client, *args, **kwargs)
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error in {func.__name__}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

# Note: Cache clearing moved to end of file after clear_cache function is defined

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

# Synchronous wrapper functions for Django views compatibility
def get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    """Synchronous wrapper for cache retrieval."""
    try:
        from asgiref.sync import async_to_sync
        return async_to_sync(get_cached_response_async)(cache_key)
    except Exception as e:
        logger.error(f"Error in sync cache retrieval: {str(e)}")
        return None

def set_cached_response(cache_key: str, response_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
    """Synchronous wrapper for cache storage."""
    try:
        from asgiref.sync import async_to_sync
        return async_to_sync(set_cached_response_async)(cache_key, response_data, ttl)
    except Exception as e:
        logger.error(f"Error in sync cache storage: {str(e)}")
        return False

async def get_cached_response_async(cache_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve a cached response using async Redis."""
    try:
        client = get_redis_client()
        cached_data = await client.get(cache_key)
        
        if cached_data:
            try:
                # Increment access counter
                await client.incr(f"{cache_key}:access_count")
                
                # Update TTL based on new access count
                new_ttl = calculate_ttl(cache_key)
                await client.expire(cache_key, new_ttl)
                await client.expire(f"{cache_key}:access_count", new_ttl)
                
                # Update metadata
                metadata = {
                    'last_accessed': time.time(),
                    'ttl': new_ttl,
                    'access_count': int(await client.get(f"{cache_key}:access_count") or 1)
                }
                await client.setex(f"{cache_key}:metadata", new_ttl, json.dumps(metadata))
                
                # Increment hit counter
                await client.incr('cache_hits')
                
                logger.info(f"✅ Cache HIT for key: {cache_key}")
                return pickle.loads(cached_data)
            except Exception as e:
                logger.warning(f"Non-critical error updating cache metadata: {str(e)}")
                return pickle.loads(cached_data)
        
        # Increment miss counter
        await client.incr('cache_misses')
        logger.info(f"❌ Cache MISS for key: {cache_key}")
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving from cache: {str(e)}")
        return None


def get_cached_response_sync(cache_key: str) -> Optional[Dict[str, Any]]:
    """Synchronous wrapper for the async cache getter."""
    return asyncio.run(get_cached_response_async(cache_key))


async def set_cached_response_async(cache_key: str, response_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
    """Store a response in cache."""
    try:
        client = get_redis_client()
        # Calculate TTL if not provided
        if ttl is None:
            try:
                ttl = calculate_ttl(cache_key)
            except:
                ttl = CACHE_TTL
        
        # Core caching operation
        serialized_data = pickle.dumps(response_data)
        success = await client.setex(cache_key, ttl, serialized_data)
        
        if success:
            try:
                # Increment access counter
                await client.incr(f"{cache_key}:access_count")
                await client.expire(f"{cache_key}:access_count", ttl)
                
                # Store metadata
                metadata = {
                    'created_at': time.time(),
                    'ttl': ttl,
                    'access_count': 1
                }
                await client.setex(f"{cache_key}:metadata", ttl, json.dumps(metadata))
                
                # Increment total entries counter
                await client.incr('total_cached_entries')
                
                logger.info(f"💾 Successfully cached response with key: {cache_key}")
            except Exception as e:
                logger.warning(f"Non-critical error setting cache metadata: {str(e)}")
        
        return bool(success)
    except Exception as e:
        logger.error(f"Error setting cache: {str(e)}")
        return False


def set_cached_response_sync(cache_key: str, response_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
    """Synchronous wrapper for the async cache setter."""
    return asyncio.run(set_cached_response_async(cache_key, response_data, ttl))


# Prompt management configuration
PROMPT_MANAGER = "langfuse"  # os.getenv("PROMPT_MANAGER", "default")

# Default system prompt (fallback when Langfuse is not available)
DEFAULT_SYSTEM_PROMPT = """Task: You will be given a topic or positions regarding a specific topic. Your task is to provide authoritative and reputable arguments supporting different positions with citations. Each supporting argument must consist of a definitive claim supported by at least one premise or piece of evidence.

Structure Requirements:
1. For each distinct position on the topic, provide:
   - Supporting Arguments: Multiple substantive claims or premises that strengthen the core argument
   - Citations: Each supporting argument should include a link to a verifiable source when possible

Output Format: 
- Present your analysis in a JSON-like array structure
- Each object in the array represents a stance
- Within each stance object, include a key-value pair where:
  * Key = The stance statement
  * Value = An array of supporting arguments with citations
- Use single quotes for any quoted text

Guidelines:
- Maintain absolute neutrality in your analysis
- Provide 3-5 substantive supporting arguments for each position
- Present the strongest version of each argument
- Analyze nuanced positions rather than just extreme viewpoints"""

# Log environment variables (excluding sensitive data)
logger.debug("Environment variables loaded:")
logger.debug(f"PROMPT_MANAGER: {PROMPT_MANAGER}")
logger.debug(f"JUNTO_ENABLED: {JUNTO_ENABLED}")
logger.debug(f"JUNTO_MAX_ITERATIONS: {JUNTO_MAX_ITERATIONS}")
logger.debug(f"JUNTO_API_KEY configured: {bool(JUNTO_API_KEY)}")
logger.debug(f"JUNTO_EVIDENCE_ENABLED: {JUNTO_EVIDENCE_ENABLED}")
logger.debug(f"USE_JUNTO_EVIDENCE_PIPELINE: {USE_JUNTO_EVIDENCE_PIPELINE}")
logger.debug(f"LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST', 'NOT_SET')}")
logger.debug(f"LANGFUSE_PUBLIC_KEY configured: {bool(os.getenv('LANGFUSE_PUBLIC_KEY'))}")
logger.debug(f"LANGFUSE_SECRET_KEY configured: {bool(os.getenv('LANGFUSE_SECRET_KEY'))}")

# Initialize Langfuse only if selected
langfuse = None
if PROMPT_MANAGER == "langfuse":
    try:
        from langfuse import Langfuse
        # Get Langfuse configuration
        langfuse_host = os.getenv("LANGFUSE_HOST", "https://vector.allsides.com")
        langfuse_public_key = "" #os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key ="" #os.getenv("LANGFUSE_SECRET_KEY")

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


# OpenAI pricing (as of 2024)
GPT4_INPUT_COST_PER_1K_TOKENS = 0.03
GPT4_OUTPUT_COST_PER_1K_TOKENS = 0.06
GPT4O_INPUT_COST_PER_1K_TOKENS = 0.0025  # GPT-4o pricing
GPT4O_OUTPUT_COST_PER_1K_TOKENS = 0.01
GPT4O_MINI_INPUT_COST_PER_1K_TOKENS = 0.00015  # GPT-4o mini pricing
GPT4O_MINI_OUTPUT_COST_PER_1K_TOKENS = 0.0006

def calculate_openai_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4") -> float:
    """Calculate the cost of an OpenAI API call based on token usage."""
    try:
        if "mini" in model.lower() or model.startswith("gpt-4o-mini"):
            input_cost = (input_tokens / 1000) * GPT4O_MINI_INPUT_COST_PER_1K_TOKENS
            output_cost = (output_tokens / 1000) * GPT4O_MINI_OUTPUT_COST_PER_1K_TOKENS
            return round(input_cost + output_cost, 6)
        elif model.startswith("gpt-4o"):
            input_cost = (input_tokens / 1000) * GPT4O_INPUT_COST_PER_1K_TOKENS
            output_cost = (output_tokens / 1000) * GPT4O_OUTPUT_COST_PER_1K_TOKENS
            return round(input_cost + output_cost, 6)
        elif model.startswith("gpt-4"):
            input_cost = (input_tokens / 1000) * GPT4_INPUT_COST_PER_1K_TOKENS
            output_cost = (output_tokens / 1000) * GPT4_OUTPUT_COST_PER_1K_TOKENS
            return round(input_cost + output_cost, 6)
        else:
            # Default to GPT-4o mini pricing for unknown models
            input_cost = (input_tokens / 1000) * GPT4O_MINI_INPUT_COST_PER_1K_TOKENS
            output_cost = (output_tokens / 1000) * GPT4O_MINI_OUTPUT_COST_PER_1K_TOKENS
            return round(input_cost + output_cost, 6)
    except Exception as e:
        logger.error(f"Error calculating cost: {str(e)}")
        return 0.0


# Junto Position Generation API Client
class JuntoPositionGenerator:
    """Client for the Junto Position Generation API"""
    
    def __init__(self):
        self.api_url = JUNTO_API_URL
        self.api_key = JUNTO_API_KEY
        self.enabled = JUNTO_ENABLED and bool(self.api_key)
        
        if self.enabled:
            logger.info(f"Junto Position Generator initialized - API URL: {self.api_url}")
        else:
            logger.info("Junto Position Generator disabled (no API key or disabled in config)")
    
    async def generate_positions(self, question: str, trace=None) -> List[str]:
        """Generate positions for a given question using the Junto API"""
        if not self.enabled:
            logger.warning("Junto API is disabled, returning empty positions list")
            return []
        
        task_id = None
        positions = []
        
        try:
            # Start position generation task
            task_id = await self._start_position_generation(question)
            logger.info(f"Started Junto position generation task: {task_id}")
            
            # Add span for position generation if trace exists
            position_span = None
            if trace:
                position_span = trace.span(
                    name="junto_position_generation",
                    metadata={
                        "question": question,
                        "task_id": task_id,
                        "api_url": self.api_url,
                        "model": JUNTO_MODEL,
                        "max_iterations": JUNTO_MAX_ITERATIONS,
                        "similarity_threshold": JUNTO_SIMILARITY_THRESHOLD
                    }
                )
            
            # Monitor task progress and get results
            positions, cost_estimate = await self._monitor_task_progress(task_id)
            
            if position_span:
                position_span.end(
                    output=positions,
                    metadata={
                        "positions_count": len(positions),
                        "task_completed": True,
                        "junto_cost_estimate_usd": cost_estimate,
                        "cost_source": "junto_api"
                    }
                )
            
            logger.info(f"Generated {len(positions)} positions from Junto API")
            return positions
            
        except Exception as e:
            logger.error(f"Error generating positions with Junto API: {str(e)}")
            
            if trace and position_span:
                position_span.end(
                    output=None,
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "task_id": task_id,
                        "junto_cost_estimate_usd": 0.0,
                        "cost_source": "junto_api"
                    }
                )
            
            # Return empty list on error - main processing can continue
            return []
    
    async def _start_position_generation(self, question: str) -> str:
        """Start a position generation task and return task_id"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"X-API-Key": self.api_key}
            payload = {
                "question": question,
                "model": JUNTO_MODEL,
                "max_iterations": JUNTO_MAX_ITERATIONS,
                "similarity_threshold": JUNTO_SIMILARITY_THRESHOLD,
                "temperature_step": JUNTO_TEMPERATURE_STEP
            }
            
            response = await client.post(
                f"{self.api_url}/generate-positions/",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            return result["task_id"]
    
    async def _monitor_task_progress(self, task_id: str) -> tuple[List[str], float]:
        """Monitor task progress via WebSocket and return final positions and cost estimate"""
        ws_url = self.api_url.replace("http://", "ws://").replace("https://", "wss://")
        uri = f"{ws_url}/ws/{task_id}"
        
        try:
            # Use websockets for real-time updates with increased timeout
            async with websockets.connect(uri, ping_timeout=30, close_timeout=10, open_timeout=30) as websocket:
                start_time = time.time()
                total_cost_estimate = 0.0
                
                while True:
                    # Check timeout
                    if time.time() - start_time > JUNTO_TIMEOUT:
                        logger.warning(f"Junto API timeout after {JUNTO_TIMEOUT} seconds")
                        break
                    
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        parsed = json.loads(message)
                        
                        logger.debug(f"Junto API progress: {parsed.get('status', 'unknown')}")
                        
                        # Handle progress updates and extract cost estimates
                        if parsed.get("progress_updates"):
                            for update in parsed["progress_updates"]:
                                update_msg = update.get('message', 'Progress update')
                                logger.info(f"Junto API: {update_msg}")
                                
                                # Extract cost estimate from progress messages
                                if update.get("type") == "cost_estimate" and "$" in update_msg:
                                    try:
                                        # Extract cost from message like "Estimated cost for initial claims: $0.13 for 8,000 input tokens"
                                        cost_match = re.search(r'\$(\d+\.?\d*)', update_msg)
                                        if cost_match:
                                            cost = float(cost_match.group(1))
                                            total_cost_estimate += cost
                                            logger.info(f"Junto API cost estimate: ${cost}")
                                    except (ValueError, AttributeError) as e:
                                        logger.debug(f"Could not parse cost from message: {update_msg}")
                        
                        # Check if completed
                        if parsed.get("status") == "completed":
                            positions = parsed.get("result", [])
                            logger.info(f"Junto API completed with {len(positions)} positions, total cost estimate: ${total_cost_estimate}")
                            logger.info(f"Junto API raw positions JSON: {json.dumps(positions, indent=2)}")
                            return positions, total_cost_estimate
                        
                        # Check if failed
                        elif parsed.get("status") == "failed":
                            error_msg = parsed.get("error", "Unknown error")
                            logger.error(f"Junto API task failed: {error_msg}")
                            return [], total_cost_estimate
                            
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for Junto API WebSocket message")
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Junto API WebSocket connection closed")
                        break
                        
        except Exception as e:
            logger.info(f"WebSocket connection failed (this is normal for some network configurations), falling back to HTTP polling: {str(e)}")
            # Fallback to HTTP polling
            return await self._poll_task_status(task_id)
        
        return [], 0.0
    
    async def _poll_task_status(self, task_id: str) -> tuple[List[str], float]:
        """Fallback method to poll task status via HTTP"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"X-API-Key": self.api_key}
            start_time = time.time()
            
            while time.time() - start_time < JUNTO_TIMEOUT:
                try:
                    response = await client.get(
                        f"{self.api_url}/task/{task_id}",
                        headers=headers
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    status = result.get("status")
                    
                    if status == "completed":
                        positions = result.get("result", [])
                        logger.info(f"Junto API completed via polling with {len(positions)} positions")
                        if positions:
                            logger.info(f"📋 Positions from polling: {json.dumps(positions, indent=2)}")
                        # Note: HTTP polling doesn't provide real-time cost estimates, 
                        # so we return 0.0 as a placeholder
                        return positions, 0.0
                    elif status == "failed":
                        error_msg = result.get("error", "Unknown error")
                        logger.error(f"Junto API task failed: {error_msg}")
                        return [], 0.0
                    else:
                        # Still in progress, wait before next poll
                        await asyncio.sleep(2)
                        
                except Exception as e:
                    logger.warning(f"Error polling task status: {str(e)}")
                    await asyncio.sleep(5)
            
            logger.warning(f"Junto API polling timeout after {JUNTO_TIMEOUT} seconds")
            return [], 0.0


# Junto Evidence Finder API Client
class JuntoEvidenceFinder:
    """Client for the Junto Evidence Finder API"""
    
    def __init__(self):
        self.api_url = JUNTO_API_URL
        self.api_key = JUNTO_API_KEY
        self.enabled = JUNTO_EVIDENCE_ENABLED and bool(self.api_key)
        
        if self.enabled:
            logger.info(f"Junto Evidence Finder initialized - API URL: {self.api_url}")
        else:
            logger.info("Junto Evidence Finder disabled (no API key or disabled in config)")
    
    async def find_evidence(self, claim: str, model: str = None, iterations: int = None, trace=None) -> Dict[str, Any]:
        """Find evidence for a given claim using the Junto Evidence API"""
        if not self.enabled:
            logger.warning("Junto Evidence API is disabled, returning empty evidence")
            return {"evidence": [], "cost_estimate": 0.0}
        
        # Use environment defaults if not provided
        if model is None:
            model = JUNTO_EVIDENCE_MODEL
        if iterations is None:
            iterations = JUNTO_EVIDENCE_ITERATIONS
        
        task_id = None
        evidence_result = {"evidence": [], "cost_estimate": 0.0}
        evidence_span = None  # Initialize before try block to avoid scope issues
        
        try:
            # Start evidence finding task
            task_id = await self._start_evidence_task(claim, model, iterations)
            logger.info(f"Started Junto evidence finding task: {task_id}")
            
            # Add span for evidence finding if trace exists
            if trace:
                evidence_span = trace.span(
                    name="junto_evidence_finding",
                    metadata={
                        "claim": claim,
                        "task_id": task_id,
                        "api_url": self.api_url,
                        "model": model,
                        "iterations": iterations
                    }
                )
            
            # Monitor task progress and get results
            evidence, cost_estimate = await self._monitor_evidence_task_progress(task_id)
            evidence_result = {"evidence": evidence, "cost_estimate": cost_estimate}
            
            if evidence_span:
                evidence_span.end(
                    output=evidence_result,
                    metadata={
                        "evidence_count": len(evidence),
                        "task_completed": True,
                        "junto_cost_estimate_usd": cost_estimate,
                        "cost_source": "junto_evidence_api"
                    }
                )
            
            logger.info(f"Found {len(evidence)} pieces of evidence from Junto Evidence API")
            return evidence_result
            
        except Exception as e:
            logger.error(f"Error finding evidence with Junto API: {str(e)}")
            
            if trace and evidence_span:
                evidence_span.end(
                    output=None,
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "task_id": task_id,
                        "junto_cost_estimate_usd": 0.0,
                        "cost_source": "junto_evidence_api"
                    }
                )
            
            # Return empty evidence on error - main processing can continue
            return {"evidence": [], "cost_estimate": 0.0}
    
    async def _start_evidence_task(self, claim: str, model: str, iterations: int) -> str:
        """Start an evidence finding task and return task_id"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"X-API-Key": self.api_key}
            payload = {
                "claim": claim,
                "model": model,
                "iterations": iterations
            }
            
            response = await client.post(
                f"{self.api_url}/find-evidence/",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            return result["task_id"]
    
    async def _monitor_evidence_task_progress(self, task_id: str) -> tuple[List[Dict[str, Any]], float]:
        """Monitor evidence task progress via WebSocket and return final evidence and cost estimate"""
        ws_url = self.api_url.replace("http://", "ws://").replace("https://", "wss://")
        uri = f"{ws_url}/ws/{task_id}"
        
        try:
            # Use websockets for real-time updates with increased timeout
            async with websockets.connect(uri, ping_timeout=30, close_timeout=10, open_timeout=30) as websocket:
                start_time = time.time()
                total_cost_estimate = 0.0
                
                while True:
                    # Check timeout
                    if time.time() - start_time > JUNTO_TIMEOUT:
                        logger.warning(f"Junto Evidence API timeout after {JUNTO_TIMEOUT} seconds")
                        break
                    
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        parsed = json.loads(message)
                        
                        logger.debug(f"Junto Evidence API progress: {parsed.get('status', 'unknown')}")
                        
                        # Handle progress updates and extract cost estimates
                        if parsed.get("progress_updates"):
                            for update in parsed["progress_updates"]:
                                update_msg = update.get('message', 'Progress update')
                                logger.info(f"Junto Evidence API: {update_msg}")
                                
                                # Extract cost estimate from progress messages
                                if update.get("type") == "cost_estimate" and "$" in update_msg:
                                    try:
                                        # Extract cost from message like "Estimated cost for evidence search: $0.15"
                                        cost_match = re.search(r'\$(\d+\.?\d*)', update_msg)
                                        if cost_match:
                                            cost = float(cost_match.group(1))
                                            total_cost_estimate += cost
                                            logger.info(f"Junto Evidence API cost estimate: ${cost}")
                                    except (ValueError, AttributeError) as e:
                                        logger.debug(f"Could not parse cost from message: {update_msg}")
                        
                        # Check if completed
                        if parsed.get("status") == "completed":
                            evidence = parsed.get("result", [])
                            logger.info(f"Junto Evidence API completed with {len(evidence)} pieces of evidence, total cost estimate: ${total_cost_estimate}")
                            logger.info(f"Junto Evidence API raw evidence JSON: {json.dumps(evidence[:3], indent=2)}...")  # Log first 3 for brevity
                            return evidence, total_cost_estimate
                        
                        # Check if failed
                        elif parsed.get("status") == "failed":
                            error_msg = parsed.get("error", "Unknown error")
                            logger.error(f"Junto Evidence API task failed: {error_msg}")
                            return [], total_cost_estimate
                            
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for Junto Evidence API WebSocket message")
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Junto Evidence API WebSocket connection closed")
                        break
                        
        except Exception as e:
            logger.info(f"WebSocket connection failed for evidence API (this is normal for some network configurations), falling back to HTTP polling: {str(e)}")
            # Fallback to HTTP polling
            return await self._poll_evidence_task_status(task_id)
        
        return [], 0.0
    
    async def _poll_evidence_task_status(self, task_id: str) -> tuple[List[Dict[str, Any]], float]:
        """Fallback method to poll evidence task status via HTTP"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"X-API-Key": self.api_key}
            start_time = time.time()
            
            while time.time() - start_time < JUNTO_TIMEOUT:
                try:
                    response = await client.get(
                        f"{self.api_url}/task/{task_id}",
                        headers=headers
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    status = result.get("status")
                    
                    if status == "completed":
                        evidence = result.get("result", [])
                        logger.info(f"Junto Evidence API completed via polling with {len(evidence)} pieces of evidence")
                        if evidence:
                            logger.info(f"📋 Evidence from polling: {json.dumps(evidence[:2], indent=2)}...")  # Log first 2 for brevity
                        # Note: HTTP polling doesn't provide real-time cost estimates, 
                        # so we return 0.0 as a placeholder
                        return evidence, 0.0
                    elif status == "failed":
                        error_msg = result.get("error", "Unknown error")
                        logger.error(f"Junto Evidence API task failed: {error_msg}")
                        return [], 0.0
                    else:
                        # Still in progress, wait before next poll
                        await asyncio.sleep(2)
                        
                except Exception as e:
                    logger.warning(f"Error polling evidence task status: {str(e)}")
                    await asyncio.sleep(5)
            
            logger.warning(f"Junto Evidence API polling timeout after {JUNTO_TIMEOUT} seconds")
            return [], 0.0


# Initialize Junto Position Generator and Evidence Finder
junto_generator = JuntoPositionGenerator()
junto_evidence_finder = JuntoEvidenceFinder()




# Aesthetic Progress Updates for Parallel Processing
class AestheticProgressTracker:
    """Manages smooth, aesthetic progress updates for parallel processing"""
    
    def __init__(self, total_positions: int, progress_callback=None):
        self.total_positions = total_positions
        self.progress_callback = progress_callback
        self.completed_count = 0
        self.start_time = time.time()
        self.progress_phases = [
            "🔍 Analyzing diverse perspectives...",
            "📚 Searching academic sources...",
            "🌐 Gathering web evidence...", 
            "🔬 Examining research data...",
            "📰 Reviewing news reports...",
            "🏛️ Checking institutional sources...",
            "✨ Synthesizing findings..."
        ]
        self.current_phase = 0
        self.last_update = 0
        
    async def start_search(self):
        """Called when search begins"""
        if self.progress_callback:
            self.progress_callback("🚀 Initiating parallel evidence discovery...")
            await asyncio.sleep(0.5)  # Brief pause for aesthetic
            
    async def update_progress(self, completed: int):
        """Update progress with smooth transitions"""
        self.completed_count = completed
        current_time = time.time()
        
        # Only update every 1.5 seconds for smooth transitions
        if current_time - self.last_update < 1.5:
            return
            
        if self.progress_callback:
            # Calculate completion percentage
            completion_pct = (self.completed_count / self.total_positions) * 100
            
            # Select appropriate phase based on completion
            phase_index = min(int(completion_pct / 15), len(self.progress_phases) - 1)
            
            # Create smooth progress message
            if completion_pct < 90:
                message = f"{self.progress_phases[phase_index]} ({completion_pct:.0f}%)"
            else:
                message = "🔄 Finalizing evidence collection..."
                
            self.progress_callback(message)
            self.last_update = current_time
            
    async def complete_search(self, successful: int, total_evidence: int, total_time: float):
        """Called when search completes"""
        if self.progress_callback:
            await asyncio.sleep(0.3)  # Brief pause before final message
            self.progress_callback(f"✅ Evidence collection complete! Found {total_evidence} items from {successful} perspectives in {total_time:.1f}s")


# Parallel Evidence Search Implementation
async def parallel_evidence_search(positions: List[str], junto_evidence_finder: JuntoEvidenceFinder, 
                                 trace=None, progress_callback=None) -> List[Dict[str, Any]]:
    """
    Search evidence for all positions in parallel with aesthetic progress updates.
    Returns list of evidence results with position index preserved.
    """
    # Initialize aesthetic progress tracker
    progress_tracker = AestheticProgressTracker(len(positions), progress_callback)
    await progress_tracker.start_search()
    
    # Track completion for smooth progress updates
    completed_positions = 0
    completion_lock = asyncio.Lock()
    
    async def search_single_position(position: str, index: int) -> Dict[str, Any]:
        """Search evidence for a single position"""
        nonlocal completed_positions
        
        try:
            start_time = time.time()
            evidence_result = await junto_evidence_finder.find_evidence(position, trace=trace)
            elapsed = time.time() - start_time
            
            result = {
                "position": position,
                "index": index,
                "evidence": evidence_result.get("evidence", []) if evidence_result else [],
                "evidence_count": len(evidence_result.get("evidence", [])) if evidence_result else 0,
                "cost": evidence_result.get("cost_estimate", 0.0) if evidence_result else 0.0,
                "search_time": elapsed,
                "success": True
            }
            
            # Update progress counter thread-safely
            async with completion_lock:
                completed_positions += 1
                await progress_tracker.update_progress(completed_positions)
            
            logger.info(f"✅ Position {index+1} evidence search completed in {elapsed:.1f}s: {result['evidence_count']} items found")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error finding evidence for position '{position}': {str(e)}")
            
            # Update progress counter even for failures
            async with completion_lock:
                completed_positions += 1
                await progress_tracker.update_progress(completed_positions)
            
            return {
                "position": position,
                "index": index,
                "evidence": [],
                "evidence_count": 0,
                "cost": 0.0,
                "search_time": 0.0,
                "success": False,
                "error": str(e)
            }
    
    # Create tasks for all positions
    tasks = [search_single_position(pos, i) for i, pos in enumerate(positions)]
    
    # Execute all searches in parallel
    logger.info(f"🚀 Starting parallel evidence search for {len(positions)} positions")
    start_time = time.time()
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out any exceptions and ensure all results have proper structure
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"❌ Task {i} failed with exception: {result}")
            # Create a fallback result structure
            valid_results.append({
                "position": positions[i],
                "index": i,
                "evidence": [],
                "evidence_count": 0,
                "cost": 0.0,
                "search_time": 0.0,
                "success": False,
                "error": str(result)
            })
        elif isinstance(result, dict) and "index" in result:
            valid_results.append(result)
        else:
            logger.error(f"❌ Task {i} returned invalid result structure: {result}")
            # Create a fallback result structure
            valid_results.append({
                "position": positions[i],
                "index": i,
                "evidence": [],
                "evidence_count": 0,
                "cost": 0.0,
                "search_time": 0.0,
                "success": False,
                "error": "Invalid result structure"
            })
    
    total_time = time.time() - start_time
    successful_searches = sum(1 for r in valid_results if r.get("success", False))
    total_evidence = sum(r.get("evidence_count", 0) for r in valid_results)
    total_cost = sum(r.get("cost", 0.0) for r in valid_results)
    
    # Final progress update
    await progress_tracker.complete_search(successful_searches, total_evidence, total_time)
    
    logger.info(f"🎉 Parallel evidence search completed in {total_time:.1f}s - "
                f"{successful_searches}/{len(positions)} successful, "
                f"{total_evidence} total evidence items, "
                f"${total_cost:.2f} total cost")
    
    # Sort results by original index to maintain position order
    valid_results.sort(key=lambda x: x.get("index", 0))
    
    return valid_results


async def progressive_format_and_enhance(evidence_results: List[Dict[str, Any]], 
                                       normalized_topic: str,
                                       reference_id_counter: int = 1,
                                       progress_callback=None) -> Tuple[List[Dict[str, Any]], List["Reference"]]:
    """
    Progressively format evidence and enhance with LLM as results arrive.
    This allows enhancement to start before all evidence is collected.
    """
    formatted_arguments = []
    all_references = []
    current_ref_id = reference_id_counter
    
    # Process each evidence result and prepare for enhancement
    for evidence_data in evidence_results:
        # Defensive access to prevent KeyError
        if not isinstance(evidence_data, dict):
            logger.warning(f"Invalid evidence data structure: {type(evidence_data)}")
            continue
            
        position = evidence_data.get("position", "Unknown Position")
        evidence_list = evidence_data.get("evidence", [])
        
        # Format evidence into structured argument
        supporting_args = []
        position_references = []
        supporting_evidence_items = []
        refuting_evidence_items = []
        
        # Process evidence items
        for evidence_item in evidence_list:
            if isinstance(evidence_item, dict):
                quote = evidence_item.get("quote", "")
                url = evidence_item.get("url", "")
                stance = evidence_item.get("stance", "supports")
                reasoning = evidence_item.get("reasoning", "")
                source_type = evidence_item.get("source_type", "secondary")

                # Always ensure protocol
                if url and not url.startswith(("http://", "https://")):
                    url = f"https://{url}"
                domain = extract_domain_from_url(url)
                # Skip if domain or url is empty
                if not domain or not url:
                    continue
                
                # Create reference
                reference = Reference(
                    id=current_ref_id,
                    title=domain,
                    url=url,
                    source_type=source_type,
                    stance=stance,
                    domain=domain
                )
                all_references.append(reference)
                position_references.append(reference)
                
                # Create evidence entry
                evidence_entry = EvidenceItem(
                    quote=quote,
                    citation_id=current_ref_id,
                    reasoning=reasoning,
                    stance=stance,
                    url=url,
                    domain=domain,
                    formatted=f"{quote} [{domain}]"
                )
                
                current_ref_id += 1
                
                if stance == "supports":
                    supporting_evidence_items.append(evidence_entry)
                elif stance == "refutes":
                    refuting_evidence_items.append(evidence_entry)
        
        # Create supporting arguments from evidence
        for evidence in supporting_evidence_items[:10]:
            supporting_args.append(evidence.formatted)
        
        # Create metadata
        evidence_metadata = EvidenceMetadata(
            supporting_evidence_count=len(supporting_evidence_items),
            refuting_evidence_count=len(refuting_evidence_items),
            total_evidence_count=len(supporting_evidence_items) + len(refuting_evidence_items),
            primary_sources=len([ref for ref in position_references if ref.source_type == "primary"]),
            secondary_sources=len([ref for ref in position_references if ref.source_type == "secondary"])
        )
        
        # Create detailed evidence
        detailed_evidence = DetailedEvidence(
            supporting=supporting_evidence_items,
            refuting=refuting_evidence_items
        )
        
        # Create stance object
        stance_obj = JuntoEvidenceStance(
            stance=position,
            supporting_arguments=supporting_args[:10] if supporting_args else [f"Analysis of {position.lower()}"],
            references=position_references,
            evidence_metadata=evidence_metadata,
            detailed_evidence=detailed_evidence
        )
        
        formatted_arguments.append({
            "stance_obj": stance_obj,
            "index": evidence_data.get("index", 0),  # Defensive access to prevent KeyError
            "ready_for_enhancement": True
        })
    
    # Sort by index to maintain order
    formatted_arguments.sort(key=lambda x: x.get("index", 0))
    
    # If local LLM is available, enhance arguments in parallel
    if LOCAL_LLM_AVAILABLE:
        # TESTING LIMIT: Only enhance first 2 arguments to prevent long processing
        limited_arguments = formatted_arguments[:2]
        logger.info(f"🤖 Starting parallel LLM enhancement for {len(limited_arguments)} positions (LIMITED FOR TESTING)")
        logger.warning(f"🚨 TESTING: Limiting enhancement to {len(limited_arguments)} out of {len(formatted_arguments)} total arguments")
        
        if progress_callback:
            progress_callback("🤖 Analyzing arguments with AI...")
            await asyncio.sleep(0.4)  # Brief pause for aesthetic
        
        # Create enhancement tasks for LIMITED arguments only
        enhancement_tasks = []
        for arg_data in limited_arguments:
            stance_dict = arg_data["stance_obj"].dict()
            enhancement_tasks.append(enhance_single_stance(stance_dict, normalized_topic))
        
        if progress_callback:
            progress_callback("✨ Generating core summaries and insights...")
            await asyncio.sleep(0.3)
        
        # Run all enhancements in parallel
        enhanced_stances = await asyncio.gather(*enhancement_tasks, return_exceptions=True)
        
        if progress_callback:
            progress_callback("🔍 Performing dialectical analysis...")
            await asyncio.sleep(0.3)
        
        # Update LIMITED arguments with enhancements
        for i, enhanced in enumerate(enhanced_stances):
            if isinstance(enhanced, Exception):
                logger.error(f"Enhancement failed for position {i+1}: {enhanced}")
                # Keep original without enhancement
                limited_arguments[i]["enhanced_stance"] = limited_arguments[i]["stance_obj"].dict()
            else:
                limited_arguments[i]["enhanced_stance"] = enhanced
        
        # Set non-enhanced arguments to use original stances
        for i in range(len(limited_arguments), len(formatted_arguments)):
            formatted_arguments[i]["enhanced_stance"] = formatted_arguments[i]["stance_obj"].dict()
        
        # Update the enhanced limited arguments back into formatted_arguments
        for i, arg in enumerate(limited_arguments):
            formatted_arguments[i] = arg
                
        if progress_callback:
            progress_callback("✅ AI analysis complete! Preparing final response...")
            await asyncio.sleep(0.2)
    else:
        # No enhancement available, use original stances
        for arg_data in formatted_arguments:
            arg_data["enhanced_stance"] = arg_data["stance_obj"].dict()
            
        if progress_callback:
            progress_callback("📄 Formatting arguments...")
    
    # Extract final enhanced stances
    final_arguments = [arg["enhanced_stance"] for arg in formatted_arguments]
    
    return final_arguments, all_references


async def enhance_single_stance(stance_dict: Dict[str, Any], normalized_topic: str) -> Dict[str, Any]:
    """Enhance a single stance with LLM analysis"""
    try:
        # Import enhancement functions
        from .llm_helpers import CoreArgumentSummarizer, DialecticalAnalyzer
        
        # Generate core argument summary
        summary = await CoreArgumentSummarizer.summarize_position(stance_dict)
        stance_dict['core_argument_summary'] = summary
        
        # Perform dialectical analysis if evidence is available
        if stance_dict.get('detailed_evidence') or stance_dict.get('evidence_metadata'):
            evidence_data = {
                'detailed_evidence': stance_dict.get('detailed_evidence', {}),
                'evidence_metadata': stance_dict.get('evidence_metadata', {})
            }
            main_claim = stance_dict.get('stance', normalized_topic)
            
            dialectical_analysis = await DialecticalAnalyzer.analyze_dialectical_profile(
                evidence_data, main_claim
            )
            
            stance_dict['dialectical_analysis'] = dialectical_analysis
            stance_dict['key_perspectives'] = dialectical_analysis.get('key_perspectives', [])
            
            # Backward compatibility
            stance_dict['source_analysis'] = {
                'dialectical_summary': dialectical_analysis.get('dialectical_summary', ''),
                'supporting_profile': dialectical_analysis.get('supporting_profile', {}),
                'refuting_profile': dialectical_analysis.get('refuting_profile', {}),
                'key_perspectives': dialectical_analysis.get('key_perspectives', [])
            }
        
        return stance_dict
        
    except Exception as e:
        logger.error(f"Error enhancing stance: {e}")
        return stance_dict  # Return original if enhancement fails


async def complete_with_junto_evidence_pipeline(topic: str, diversity: float, num_stances: int = 3, user_id: Optional[str] = None, session_id: Optional[str] = None, request_metadata: Optional[Dict[str, Any]] = None, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Complete using the full Junto pipeline: Position Finder -> Evidence Finder -> Final formatting"""
    trace = None
    generation = None
    
    try:
        # Format query as question if local LLM is available
        if LOCAL_LLM_AVAILABLE:
            try:
                formatted_topic = QueryFormatter.format_query_as_question(topic)
                logger.info(f"📝 Formatted query for Junto: '{formatted_topic}' (original: '{topic}')")
                topic = formatted_topic
            except Exception as e:
                logger.warning(f"Failed to format query with LLM in Junto pipeline: {e}")
        
        # Normalize the topic first
        normalized_topic = normalize_query(topic)
        logger.info(f"🔗 Using Junto Evidence Pipeline for topic: '{normalized_topic}'")
        
        # Prepare trace metadata
        trace_metadata = {
            "topic": topic,
            "normalized_topic": normalized_topic,
            "diversity": diversity,
            "num_stances": num_stances,
            "pipeline": "junto_evidence",
            "cache_version": CACHE_VERSION,
            "junto_enabled": junto_generator.enabled,
            "junto_evidence_enabled": junto_evidence_finder.enabled,
            "junto_api_url": JUNTO_API_URL if junto_evidence_finder.enabled else None
        }
        
        # Add request metadata if provided
        if request_metadata:
            trace_metadata.update(request_metadata)
        
        # Start Langfuse Trace with proper context
        if langfuse:
            trace = langfuse.trace(
                name="allstances_junto_evidence_pipeline",
                user_id=user_id,
                session_id=session_id,
                metadata=trace_metadata,
                tags=["allstances", "junto_evidence_pipeline", f"stances_{num_stances}"]
            )
            logger.info(f"Started Langfuse trace for Junto evidence pipeline: {trace.id if trace else 'None'}")
        
        # Step 1: Generate positions using Junto Position API
        generated_positions = []
        if junto_generator.enabled:
            try:
                if progress_callback:
                    progress_callback("🎯 Identifying diverse positions...")
                logger.info(f"🎯 Step 1: Generating positions for topic: '{normalized_topic}'")
                junto_result = await junto_generator.generate_positions(normalized_topic, trace)
                if junto_result is None:
                    generated_positions = []
                elif isinstance(junto_result, tuple):
                    generated_positions = junto_result[0] if isinstance(junto_result[0], list) else []
                elif isinstance(junto_result, list):
                    generated_positions = junto_result
                else:
                    generated_positions = []
                logger.info(f"✅ Generated {len(generated_positions)} positions from Junto API")
                if progress_callback:
                    progress_callback(f"✅ Found {len(generated_positions)} diverse positions")
            except Exception as e:
                logger.error(f"❌ Error in position generation: {str(e)}")
                generated_positions = []
        
        if not generated_positions:
            logger.warning("No positions generated, falling back to regular pipeline")
            return complete(topic, diversity, num_stances, user_id, session_id, request_metadata)
        
        # TEMPORARY: Limit to first 2 positions for faster testing
        # TODO: Remove this limitation after testing is complete
        original_position_count = len(generated_positions)
        test_positions = generated_positions[:2]  # Only take first 2 positions
        logger.warning(f"🧪 TESTING MODE: Limiting evidence search to first 2 positions out of {original_position_count} total positions")
        logger.info(f"🧪 Testing with positions: {[pos[:50] + '...' if len(pos) > 50 else pos for pos in test_positions]}")
        
        # Notify user of testing mode
        if progress_callback:
            progress_callback(f"🧪 TESTING MODE: Processing only 2 positions for faster testing")
        
        # Step 2: Use PARALLEL evidence search for LIMITED positions
        all_evidence_results = []
        total_evidence_cost = 0.0
        
        if junto_evidence_finder.enabled:
            logger.info(f"🚀 Step 2: Parallel evidence search for {len(test_positions)} positions (LIMITED FOR TESTING)")
            
            # Run parallel evidence search on limited positions
            evidence_search_results = await parallel_evidence_search(test_positions, junto_evidence_finder, trace, progress_callback)
            
            if evidence_search_results:
                # Process results from parallel search
                for result in evidence_search_results:
                    # Defensive coding to prevent KeyError
                    position = result.get("position", "Unknown Position")
                    evidence = result.get("evidence", [])
                    evidence_count = result.get("evidence_count", len(evidence) if evidence else 0)
                    
                    all_evidence_results.append({
                        "position": position,
                        "evidence": evidence,
                        "evidence_count": evidence_count
                    })
                    total_evidence_cost += result.get("cost", 0.0)
                
                # Log summary
                successful = sum(1 for r in evidence_search_results if r.get("success", False))
                total_items = sum(r.get("evidence_count", 0) for r in evidence_search_results)
                logger.info(f"✅ Parallel search complete: {successful}/{len(test_positions)} successful (out of {original_position_count} total positions), "
                           f"{total_items} total evidence items, ${total_evidence_cost:.2f} cost")
            else:
                logger.error("Parallel evidence search failed, creating empty results")
                for position in test_positions:  # Use test_positions instead of all positions
                    all_evidence_results.append({
                        "position": position,
                        "evidence": [],
                        "evidence_count": 0
                    })
        else:
            logger.warning("Junto Evidence Finder disabled, creating positions without evidence")
            for position in test_positions:  # Use test_positions instead of all positions
                all_evidence_results.append({
                    "position": position,
                    "evidence": [],
                    "evidence_count": 0
                })
        
        # Step 3: Use PARALLEL formatting and enhancement
        if progress_callback:
            progress_callback("📄 Structuring evidence and preparing analysis...")
        
        # Run parallel formatting and enhancement
        try:
            formatting_result = await progressive_format_and_enhance(all_evidence_results, normalized_topic, 1, progress_callback)
            
            if formatting_result:
                arguments, all_references = formatting_result
                logger.info(f"✅ Parallel formatting complete: {len(arguments)} arguments enhanced")
            else:
                logger.error("Parallel formatting failed, using fallback")
                # Fallback to empty arguments
                arguments = []
                all_references = []
        except Exception as formatting_error:
            logger.error(f"❌ Error in progressive formatting: {str(formatting_error)}")
            logger.info("📊 Evidence collected but formatting failed, preserving data for fallback")
            
            # Create basic arguments from evidence without enhancement
            arguments = []
            all_references = []
            
            for evidence_item in all_evidence_results:
                position = evidence_item.get("position", "Unknown Position")
                evidence_list = evidence_item.get("evidence", [])
                
                # Create basic argument structure
                basic_arg = {
                    "stance": position,
                    "core_argument": f"Evidence-based analysis of {position}",
                    "supporting_arguments": [f"Evidence item {i+1}" for i, _ in enumerate(evidence_list[:3])],
                    "evidence_metadata": {
                        "supporting_evidence_count": len(evidence_list),
                        "refuting_evidence_count": 0,
                        "total_evidence_count": len(evidence_list),
                        "primary_sources": 0,
                        "secondary_sources": len(evidence_list)
                    }
                }
                arguments.append(basic_arg)
        
        # Create pipeline metadata using the new PipelineMetadata model
        pipeline_metadata = PipelineMetadata(
            positions_generated=original_position_count,  # Report original count
            evidence_searches=len(all_evidence_results),  # This will be limited to 2 during testing
            total_evidence_items=sum(item.get("evidence_count", 0) for item in all_evidence_results),
            total_evidence_cost=total_evidence_cost,
            evidence_structure_version="v2_with_stance_separation_TEST_LIMITED",  # Mark as test mode
            link_format="domain_based_clickable",
            citation_style="inline_domain_links"
        )
        
        # Create the final result using the new JuntoEvidenceResponse model
        # This structure now fully accommodates the Junto Evidence Finder's position/refute format
        # and is prepared for future UI enhancements that may want to display:
        # - Supporting vs. refuting evidence separately
        # - Primary vs. secondary source distinctions
        # - Evidence reasoning and stance information
        # - Clickable domain-based links: [nasa.gov] format that links to full URLs
        # 
        # Frontend Implementation Note:
        # - supporting_arguments contain text like "Quote text [nasa.gov]"
        # - The [domain.com] parts should be rendered as clickable links
        # - Use the references array to map domain names to full URLs
        # - Example: [nasa.gov] should link to https://nasa.gov/news/openai-2024
        
        # Create the response - arguments are already enhanced
        try:
            # Arguments are already dictionaries with enhancements
            result = {
                "arguments": arguments,  # Already enhanced with LLM
                "model": "junto-evidence-pipeline",
                "search_model": "junto-position-finder + junto-evidence-finder", 
                "references": [ref.dict() for ref in all_references],
                "pipeline_metadata": pipeline_metadata.dict()
            }
            logger.info("✅ Successfully created Junto Evidence response with parallel enhancements")
            
            # Debug: Log the structure of the first argument (remove in production)
            # if result.get("arguments") and len(result["arguments"]) > 0:
            #     first_arg = result["arguments"][0]
            #     logger.info(f"🔍 Debug first argument keys: {list(first_arg.keys())}")
            #     logger.info(f"🔍 Debug evidence_metadata: {first_arg.get('evidence_metadata')}")
            #     logger.info(f"🔍 Debug detailed_evidence keys: {list(first_arg.get('detailed_evidence', {}).keys()) if first_arg.get('detailed_evidence') else 'None'}")
            #     if first_arg.get('detailed_evidence'):
            #         detailed = first_arg['detailed_evidence']
            #         logger.info(f"🔍 Debug supporting count: {len(detailed.get('supporting', []))}")
            #         logger.info(f"🔍 Debug refuting count: {len(detailed.get('refuting', []))}")
            
        except Exception as validation_error:
            logger.error(f"❌ Pydantic validation failed for Junto Evidence response: {str(validation_error)}")
            # Fallback to dictionary format if validation fails
            result = {
                "arguments": [arg.dict() for arg in arguments],
                "model": "junto-evidence-pipeline",
                "search_model": "junto-position-finder + junto-evidence-finder",
                "references": [ref.dict() for ref in all_references],
                "pipeline_metadata": pipeline_metadata.dict()
            }
        
        # LLM enhancement already done in parallel during formatting
        # Add follow-up questions if not already present
        if LOCAL_LLM_AVAILABLE and 'follow_up_questions' not in result:
            try:
                from .llm_helpers import FollowUpQuestionGenerator
                follow_up_questions = await FollowUpQuestionGenerator.generate_follow_up_questions(
                    result.get('arguments', []), normalized_topic
                )
                if follow_up_questions:
                    result['follow_up_questions'] = follow_up_questions
                    logger.info("✅ Added follow-up questions to response")
            except Exception as e:
                logger.warning(f"Failed to generate follow-up questions: {e}")
        
        logger.info(f"🎉 Junto Evidence Pipeline completed: {len(arguments)} arguments, {len(all_references)} references, cost: ${total_evidence_cost}")
        
        # Create generation trace if available
        if trace:
            generation = trace.generation(
                name="junto_evidence_pipeline_complete",
                model="junto-evidence-pipeline",
                model_parameters={
                    "positions_count": len(generated_positions),
                    "evidence_searches": len(all_evidence_results),
                    "total_evidence_cost": total_evidence_cost
                },
                input=normalized_topic,
                output=result,
                metadata={
                    "pipeline": "junto_evidence",
                    "positions_generated": len(generated_positions),
                    "evidence_items_total": sum(item["evidence_count"] for item in all_evidence_results)
                }
            )
            
            # Flush Langfuse events
            try:
                langfuse.flush()
                logger.debug("Flushed Langfuse events for Junto evidence pipeline")
            except Exception as flush_error:
                logger.warning(f"Failed to flush Langfuse events: {str(flush_error)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Junto evidence pipeline: {str(e)}")
        
        # Track error in trace if available
        if trace:
            try:
                trace.update(
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "pipeline": "junto_evidence"
                    }
                )
            except Exception as trace_error:
                logger.warning(f"Failed to record error in trace: {str(trace_error)}")
        
        # Fallback to regular pipeline on error
        logger.info("Falling back to regular pipeline due to error")
        return complete(topic, diversity, num_stances, user_id, session_id, request_metadata)


def complete(topic: str, diversity: float, num_stances: int = 3, user_id: Optional[str] = None, session_id: Optional[str] = None, request_metadata: Optional[Dict[str, Any]] = None, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Get completion from OpenAI with structured output and caching."""
    
    # Pipeline selection: Check if we should use the Junto Evidence pipeline
    if USE_JUNTO_EVIDENCE_PIPELINE and junto_evidence_finder.enabled and junto_generator.enabled:
        logger.info(f"🔄 Routing to Junto Evidence Pipeline for topic: '{topic}'")
        try:
            # Run the async pipeline in a proper async context
            # Use nest_asyncio to handle existing event loops
            import nest_asyncio
            nest_asyncio.apply()
            
            # Create a new event loop for this execution
            try:
                loop = asyncio.get_running_loop()
                # If we have a running loop, use run_coroutine_threadsafe
                import threading
                import concurrent.futures
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            complete_with_junto_evidence_pipeline(topic, diversity, num_stances, user_id, session_id, request_metadata, progress_callback)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=600)  # 10 minute timeout
                    
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                return asyncio.run(
                    complete_with_junto_evidence_pipeline(topic, diversity, num_stances, user_id, session_id, request_metadata, progress_callback)
                )
        except Exception as pipeline_error:
            logger.error(f"❌ Junto Evidence Pipeline failed, falling back to standard pipeline: {str(pipeline_error)}")
            # Continue to standard pipeline below
    else:
        pipeline_reason = []
        if not USE_JUNTO_EVIDENCE_PIPELINE:
            pipeline_reason.append("USE_JUNTO_EVIDENCE_PIPELINE=false")
        if not junto_evidence_finder.enabled:
            pipeline_reason.append("evidence_finder_disabled")
        if not junto_generator.enabled:
            pipeline_reason.append("position_generator_disabled")
        logger.info(f"🔄 Using standard pipeline (Junto Position + GPT-4o Search) for topic: '{topic}' - Reason: {', '.join(pipeline_reason)}")
    
    # Send progress update if callback provided
    if progress_callback:
        progress_callback("🔍 Initializing query processing...")
    
    trace = None
    generation = None
    
    try:
        # Format query as question if local LLM is available
        if LOCAL_LLM_AVAILABLE:
            try:
                formatted_topic = QueryFormatter.format_query_as_question(topic)
                logger.info(f"📝 Formatted query: '{formatted_topic}' (original: '{topic}')")
                topic = formatted_topic
            except Exception as e:
                logger.warning(f"Failed to format query with LLM: {e}")
        
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
            "prompt_manager": PROMPT_MANAGER,
            "junto_enabled": junto_generator.enabled,
            "junto_api_url": JUNTO_API_URL if junto_generator.enabled else None
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
        
        # Note: The new prompt format doesn't need stance count addition - it handles this internally
        
        # Generate positions using Junto API if enabled (moved before cache check)
        generated_positions = []
        if junto_generator.enabled:
            try:
                if progress_callback:
                    progress_callback("🎯 Identifying diverse positions...")
                
                logger.info(f"🎯 Generating positions for topic: '{normalized_topic}'")
                # Fix asyncio event loop issue
                import nest_asyncio
                nest_asyncio.apply()
                
                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures
                    
                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                junto_generator.generate_positions(normalized_topic, trace)
                            )
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        junto_result = future.result(timeout=300)
                        
                except RuntimeError:
                    junto_result = asyncio.run(
                        junto_generator.generate_positions(normalized_topic, trace)
                    )
                if junto_result is None:
                    generated_positions = []
                elif isinstance(junto_result, tuple):
                    # Extract positions from tuple (positions, cost_estimate)
                    generated_positions = junto_result[0] if isinstance(junto_result[0], list) else []
                elif isinstance(junto_result, list):
                    generated_positions = junto_result
                else:
                    generated_positions = []
                logger.info(f"✅ Generated {len(generated_positions)} positions from Junto API")
                if generated_positions:
                    logger.info(f"📋 Positions: {json.dumps(generated_positions, indent=2)}")
                    if progress_callback:
                        progress_callback(f"✅ Found {len(generated_positions)} diverse positions")
            except Exception as e:
                logger.error(f"❌ Error in position generation: {str(e)}")
                generated_positions = []
                if progress_callback:
                    progress_callback("⚠️ Fallback: Using default position generation")

        # Update system content to include generated positions in the expected format
        if generated_positions:
            # Log the positions we're about to use
            logger.info(f"📋 Processing {len(generated_positions)} positions from Junto API")
            logger.info(f"Raw positions: {json.dumps(generated_positions, indent=2)}")
            
            # Format positions as a list for the prompt
            if isinstance(generated_positions, list) and all(isinstance(p, str) for p in generated_positions):
                # Positions are already a list of strings, use them directly
                positions_list = generated_positions
            else:
                # Convert to list of strings if needed
                positions_list = [str(p) for p in generated_positions]
            
            logger.info(f"Formatted positions list: {positions_list}")
            system_content = system_content + f"\n\nInput: {json.dumps(positions_list)}"
            logger.info(f"📝 Updated system prompt with {len(positions_list)} generated positions")
        else:
            # If no positions generated, we should still provide the input format but with topic directly
            system_content = system_content + f"\n\nInput: [\"{normalized_topic}\"]"
            logger.info(f"📝 No positions generated, using topic directly as input")
    
        # Generate cache key using the complete system prompt (including positions)
        try:
            cache_key = generate_cache_key(normalized_topic, diversity, num_stances, system_content)
            logger.info(f"🔑 Generated cache key: {cache_key}")
        except Exception as e:
            logger.error(f"Error generating cache key: {str(e)}")
            raise

        # Try to get cached response
        cached_result = get_cached_response_sync(cache_key)
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
        
        # Note: JSON formatting will be handled in step 2 by gpt-4o-mini
        
        # Two-step process for web search with citations
        try:
            # Step 1: Use gpt-4o-search-preview to get raw content with web search
            start_time = time.time()
            
            # Prepare messages for the first call
            search_messages = [
                {"role": "system", "content": system_content},  # Original system content without JSON instruction
                {"role": "user", "content": normalized_topic}
            ]
            
            # Create generation in trace if available (after we have the messages defined)
            if trace:
                generation = trace.generation(
                    name="allstances_two_step_output",
                    model="gpt-4o-search-preview",
                    model_parameters={
                        "max_tokens": None,
                        "top_p": 1.0,
                        "frequency_penalty": 0,
                        "presence_penalty": 0,
                        "stances": num_stances,
                        "two_step_process": True
                    },
                    input=search_messages,
                    metadata={
                        "cache_key": cache_key,
                        "positions_generated": len(generated_positions),
                        "positions_included": len(generated_positions) > 0,
                        "junto_enabled": junto_generator.enabled
                    }
                )
                logger.info(f"Created two-step generation: {generation.id if generation else 'None'}")
            
            # Use the regular OpenAI client (not instructor-patched) for gpt-4o-search-preview
            raw_client = OpenAI(
                api_key=openai_key,
                timeout=300.0
            )
            
            # First API call with gpt-4o-search-preview
            if progress_callback:
                progress_callback("🌐 Searching for evidence across multiple sources...")
            
            search_response = raw_client.chat.completions.create(
                model='gpt-4o-search-preview',
                messages=search_messages,
                stream=False
            )
            
            search_end_time = time.time()
            search_latency_ms = int((search_end_time - start_time) * 1000)
            
            # Extract content and search results
            raw_text_content = search_response.choices[0].message.content
            
            # Extract URLs from the content (they contain utm_source=openai)
            import re
            url_pattern = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
            found_urls = re.findall(url_pattern, raw_text_content)
            
            # Create structured sources from found URLs
            sources = []
            for i, (title, url) in enumerate(found_urls, 1):
                sources.append({
                    "id": i,
                    "title": title,
                    "url": url.split('?utm_source=')[0] if '?utm_source=' in url else url
                })
            
            # Log search results
            logger.info(f"Step 1 complete: Found {len(sources)} sources from web search")
            
            # Calculate cost for first call
            search_cost = 0.0
            if hasattr(search_response, 'usage') and search_response.usage:
                search_usage = search_response.usage
                search_cost = calculate_openai_cost(
                    search_usage.prompt_tokens, 
                    search_usage.completion_tokens, 
                    "gpt-4o-search-preview"
                )
                
                # Trace the first call with Langfuse if available
                if generation:
                    generation.update(
                        name="web_search_step",
                        output=raw_text_content,
                        usage={
                            "input_tokens": search_usage.prompt_tokens,
                            "output_tokens": search_usage.completion_tokens,
                            "total_tokens": search_usage.total_tokens
                        },
                        metadata={
                            "step": "web_search",
                            "sources_found": len(sources),
                            "cost_usd": search_cost
                        }
                    )
            
            # Step 2: Use gpt-4o-mini to format with citations
            citation_system_prompt = f"""{system_content}

IMPORTANT ADDITIONAL INSTRUCTIONS:
1. You will receive raw text content and a list of sources
2. Rewrite the content with inline citations using [1], [2], etc. format
3. Each citation number should correspond to the source ID
4. Include a "References" section at the end with all cited sources
5. Respond with valid JSON in the exact format specified below

You must respond with valid JSON in the following exact format:
{{
  "arguments": [
    {{
      "stance": "Stance name here",
      "core_argument": "Main argument with inline citations [1] where relevant",
      "supporting_arguments": [
        "Supporting point 1 with citations [2]",
        "Supporting point 2 with citations [3]", 
        "Supporting point 3"
      ]
    }}
  ],
  "references": [
    {{
      "id": 1,
      "title": "Source Title",
      "url": "https://example.com"
    }}
  ]
}}"""
            
            # Prepare content for citation formatting
            citation_input = f"""Raw content from web search:
{raw_text_content}

Available sources:
{json.dumps(sources, indent=2)}

Please reformat the content with proper inline citations and include a references section."""
            
            citation_messages = [
                {"role": "system", "content": citation_system_prompt},
                {"role": "user", "content": citation_input}
            ]
            
            # Second API call with gpt-4o-mini
            if progress_callback:
                progress_callback("🔄 Analyzing evidence and formatting arguments...")
            
            citation_start_time = time.time()
            
            citation_response = raw_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=citation_messages,
                temperature=diversity,
                stream=False
            )
            
            end_time = time.time()
            citation_latency_ms = int((end_time - citation_start_time) * 1000)
            total_latency_ms = int((end_time - start_time) * 1000)
            
            # Parse the formatted response
            try:
                response_content = citation_response.choices[0].message.content
                
                # Parse JSON content
                if isinstance(response_content, str):
                    parsed_content = json.loads(response_content)
                else:
                    parsed_content = response_content
                
                # Extract arguments and references
                arguments = parsed_content.get("arguments", [])
                references = parsed_content.get("references", [])
                
                # Add references to each argument for frontend display
                for arg in arguments:
                    arg["references"] = references
                
                result = {
                    "arguments": arguments,
                    "model": "gpt-4o-mini",
                    "search_model": "gpt-4o-search-preview", 
                    "references": references
                }
                
                logger.info(f"Step 2 complete: Successfully formatted with {len(references)} references")
                
            except json.JSONDecodeError as json_error:
                logger.error(f"Error parsing JSON response from gpt-4o-mini: {str(json_error)}")
                logger.error(f"Response content: {response_content}")
                result = {"arguments": [], "model": "gpt-4o-mini", "search_model": "gpt-4o-search-preview", "references": []}
            
            # Calculate cost for second call and combine
            citation_cost = 0.0
            citation_usage_data = None
            if hasattr(citation_response, 'usage') and citation_response.usage:
                citation_usage = citation_response.usage
                citation_cost = calculate_openai_cost(
                    citation_usage.prompt_tokens, 
                    citation_usage.completion_tokens, 
                    "gpt-4o-mini"
                )
                citation_usage_data = {
                    "input_tokens": citation_usage.prompt_tokens,
                    "output_tokens": citation_usage.completion_tokens,
                    "total_tokens": citation_usage.total_tokens
                }
                
                # Trace the second call with Langfuse if available
                if generation:
                    generation.update(
                        name="citation_formatting_step",
                        output=result,
                        usage=citation_usage_data,
                        metadata={
                            "step": "citation_formatting",
                            "latency_ms": citation_latency_ms,
                            "cost_usd": citation_cost
                        }
                    )
            
            # Combine usage data from both calls
            total_cost = search_cost + citation_cost
            combined_usage_data = None
            
            if search_response.usage and citation_response.usage:
                combined_usage_data = {
                    "search_step": {
                        "input_tokens": search_response.usage.prompt_tokens,
                        "output_tokens": search_response.usage.completion_tokens,
                        "total_tokens": search_response.usage.total_tokens,
                        "cost_usd": search_cost
                    },
                    "citation_step": citation_usage_data,
                    "total": {
                        "input_tokens": search_response.usage.prompt_tokens + citation_response.usage.prompt_tokens,
                        "output_tokens": search_response.usage.completion_tokens + citation_response.usage.completion_tokens,
                        "total_tokens": search_response.usage.total_tokens + citation_response.usage.total_tokens,
                        "cost_usd": total_cost
                    }
                }
            
            logger.info(f"Two-step process complete: Search cost ${search_cost}, Citation cost ${citation_cost}, Total cost ${total_cost}")
            
            # Use combined usage for tracking
            usage_data = combined_usage_data
            cost = total_cost
            
            # End generation with complete information
            if generation:
                generation_end_data = {
                    "output": result,
                    "end_time": end_time,
                    "metadata": {
                        "total_latency_ms": total_latency_ms,
                        "search_latency_ms": search_latency_ms,
                        "citation_latency_ms": citation_latency_ms,
                        "cache_miss": True,
                        "cost_usd": cost,
                        "two_step_process": True
                    }
                }
                
                if usage_data:
                    generation_end_data["usage"] = usage_data
                
                generation.end(**generation_end_data)
                logger.info(f"Ended two-step generation with total usage: {usage_data}")

            # Enhance result with local LLM if available
            # Support both 'arguments' (Junto format) and 'stances' (legacy format)
            has_stances = result.get('arguments') or result.get('stances')
            if LOCAL_LLM_AVAILABLE and has_stances:
                # Check if LLM services are actually available
                llm_services_available = llm_client.ollama_available or llm_client.vllm_available
                
                if llm_services_available:
                    try:
                        logger.info("🤖 Enhancing response with local LLM analysis...")
                        # CRITICAL FIX: Use sync version when already in thread context
                        # to avoid nested async/sync conflicts that crash the worker
                        enhanced_result = enhance_response_with_llm_sync(result, normalized_topic)
                        result = enhanced_result
                        logger.info("✅ Successfully enhanced response with local LLM")
                    except Exception as llm_error:
                        logger.error(f"Failed to enhance with local LLM: {llm_error}", exc_info=True)
                        # Continue with original result
                else:
                    logger.info("🔌 Local LLM services not available, skipping enhancement")
            
            # Cache the result
            cache_success = set_cached_response_sync(cache_key, result)
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
                        supporting_arguments=["An error occurred while processing your request", "Please try again later"]
                    ),
                    Stance(
                        stance="Technical Details",
                        supporting_arguments=["API Error Information", str(api_error)]
                    )
                ],
                model="gpt-4o-mini"
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
                    supporting_arguments=["A system error occurred", str(e)]
                ),
                Stance(
                    stance="Technical Details",
                    supporting_arguments=["Please try again later", "If the issue persists, contact support"]
                )
            ],
            model="gpt-4o-search-preview"
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
            "supporting_arguments": current_supporting_arguments
        })

    return {"arguments": arguments}



        
class Stance(BaseModel):
    stance: str = Field(description="The stance or perspective on the topic")
    supporting_arguments: List[str] = Field(
        description="Supporting evidence and arguments for this stance",
        min_items=1,
        max_items=10,
        alias="supporting_arguments"
    )
    # Local LLM enhancements
    core_argument_summary: Optional[str] = Field(
        default=None,
        description="LLM-generated 2-3 sentence summary of the position"
    )
    source_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="LLM-generated analysis of sources including distribution and trust scores"
    )

    class Config:
        populate_by_name = True

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "stance": self.stance,
            "supporting_arguments": self.supporting_arguments
        }
        if self.core_argument_summary:
            result["core_argument_summary"] = self.core_argument_summary
        if self.source_analysis:
            result["source_analysis"] = self.source_analysis
        return result

class LangfuseResponseArray(RootModel[List[Dict[str, List[str]]]]):
    """Model to handle the new Langfuse prompt response format - array of stance objects"""
    
    def to_argument_response(self) -> 'ArgumentResponse':
        """Convert Langfuse format to standard ArgumentResponse format"""
        stances = []
        
        for stance_obj in self.root:
            # Each stance_obj is a dictionary with stance as key and arguments array as value
            for stance_name, arguments in stance_obj.items():
                # Create a stance with all arguments as supporting_arguments
                if isinstance(arguments, list) and arguments:
                    stance = Stance(
                        stance=stance_name,
                        supporting_arguments=arguments
                    )
                    stances.append(stance)
        
        return ArgumentResponse(arguments=stances, model="gpt-4o-search-preview")

class ArgumentResponse(BaseModel):
    arguments: List[Stance] = Field(
        description="List of stances and their arguments",
        min_items=2,
        max_items=20  # Updated to allow up to 20 stances
    )
    model: str = Field(default="gpt-4", description="The model used for generating the response")
    # Local LLM enhancements
    follow_up_questions: Optional[List[str]] = Field(
        default=None,
        description="LLM-generated follow-up questions to broaden perspectives"
    )

    class Config:
        populate_by_name = True

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "arguments": [arg.to_dict() for arg in self.arguments],
            "model": self.model
        }
        if self.follow_up_questions:
            result["follow_up_questions"] = self.follow_up_questions
        return result

# Junto Evidence Pipeline Data Models
class EvidenceItem(BaseModel):
    """Individual evidence item from Junto Evidence Finder API"""
    quote: str = Field(description="The evidence quote text")
    citation_id: int = Field(description="Reference ID for citation linking")
    reasoning: str = Field(description="Explanation of how the evidence relates to the claim")
    stance: str = Field(description="Whether evidence supports or refutes", pattern="^(supports|refutes)$")
    url: str = Field(description="Source URL for the evidence")
    domain: str = Field(description="Extracted domain name for display")
    formatted: str = Field(description="Formatted text with clickable domain citation")

class EvidenceMetadata(BaseModel):
    """Metadata about evidence collection for a stance"""
    supporting_evidence_count: int = Field(description="Number of supporting evidence items found")
    refuting_evidence_count: int = Field(description="Number of refuting evidence items found") 
    total_evidence_count: int = Field(description="Total evidence items processed")
    primary_sources: int = Field(description="Number of primary source citations")
    secondary_sources: int = Field(description="Number of secondary source citations")

class DetailedEvidence(BaseModel):
    """Detailed evidence breakdown for potential future UI enhancements"""
    supporting: List[EvidenceItem] = Field(description="Supporting evidence items")
    refuting: List[EvidenceItem] = Field(description="Refuting evidence items")

class Reference(BaseModel):
    """Reference/citation with clickable domain-based format"""
    id: int = Field(description="Reference ID number")
    title: str = Field(description="Display title (domain name for clickable links)")
    url: str = Field(description="Full URL for the reference")
    source_type: str = Field(description="Primary or secondary source", pattern="^(primary|secondary)$")
    stance: str = Field(description="Whether reference supports or refutes", pattern="^(supports|refutes)$")
    domain: str = Field(description="Extracted domain name")

class JuntoEvidenceStance(BaseModel):
    """Enhanced stance model for Junto Evidence pipeline with evidence metadata"""
    stance: str = Field(description="The stance or perspective on the topic")
    supporting_arguments: List[str] = Field(
        description="Supporting evidence and arguments with clickable domain citations like [nasa.gov]",
        min_items=1,
        max_items=10
    )
    references: List[Reference] = Field(description="All references for clickable link mapping")
    evidence_metadata: EvidenceMetadata = Field(description="Metadata about evidence collection")
    detailed_evidence: DetailedEvidence = Field(description="Detailed evidence breakdown")
    # Local LLM enhancements
    core_argument_summary: Optional[str] = Field(
        default=None,
        description="LLM-generated 2-3 sentence summary of the position"
    )
    source_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="LLM-generated analysis of sources including distribution and trust scores"
    )

class PipelineMetadata(BaseModel):
    """Metadata about the Junto Evidence pipeline execution"""
    positions_generated: int = Field(description="Number of positions generated by Junto Position API")
    evidence_searches: int = Field(description="Number of evidence searches performed")
    total_evidence_items: int = Field(description="Total evidence items collected")
    total_evidence_cost: float = Field(description="Total cost estimate from Junto Evidence API")
    evidence_structure_version: str = Field(description="Version marker for compatibility")
    link_format: str = Field(description="Format used for links (domain_based_clickable)")
    citation_style: str = Field(description="Citation style used (inline_domain_links)")

class JuntoEvidenceResponse(BaseModel):
    """Response model for the Junto Evidence pipeline with enhanced evidence structure"""
    arguments: List[JuntoEvidenceStance] = Field(
        description="List of stances with evidence-based arguments and clickable citations",
        min_items=1,
        max_items=20
    )
    model: str = Field(default="junto-evidence-pipeline", description="Pipeline identifier")
    search_model: str = Field(
        default="junto-position-finder + junto-evidence-finder", 
        description="Combined model description"
    )
    references: List[Reference] = Field(description="All references for the response")
    pipeline_metadata: PipelineMetadata = Field(description="Pipeline execution metadata")
    # Local LLM enhancements
    follow_up_questions: Optional[List[str]] = Field(
        default=None,
        description="LLM-generated follow-up questions to broaden perspectives"
    )

    class Config:
        populate_by_name = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API response"""
        return {
            "arguments": [arg.dict() for arg in self.arguments],
            "model": self.model,
            "search_model": self.search_model,
            "references": [ref.dict() for ref in self.references],
            "pipeline_metadata": self.pipeline_metadata.dict()
        }

# Add cache management endpoints
async def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the cache."""
    try:
        redis_client = get_redis_client()
        keys = redis_client.keys("allstances:response:*")
        stats = {
            "total_cached_entries": len(keys),
            "cache_size_bytes": sum(redis_client.memory_usage(key) or 0 for key in keys) if keys else 0,
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
        total_size = sum(redis_client.memory_usage(key) or 0 for key in keys) if keys else 0
        
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

# Clear cache on startup
try:
    logger.info("🧹 Clearing cache on startup...")
    clear_result = clear_cache()
    logger.info(f"Cache clear result: {clear_result}")
except Exception as e:
    logger.error(f"Failed to clear cache on startup: {str(e)}")

# =====================================
# LOCAL LLM NLP HELPER FUNCTIONS
# =====================================

def enhance_query_with_local_llm(query_text: str) -> Dict[str, Any]:
    """
    Enhance query analysis using local LLM for NLP tasks
    
    Args:
        query_text: The user's query text
        
    Returns:
        Dictionary with enhanced analysis data
    """
    if not LOCAL_LLM_AVAILABLE:
        logger.debug("Local LLM not available, skipping enhancement")
        return {"enhanced": False}
    
    try:
        logger.info(f"🧠 Enhancing query with local LLM: {query_text[:50]}...")
        
        # Get LLM service status
        service_status = llm_client.get_service_status()
        
        if not service_status.get('any_available'):
            logger.warning("No local LLM services available")
            return {"enhanced": False, "reason": "no_services_available"}
        
        enhancement_data = {
            "enhanced": True,
            "services_available": service_status,
            "original_query": query_text
        }
        
        # Extract keywords from the query
        keywords = llm_client.extract_keywords(query_text, max_keywords=5)
        if keywords:
            enhancement_data["keywords"] = keywords
            logger.info(f"📋 Extracted keywords: {keywords}")
        
        # Classify query sentiment
        sentiment = llm_client.classify_sentiment(query_text)
        enhancement_data["sentiment"] = sentiment
        logger.info(f"😊 Query sentiment: {sentiment}")
        
        # Generate query summary if long
        if len(query_text) > 100:
            summary = llm_client.summarize_text(query_text, max_length=50)
            enhancement_data["summary"] = summary
            logger.info(f"📝 Query summary: {summary}")
        
        return enhancement_data
        
    except Exception as e:
        logger.error(f"Error enhancing query with local LLM: {e}")
        return {"enhanced": False, "error": str(e)}

def analyze_argument_with_local_llm(argument_text: str, stance: str) -> Dict[str, Any]:
    """
    Analyze argument quality and characteristics using local LLM
    
    Args:
        argument_text: The argument text to analyze
        stance: The stance this argument supports
        
    Returns:
        Dictionary with analysis results
    """
    if not LOCAL_LLM_AVAILABLE:
        return {"analyzed": False}
    
    try:
        logger.debug(f"🔍 Analyzing argument for stance '{stance}': {argument_text[:50]}...")
        
        service_status = llm_client.get_service_status()
        if not service_status.get('any_available'):
            return {"analyzed": False, "reason": "no_services_available"}
        
        analysis = {
            "analyzed": True,
            "stance": stance,
            "original_text": argument_text[:200] + "..." if len(argument_text) > 200 else argument_text
        }
        
        # Extract key points from the argument
        key_points = llm_client.extract_keywords(argument_text, max_keywords=3)
        if key_points:
            analysis["key_points"] = key_points
        
        # Assess argument sentiment
        sentiment = llm_client.classify_sentiment(argument_text)
        analysis["sentiment"] = sentiment
        
        # Generate a brief summary
        if len(argument_text) > 150:
            summary = llm_client.summarize_text(argument_text, max_length=30)
            analysis["brief_summary"] = summary
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing argument with local LLM: {e}")
        return {"analyzed": False, "error": str(e)}

def get_local_llm_status() -> Dict[str, Any]:
    """
    Get detailed status of local LLM services
    
    Returns:
        Dictionary with service status information
    """
    if not LOCAL_LLM_AVAILABLE:
        return {
            "available": False,
            "reason": "llm_client_not_imported",
            "services": {}
        }
    
    try:
        status = llm_client.get_service_status()
        
        return {
            "available": status.get('any_available', False),
            "services": {
                "ollama": {
                    "available": status.get('ollama', False),
                    "url": llm_client.ollama_base_url,
                    "default_model": llm_client.ollama_default_model
                },
                "vllm": {
                    "available": status.get('vllm', False),
                    "url": llm_client.vllm_base_url,
                    "model": llm_client.vllm_model_name
                }
            },
            "last_check": llm_client.last_health_check
        }
        
    except Exception as e:
        logger.error(f"Error getting local LLM status: {e}")
        return {
            "available": False,
            "error": str(e),
            "services": {}
        }

# =====================================
# EXAMPLE USAGE IN QUERY PROCESSING
# =====================================

def process_query_with_llm_enhancement(query_text: str, diversity_score: float, num_stances: int):
    """
    Example function showing how to integrate local LLM analysis into query processing
    This is a demonstration function - integrate the concepts into your main processing pipeline
    """
    logger.info(f"🚀 Processing query with LLM enhancement: {query_text}")
    
    # Step 1: Enhance query with local LLM analysis
    enhancement = enhance_query_with_local_llm(query_text)
    
    if enhancement.get("enhanced"):
        logger.info(f"✨ Query enhanced successfully:")
        logger.info(f"   Keywords: {enhancement.get('keywords', [])}")
        logger.info(f"   Sentiment: {enhancement.get('sentiment', 'unknown')}")
        logger.info(f"   Summary: {enhancement.get('summary', 'N/A')}")
    
    # Step 2: Your existing query processing logic would go here
    # For example: call OpenAI GPT-4, process with Junto Evidence, etc.
    
    # Step 3: Analyze generated arguments with local LLM
    # This would be done for each generated argument
    example_argument = "This is an example argument text that would be analyzed..."
    argument_analysis = analyze_argument_with_local_llm(example_argument, "Example Stance")
    
    if argument_analysis.get("analyzed"):
        logger.info(f"🔍 Argument analysis complete:")
        logger.info(f"   Key points: {argument_analysis.get('key_points', [])}")
        logger.info(f"   Sentiment: {argument_analysis.get('sentiment', 'unknown')}")
    
    return {
        "query_enhancement": enhancement,
        "argument_analysis": argument_analysis,
        "llm_status": get_local_llm_status()
    }