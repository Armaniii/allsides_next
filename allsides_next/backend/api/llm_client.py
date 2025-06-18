"""
Local LLM Client for accessing Ollama and vLLM services
Supports both Ollama API and OpenAI-compatible vLLM API for NLP tasks
"""

import os
import logging
import requests
import time
from typing import Dict, List, Optional, Union
from openai import OpenAI
from requests.exceptions import RequestException, ConnectionError, Timeout

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Unified client for accessing local LLM services (Ollama and vLLM)
    Provides fallback mechanisms and health checking
    """
    
    def __init__(self):
        # Ollama configuration
        self.ollama_base_url = os.getenv('OLLAMA_API_URL', 'http://ollama:11434/api')
        self.ollama_host = os.getenv('OLLAMA_HOST', 'ollama:11434')
        
        # vLLM configuration  
        self.vllm_base_url = os.getenv('VLLM_API_URL', 'http://vllm:8000/v1')
        
        # OpenAI client for vLLM (OpenAI-compatible API)
        self.vllm_client = OpenAI(
            base_url=self.vllm_base_url,
            api_key="dummy-key",  # vLLM doesn't require a real API key
            timeout=180.0  # Allow time for text generation
        )
        
        # Timeout Strategy:
        # - Health checks: (3, 5-10) seconds - quick connection test, short read
        # - Model pulls: (5, 300) seconds - allow connection, long download time
        # - Text generation: (5, 180) seconds - quick connection check, allow time for CPU inference
        
        # Service availability tracking
        self.ollama_available = False
        self.vllm_available = False
        self.last_health_check = 0
        self.health_check_interval = 60  # seconds
        
        # Default models
        self.ollama_default_model = "gemma3:1b-it-qat"  # Small model for CPU
        self.vllm_model_name = "dialogpt-small"
        
        logger.info(f"Initialized LLMClient with Ollama: {self.ollama_base_url}, vLLM: {self.vllm_base_url}")
        
        # Initial health check
        self._check_service_health()
    
    def _check_service_health(self) -> None:
        """Check availability of both Ollama and vLLM services"""
        current_time = time.time()
        
        # Skip if checked recently
        if current_time - self.last_health_check < self.health_check_interval:
            return
            
        self.last_health_check = current_time
        
        # Check Ollama
        try:
            response = requests.get(f"http://{self.ollama_host}/api/tags", timeout=(3, 5))
            self.ollama_available = response.status_code == 200
            if self.ollama_available:
                logger.info("✅ Ollama service is available")
            else:
                logger.warning(f"⚠️ Ollama service returned status {response.status_code}")
        except Exception as e:
            self.ollama_available = False
            logger.warning(f"❌ Ollama service unavailable: {e}")
        
        # Check vLLM
        try:
            response = requests.get(f"{self.vllm_base_url}/models", timeout=(3, 10))
            self.vllm_available = response.status_code == 200
            if self.vllm_available:
                logger.info("✅ vLLM service is available")
            else:
                logger.warning(f"⚠️ vLLM service returned status {response.status_code}")
        except Exception as e:
            self.vllm_available = False
            logger.warning(f"❌ vLLM service unavailable: {e}")
    
    def ensure_ollama_model(self, model_name: str) -> bool:
        """Ensure the specified model is available in Ollama"""
        if not self.ollama_available:
            return False
            
        try:
            # Check if model exists
            response = requests.get(f"http://{self.ollama_host}/api/tags", timeout=(3, 10))
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                if model_name in model_names:
                    logger.info(f"✅ Model {model_name} already available")
                    return True
                
                # Pull model if not available
                logger.info(f"📥 Pulling model {model_name}...")
                pull_response = requests.post(
                    f"http://{self.ollama_host}/api/pull",
                    json={"name": model_name},
                    timeout=(5, 300)  # 5 seconds to connect, 5 minutes for model download
                )
                
                if pull_response.status_code == 200:
                    logger.info(f"✅ Successfully pulled model {model_name}")
                    return True
                else:
                    logger.error(f"❌ Failed to pull model {model_name}: {pull_response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error ensuring Ollama model {model_name}: {e}")
            return False
            
        return False
    
    def generate_json_with_ollama(self, prompt: str, model: str = None, max_tokens: int = 256) -> Optional[str]:
        """Generate structured JSON using Ollama with format enforcement"""
        self._check_service_health()
        
        if not self.ollama_available:
            logger.warning("Ollama service not available")
            return None
            
        model = model or self.ollama_default_model
        
        # Ensure model is available
        if not self.ensure_ollama_model(model):
            logger.error(f"Failed to ensure model {model} is available")
            return None
        
        # Performance monitoring
        start_time = time.time()
        prompt_length = len(prompt)
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",  # Enforce JSON output format
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            logger.info(f"🚀 Ollama JSON request starting: {prompt_length} chars, max_tokens={max_tokens}")
            
            response = requests.post(
                f"http://{self.ollama_host}/api/generate",
                json=payload,
                timeout=(5, 180)  # (connect_timeout, read_timeout) - quick connection check, allow time for generation
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                response_length = len(response_text)
                
                # Log performance with severity based on duration
                if duration > 30:
                    logger.warning(f"🐌 SLOW Ollama JSON response: {duration:.1f}s ({prompt_length}→{response_length} chars)")
                elif duration > 15:
                    logger.info(f"⚠️  Ollama JSON response: {duration:.1f}s ({prompt_length}→{response_length} chars)")
                else:
                    logger.info(f"✅ Ollama JSON response: {duration:.1f}s ({prompt_length}→{response_length} chars)")
                
                return response_text
            else:
                logger.error(f"❌ Ollama JSON generation failed after {duration:.1f}s: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"💥 Ollama JSON error after {duration:.1f}s: {e}")
            return None

    def generate_with_ollama(self, prompt: str, model: str = None, max_tokens: int = 256) -> Optional[str]:
        """Generate text using Ollama with performance monitoring"""
        self._check_service_health()
        
        if not self.ollama_available:
            logger.warning("Ollama service not available")
            return None
            
        model = model or self.ollama_default_model
        
        # Ensure model is available
        if not self.ensure_ollama_model(model):
            logger.error(f"Failed to ensure model {model} is available")
            return None
        
        # Performance monitoring
        start_time = time.time()
        prompt_length = len(prompt)
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            logger.info(f"🚀 Ollama request starting: {prompt_length} chars, max_tokens={max_tokens}")
            
            response = requests.post(
                f"http://{self.ollama_host}/api/generate",
                json=payload,
                timeout=(5, 180)  # (connect_timeout, read_timeout) - quick connection check, allow time for generation
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                response_length = len(response_text)
                
                # Log performance with severity based on duration
                if duration > 30:
                    logger.warning(f"🐌 SLOW Ollama response: {duration:.1f}s ({prompt_length}→{response_length} chars)")
                elif duration > 15:
                    logger.info(f"⚠️  Ollama response: {duration:.1f}s ({prompt_length}→{response_length} chars)")
                else:
                    logger.info(f"✅ Ollama response: {duration:.1f}s ({prompt_length}→{response_length} chars)")
                
                return response_text
            else:
                logger.error(f"❌ Ollama generation failed after {duration:.1f}s: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"💥 Ollama error after {duration:.1f}s: {e}")
            return None
    
    def generate_with_vllm(self, prompt: str, max_tokens: int = 256) -> Optional[str]:
        """Generate text using vLLM via OpenAI-compatible API"""
        self._check_service_health()
        
        if not self.vllm_available:
            logger.warning("vLLM service not available")
            return None
        
        try:
            response = self.vllm_client.completions.create(
                model=self.vllm_model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].text.strip()
            else:
                logger.warning("vLLM returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"Error generating with vLLM: {e}")
            return None
    
    def chat_with_vllm(self, messages: List[Dict[str, str]], max_tokens: int = 256) -> Optional[str]:
        """Chat completion using vLLM via OpenAI-compatible API"""
        self._check_service_health()
        
        if not self.vllm_available:
            logger.warning("vLLM service not available for chat")
            return None
        
        try:
            response = self.vllm_client.chat.completions.create(
                model=self.vllm_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                logger.warning("vLLM chat returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"Error with vLLM chat: {e}")
            return None
    
    def generate_json(self, prompt: str, max_tokens: int = 256, prefer_service: str = "ollama") -> Optional[str]:
        """
        Generate structured JSON with format enforcement
        
        Args:
            prompt: Input text prompt (should specify JSON requirements)
            max_tokens: Maximum tokens to generate
            prefer_service: Preferred service ("ollama" or "vllm")
        
        Returns:
            Generated JSON text or None if services fail
        """
        self._check_service_health()
        
        # Early return if no services are available
        if not self.ollama_available and not self.vllm_available:
            logger.warning("No LLM services available for JSON generation")
            return None
        
        # Always prioritize Ollama for JSON generation (has format enforcement)
        if self.ollama_available:
            logger.debug("Using Ollama for JSON generation")
            return self.generate_json_with_ollama(prompt, max_tokens=max_tokens)
        
        # Fallback to regular text generation with vLLM if available
        if self.vllm_available:
            logger.warning("Using vLLM fallback for JSON (no format enforcement)")
            return self.generate_with_vllm(prompt, max_tokens=max_tokens)
        
        logger.error("No LLM services available for JSON generation")
        return None

    def generate_text(self, prompt: str, max_tokens: int = 256, prefer_service: str = "ollama") -> Optional[str]:
        """
        Generate text with automatic fallback between services
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            prefer_service: Preferred service ("ollama" or "vllm")
        
        Returns:
            Generated text or None if both services fail
        """
        self._check_service_health()
        
        # Early return if no services are available
        if not self.ollama_available and not self.vllm_available:
            logger.warning("No LLM services available for text generation")
            return None
        
        # Always prioritize Ollama - it's optimized for our CPU-only system
        if self.ollama_available:
            logger.debug("Using Ollama for text generation")
            return self.generate_with_ollama(prompt, max_tokens=max_tokens)
        
        if prefer_service == "ollama" and self.ollama_available:
            result = self.generate_with_ollama(prompt, max_tokens=max_tokens)
            if result:
                return result
            # Fallback to vLLM
            logger.info("Ollama failed, trying vLLM fallback...")
            return self.generate_with_vllm(prompt, max_tokens=max_tokens)
            
        elif prefer_service == "vllm" and self.vllm_available:
            result = self.generate_with_vllm(prompt, max_tokens=max_tokens)
            if result:
                return result
            # Fallback to Ollama
            logger.info("vLLM failed, trying Ollama fallback...")
            return self.generate_with_ollama(prompt, max_tokens=max_tokens)
        
        # Try any available service
        if self.ollama_available:
            return self.generate_with_ollama(prompt, max_tokens=max_tokens)
        elif self.vllm_available:
            return self.generate_with_vllm(prompt, max_tokens=max_tokens)
        
        logger.error("No LLM services available")
        return None
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text using local LLM"""
        prompt = f"""Extract the {max_keywords} most important keywords from the following text. Return only the keywords separated by commas, no other text.

Text: {text[:500]}

Keywords:"""
        
        result = self.generate_text(prompt, max_tokens=50, prefer_service="ollama")
        if result:
            keywords = [kw.strip() for kw in result.split(',') if kw.strip()]
            return keywords[:max_keywords]
        
        return []
    
    def classify_sentiment(self, text: str) -> str:
        """Classify sentiment of text (positive, negative, neutral)"""
        prompt = f"""Classify the sentiment of the following text. Respond with only one word: positive, negative, or neutral.

Text: {text[:300]}

Sentiment:"""
        
        result = self.generate_text(prompt, max_tokens=10, prefer_service="vllm")
        if result:
            sentiment = result.lower().strip()
            if sentiment in ['positive', 'negative', 'neutral']:
                return sentiment
        
        return 'neutral'
    
    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Generate a summary of the given text"""
        prompt = f"""Summarize the following text in no more than {max_length} words. Be concise and capture the main points.

Text: {text[:1000]}

Summary:"""
        
        result = self.generate_text(prompt, max_tokens=max_length, prefer_service="ollama")
        return result or "Summary unavailable"
    
    def get_service_status(self) -> Dict[str, bool]:
        """Get current status of all LLM services"""
        self._check_service_health()
        return {
            'ollama': self.ollama_available,
            'vllm': self.vllm_available,
            'any_available': self.ollama_available or self.vllm_available
        }


# Global instance
llm_client = LLMClient()