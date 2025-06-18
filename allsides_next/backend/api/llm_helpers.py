"""
Local LLM Helper Functions for AllSides Next
Provides synchronous functions for enhancing query responses with local LLM
"""

import json
import logging
import os
import time
import fcntl
from functools import wraps
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import contextmanager
import re

from .llm_client import llm_client

logger = logging.getLogger(__name__)


# Timeout handling is now done at the network level in llm_client.py
# This eliminates thread-safety issues with signal-based timeouts


class OllamaRateLimiter:
    """
    Process-safe rate limiter for Ollama API calls using file locks.
    Ensures only N concurrent requests across all WSGI workers.
    """
    
    def __init__(self, max_concurrent=2, lock_dir="/tmp/allsides_ollama_locks"):
        self.max_concurrent = max_concurrent
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(exist_ok=True)
        self.lock_files = [
            self.lock_dir / f"ollama_lock_{i}.lock" 
            for i in range(max_concurrent)
        ]
        # Ensure lock files exist
        for lock_file in self.lock_files:
            lock_file.touch()
        
    @contextmanager
    def acquire(self, timeout=30):
        """Acquire a lock for Ollama access. Use as a context manager."""
        fp = self._acquire_lock_file(timeout)
        try:
            yield  # This is where the user's code runs
        finally:
            self._release_lock_file(fp)
    
    def _acquire_lock_file(self, timeout):
        """Internal method to acquire a lock file"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            for lock_file in self.lock_files:
                try:
                    fp = open(lock_file, 'r+')
                    fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # Lock acquired! Write metadata and return the file pointer
                    fp.seek(0)
                    fp.write(f"{os.getpid()}:{time.time()}\n")
                    fp.truncate()
                    fp.flush()
                    return fp
                
                except (IOError, BlockingIOError):
                    # Lock is held, close our handle and try the next one
                    fp.close()
                    logger.debug(f"Lock {lock_file.name} is held by another process")
                    continue  # Try next lock file
                
                except Exception as e:
                    # Clean up on unexpected error
                    if 'fp' in locals():
                        fp.close()
                    logger.error(f"Unexpected error acquiring lock {lock_file}: {e}")
                    continue
            
            # Brief sleep before retry
            time.sleep(0.1)
        
        raise Exception(f"Could not acquire Ollama lock within {timeout} seconds")
    
    def _release_lock_file(self, fp):
        """Internal method to release a lock file"""
        if not fp or fp.closed:
            return
        try:
            fp.seek(0)
            fp.truncate()
            fcntl.flock(fp, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")
        finally:
            fp.close()
    
    # Backward compatibility: keep old acquire/release methods
    def acquire_old(self, timeout=30):
        """Legacy acquire method for backward compatibility"""
        return self._acquire_lock_file(timeout)
    
    def release_old(self, fp):
        """Legacy release method for backward compatibility"""  
        self._release_lock_file(fp)


# Global rate limiter instance (configured from environment)
OLLAMA_MAX_CONCURRENT = int(os.environ.get('OLLAMA_MAX_CONCURRENT', '2'))
OLLAMA_LOCK_DIR = os.environ.get('OLLAMA_LOCK_DIR', '/tmp/allsides_ollama_locks')
ollama_rate_limiter = OllamaRateLimiter(max_concurrent=OLLAMA_MAX_CONCURRENT, lock_dir=OLLAMA_LOCK_DIR)


class QueryFormatter:
    """Ensures user queries are formatted as questions"""
    
    @staticmethod
    def format_query_as_question(query: str) -> str:
        """
        Formats a query as a question if it's not already one.
        Relies on network timeouts in llm_client for safety.
        """
        query = query.strip()
        
        # Check if it's already a question
        if query.endswith('?') or any(query.lower().startswith(q) for q in 
                                     ['what', 'why', 'how', 'when', 'where', 'who', 
                                      'is', 'are', 'can', 'should', 'does', 'do']):
            return query
        
        # Use local LLM to transform topic into question
        prompt = f"""Transform this topic into a general, balanced question. 
If it's already a question, return it as-is.
Topic: {query}

Return ONLY the question, nothing else.
Examples:
- "abortion" -> "What are the different perspectives on abortion?"
- "climate change" -> "How should society address climate change?"
- "gun control" -> "What are the arguments for and against gun control?"

Question:"""
        
        try:
            with ollama_rate_limiter.acquire():
                result = llm_client.generate_text(prompt, max_tokens=100, prefer_service="ollama")
                if result:
                    # Clean up the response
                    question = result.strip().strip('"').strip()
                    if not question.endswith('?'):
                        question += '?'
                    return question
        except Exception as e:
            logger.error(f"Error formatting query: {e}")
        
        # Fallback: simple question formation
        return f"What are the different perspectives on {query}?"


class FollowUpQuestionGenerator:
    """Generates follow-up questions based on positions"""
    
    @staticmethod
    def generate_follow_up_questions(positions: List[Dict], original_query: str) -> List[str]:
        """
        Generates 4-5 follow-up questions based on the positions.
        Relies on network timeouts in llm_client for safety.
        """
        # Extract key arguments from positions
        all_arguments = []
        for position in positions:
            all_arguments.append(position.get('name', ''))
            all_arguments.append(position.get('argument', ''))
            # Handle both string and dict formats for supporting arguments
            for arg in position.get('supporting_arguments', []):
                if isinstance(arg, str):
                    all_arguments.append(arg)
                elif isinstance(arg, dict):
                    all_arguments.append(arg.get('argument', ''))
                else:
                    all_arguments.append(str(arg))
        
        arguments_text = ' '.join(all_arguments[:1000])  # Limit context size
        
        prompt = f"""Based on this discussion about "{original_query}", generate exactly 4 insightful follow-up questions.

Key arguments discussed:
{arguments_text}

Generate questions that:
1. Explore specific aspects mentioned in the arguments
2. Address peripheral but important related topics
3. Encourage deeper understanding of different perspectives
4. Are diverse and thought-provoking

Output format: Return ONLY a JSON array containing 4 strings. Do not use object keys. Each string should be a complete question ending with a question mark.

Example: ["What are the economic implications?", "How might this affect different demographics?", "What are the long-term consequences?", "What alternative approaches exist?"]
"""
        
        try:
            with ollama_rate_limiter.acquire():
                result = llm_client.generate_json(prompt, max_tokens=600, prefer_service="ollama")
            logger.info(f"LLM JSON result for follow-up questions: {result[:100] if result else 'None'}...")
            if result:
                try:
                    # With format="json", the result should be clean JSON
                    parsed_result = json.loads(result.strip())
                    
                    if isinstance(parsed_result, list):
                        logger.info(f"✅ Successfully parsed {len(parsed_result)} questions as array with structured output")
                        return parsed_result[:4]  # Ensure max 4 questions
                    elif isinstance(parsed_result, dict):
                        # Convert object to array (common LLM behavior)
                        questions = list(parsed_result.values())
                        logger.info(f"✅ Converted dict to array: {len(questions)} questions with structured output")
                        return questions[:4]
                    else:
                        logger.warning(f"Unexpected JSON type {type(parsed_result)}: {parsed_result}")
                        return []
                except json.JSONDecodeError as parse_error:
                    logger.error(f"JSON parsing error despite format enforcement: {parse_error}")
                    logger.error(f"Raw result: {result}")
                    
                    # Fallback: try to extract array from potentially wrapped response
                    try:
                        json_match = re.search(r'\[.*\]', result, re.DOTALL)
                        if json_match:
                            logger.info("Attempting fallback regex extraction")
                            questions = json.loads(json_match.group())
                            logger.info(f"Fallback parsing successful: {len(questions)} questions")
                            return questions[:4]
                    except Exception as fallback_error:
                        logger.error(f"Fallback parsing also failed: {fallback_error}")
                    
                    return []
            else:
                logger.warning("LLM returned empty/null result")
        except Exception as network_error:
            logger.error(f"Network error in follow-up question generation: {network_error}")
        
        return []


class CoreArgumentSummarizer:
    """Generates overarching summaries for positions"""
    
    @staticmethod
    def summarize_position(position: Dict) -> str:
        """
        Generates a 2-3 sentence summary of a position.
        Relies on network timeouts in llm_client for safety.
        """
        # Extract all arguments and evidence
        stance_name = position.get('name', '')
        main_argument = position.get('argument', '')
        supporting_args = position.get('supporting_arguments', [])
        
        # Build context
        context_parts = [f"Position: {stance_name}", f"Main argument: {main_argument}"]
        
        for i, arg in enumerate(supporting_args[:3]):  # Limit to first 3
            if isinstance(arg, str):
                arg_text = arg
            elif isinstance(arg, dict):
                arg_text = arg.get('argument', '')
            else:
                arg_text = str(arg)
            
            if arg_text:
                context_parts.append(f"Supporting point {i+1}: {arg_text}")
        
        context = '\n'.join(context_parts)
        
        prompt = f"""Synthesize this position into a compelling 2-3 sentence summary that captures the essence of all arguments.

{context}

Write a clear, persuasive summary that:
1. Integrates the main and supporting arguments
2. Is 2-3 sentences maximum
3. Captures the core reasoning
4. Is written from the perspective's viewpoint

Summary:"""
        
        try:
            with ollama_rate_limiter.acquire():
                result = llm_client.generate_text(prompt, max_tokens=150, prefer_service="ollama")
                if result:
                    summary = result.strip()
                    # Ensure it's not too long
                    sentences = summary.split('.')
                    if len(sentences) > 3:
                        summary = '. '.join(sentences[:3]) + '.'
                    return summary
        except Exception as network_error:
            logger.error(f"Network error in position summarization: {network_error}")
        
        # Fallback
        return main_argument


class DialecticalAnalyzer:
    """
    Placeholder class for backwards compatibility with main_v3.py imports.
    The advanced dialectical analysis was removed in favor of simpler LLM enhancements.
    """
    
    @staticmethod
    async def analyze_dialectical_profile(evidence_data: Dict, main_claim: str) -> Dict[str, Any]:
        """
        Placeholder method for backwards compatibility.
        Returns minimal analysis to maintain API compatibility.
        """
        return {
            "main_claim": main_claim,
            "dialectical_summary": "Analysis simplified for stability",
            "supporting_profile": {"count": 0},
            "refuting_profile": {"count": 0},
            "key_perspectives": []
        }


def _normalize_response_format(response_data: Dict) -> Dict:
    """
    Normalizes the response format to handle both 'arguments' (Junto) and 'stances' (legacy).
    Ensures consistent field mapping for frontend compatibility.
    """
    # Support both 'stances' (older format) and 'arguments' (Junto format)
    if 'arguments' in response_data and 'stances' not in response_data:
        response_data['stances'] = response_data['arguments']
    
    stances = response_data.get('stances', [])
    
    # Ensure each stance has required fields for frontend
    for stance in stances:
        # Frontend expects 'stance' but Junto API provides 'name'
        if 'name' in stance and 'stance' not in stance:
            stance['stance'] = stance['name']
        # Ensure we have a stance field (fallback to name or argument)
        elif 'stance' not in stance:
            stance['stance'] = stance.get('name', stance.get('argument', 'Unknown Position'))
    
    return response_data


def enhance_response_with_llm_sync(response_data: Dict, original_query: str) -> Dict:
    """
    Main synchronous function for LLM enhancement.
    Works reliably in WSGI environment with process-safe rate limiting.
    """
    try:
        # Normalize the response format
        response_data = _normalize_response_format(response_data)
        stances = response_data.get('stances', [])
        
        if not stances:
            logger.warning("No stances found to enhance")
            return response_data
        
        logger.info(f"Enhancing {len(stances)} stances with LLM summaries")
        
        # Process each stance with rate limiting
        for i, stance in enumerate(stances):
            try:
                # Generate core argument summary
                logger.info(f"Generating summary for stance {i+1}/{len(stances)}")
                summary = CoreArgumentSummarizer.summarize_position(stance)
                stance['core_argument_summary'] = summary
            except Exception as e:
                logger.error(f"Error generating core argument summary for stance {i}: {e}")
                # Use fallback
                stance['core_argument_summary'] = stance.get('argument', 'Summary unavailable')
        
        # Generate follow-up questions (single LLM call for all stances)
        try:
            logger.info("Generating follow-up questions")
            follow_up_questions = FollowUpQuestionGenerator.generate_follow_up_questions(stances, original_query)
            response_data['follow_up_questions'] = follow_up_questions
            logger.info(f"Generated {len(follow_up_questions)} follow-up questions")
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            response_data['follow_up_questions'] = []
        
        # Update the response with enhanced stances
        if 'arguments' in response_data:
            response_data['arguments'] = stances
        else:
            response_data['stances'] = stances
        
        logger.info("✅ Successfully completed LLM enhancement")
        return response_data
        
    except Exception as e:
        logger.error(f"Fatal error in LLM enhancement: {e}", exc_info=True)
        # Return original response data on any critical failure
        return response_data


# Keep for backward compatibility if needed
enhance_response_with_llm = enhance_response_with_llm_sync