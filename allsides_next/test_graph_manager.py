import os
import asyncio
import sys
import json
from typing import Dict, Any, Optional, List, Tuple, AsyncGenerator

# Mock classes and functions to test without Django
class MemorySaver:
    def __init__(self):
        pass

class Command:
    def __init__(self, resume=None):
        self.resume = resume

class MockBuilder:
    def compile(self, checkpointer=None):
        return MockGraph()

class MockGraph:
    def __init__(self):
        pass
    
    async def astream(self, *args, **kwargs):
        # Mock yielding a few events
        yield {"type": "test_event", "message": "Test started"}
        yield {"type": "report_status", "status": "COMPLETED"}
        
    def get_state(self, state):
        return state

# Mock the package imports  
class MockModule:
    pass

sys.modules['langgraph.checkpoint.memory'] = MockModule()
sys.modules['langgraph.checkpoint.memory'].MemorySaver = MemorySaver
sys.modules['langgraph.types'] = MockModule()
sys.modules['langgraph.types'].Command = Command
sys.modules['open_deep_research.graph'] = MockModule()
sys.modules['open_deep_research.graph'].builder = MockBuilder()

# Now we can import and test our graph_manager
class ResearchManager:
    """Simplified version of the ResearchManager for testing."""
    
    DEFAULT_CONFIG = {
        "search_api": "tavily",
        "planner_provider": "openai",
        "planner_model": "gpt-4o",
        "writer_provider": "openai",
        "writer_model": "gpt-4o",
        "max_search_depth": "2",  # String on purpose
        "number_of_queries": "2",  # String on purpose
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the research manager."""
        # Copy default config and update with provided config
        self.config = self.DEFAULT_CONFIG.copy()

        if config:
            self.config.update(config)
        
        # Ensure numeric values are integers to avoid type errors in comparisons
        if "max_search_depth" in self.config:
            try:
                self.config["max_search_depth"] = int(self.config["max_search_depth"])
            except (ValueError, TypeError):
                self.config["max_search_depth"] = 2  # Default to 2 if conversion fails
                
        if "number_of_queries" in self.config:
            try:
                self.config["number_of_queries"] = int(self.config["number_of_queries"])
            except (ValueError, TypeError):
                self.config["number_of_queries"] = 2  # Default to 2 if conversion fails
            
        # Initialize the graph
        self.graph = MockGraph()
        self._thread_cache = {}
    
    async def start_research(self, topic: str) -> Tuple[str, Dict[str, Any]]:
        """Start a new research session."""
        thread_id = "test-thread-id"
            
        try:
            # Create a thread configuration
            thread_config = {
                "thread_id": thread_id,
                "topic": topic,
                "sections": [],
                "report_structure": "Test structure"
            }
            
            # Add all other config parameters, ensuring numeric values are integers
            for key, value in self.config.items():
                if key not in thread_config:
                    # Convert numeric values to int as needed
                    if key in ["max_search_depth", "number_of_queries"] and value is not None:
                        try:
                            thread_config[key] = int(value)
                        except (ValueError, TypeError):
                            thread_config[key] = value
                    else:
                        thread_config[key] = value
            
            # Create the final thread object with configurable dict
            thread = {
                "configurable": thread_config
            }
            
            return thread_id, thread
        except Exception as e:
            print(f"Error starting research: {str(e)}")
            raise

async def test_research_manager():
    try:
        # Create manager with string values for numeric parameters
        config = {
            "max_search_depth": "3",  # String instead of int
            "number_of_queries": "2"  # String instead of int
        }
        manager = ResearchManager(config)
        
        # Check config conversion in the manager
        print("Manager config types:")
        print(f"max_search_depth: {type(manager.config['max_search_depth']).__name__} = {manager.config['max_search_depth']}")
        print(f"number_of_queries: {type(manager.config['number_of_queries']).__name__} = {manager.config['number_of_queries']}")
        
        # Start research and check thread config
        thread_id, thread = await manager.start_research('Climate change testing')
        print(f"\nThread created with ID: {thread_id}")
        print("Thread config types:")
        print(f"max_search_depth: {type(thread['configurable']['max_search_depth']).__name__} = {thread['configurable']['max_search_depth']}")
        print(f"number_of_queries: {type(thread['configurable']['number_of_queries']).__name__} = {thread['configurable']['number_of_queries']}")
        
        print("\nTest completed successfully!")
        return True
    except Exception as e:
        print(f"Error in test: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_research_manager())
    sys.exit(0 if success else 1) 