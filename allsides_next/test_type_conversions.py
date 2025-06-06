import os
import asyncio
import sys
import json
from typing import Dict, Any, Optional, List

# Test script for verifying numeric type conversions in ResearchManager

async def test_config_conversion():
    """Test the numeric type conversions in ResearchManager."""
    try:
        # Import from the correct module path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from backend.api.research.graph_manager import ResearchManager
        
        # Create a manager with string values for numeric parameters
        test_configs = [
            {"max_search_depth": "3", "number_of_queries": "2"},
            {"max_search_depth": 3, "number_of_queries": "2"},
            {"max_search_depth": "3", "number_of_queries": 2},
            {"max_search_depth": None, "number_of_queries": "invalid"},
            {"max_search_depth": "invalid", "number_of_queries": None},
            {}  # Empty config
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\nTest {i+1}: Configuration {config}")
            manager = ResearchManager(config)
            
            # Check config conversions in the manager itself
            print("Manager config types:")
            max_depth = manager.config.get('max_search_depth')
            num_queries = manager.config.get('number_of_queries')
            print(f"max_search_depth: {type(max_depth).__name__} = {max_depth}")
            print(f"number_of_queries: {type(num_queries).__name__} = {num_queries}")
            
            # Check thread configuration
            thread_id, thread = await manager.start_research(f'Test topic {i+1}')
            
            # Verify types in the thread configuration
            thread_max_depth = thread['configurable'].get('max_search_depth')
            thread_num_queries = thread['configurable'].get('number_of_queries')
            
            print("Thread config types:")
            print(f"max_search_depth: {type(thread_max_depth).__name__} = {thread_max_depth}")
            print(f"number_of_queries: {type(thread_num_queries).__name__} = {thread_num_queries}")
            
            # Check for correctness
            assert isinstance(thread_max_depth, int), f"max_search_depth should be int, got {type(thread_max_depth).__name__}"
            assert isinstance(thread_num_queries, int), f"number_of_queries should be int, got {type(thread_num_queries).__name__}"
        
        print("\nAll type conversion tests passed successfully!")
        return True
    except Exception as e:
        print(f"Error in test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_config_conversion())
    sys.exit(0 if success else 1) 