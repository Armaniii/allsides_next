#!/usr/bin/env python
"""
Test script to verify that the Queries and SearchQuery patches work correctly.
This tests the conversion of string queries to SearchQuery objects.
"""

import os
import sys
import json
from pprint import pprint

# Add the project root to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the necessary modules
try:
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.pydantic_v1 import BaseModel, Field
    from typing import List, Union
except ImportError:
    print("Please install the required packages with:")
    print("pip install langchain-ollama langchain-core pydantic")
    sys.exit(1)

# First define the unpatched model
class SearchQuery(BaseModel):
    """Search query for research."""
    search_query: str = Field(description="Query for web search.")

class Queries(BaseModel):
    """List of search queries."""
    queries: List[SearchQuery] = Field(description="List of search queries.")

# Create a patched Queries model that can handle string queries
class PatchedQueries(BaseModel):
    """Patched Queries model that can handle string queries."""
    queries: List[Union[SearchQuery, str]] = Field(description="List of search queries that can be strings or SearchQuery objects.")
    
    def __init__(self, **data):
        # Convert string queries to SearchQuery objects
        if "queries" in data and isinstance(data["queries"], list):
            for i, query in enumerate(data["queries"]):
                if isinstance(query, str):
                    data["queries"][i] = SearchQuery(search_query=query)
        super().__init__(**data)

def test_queries_validation():
    """Test validation for Queries and PatchedQueries models."""
    print("\n=== Testing Queries and SearchQuery Validation ===\n")
    
    # Test data with strings instead of SearchQuery objects
    test_data = {
        "queries": [
            "first query string",
            "second query string",
            "third query string"
        ]
    }
    
    # Test with unpatched Queries
    print("1. Testing unpatched Queries model with string values (should fail):")
    try:
        queries = Queries(**test_data)
        print("SUCCESS (Unexpected!): Unpatched Queries accepted string values")
        print(f"Queries: {queries.queries}")
    except Exception as e:
        print(f"EXPECTED ERROR: {e}")
    
    # Test with patched Queries
    print("\n2. Testing patched Queries model with string values (should work):")
    try:
        patched_queries = PatchedQueries(**test_data)
        print("SUCCESS: Patched Queries accepted string values")
        print("Converted Queries:")
        for i, query in enumerate(patched_queries.queries):
            print(f"  {i+1}. {query.search_query}")
    except Exception as e:
        print(f"ERROR (Unexpected!): {e}")
    
    # Test with JSON string that would be returned by Ollama
    ollama_json = '{"queries": ["query about immigration", "immigration policies research", "immigration historical context"]}'
    
    print("\n3. Testing with JSON response from Ollama (unpatched, should fail):")
    try:
        # Parse JSON
        data = json.loads(ollama_json)
        # Try to create Queries
        queries = Queries(**data)
        print("SUCCESS (Unexpected!): Unpatched Queries accepted JSON with strings")
    except Exception as e:
        print(f"EXPECTED ERROR: {e}")
    
    print("\n4. Testing with JSON response from Ollama (patched, should work):")
    try:
        # Parse JSON
        data = json.loads(ollama_json)
        # Try to create PatchedQueries
        patched_queries = PatchedQueries(**data)
        print("SUCCESS: Patched Queries accepted JSON with strings")
        print("Converted Queries:")
        for i, query in enumerate(patched_queries.queries):
            print(f"  {i+1}. {query.search_query}")
    except Exception as e:
        print(f"ERROR (Unexpected!): {e}")
    
    # Test with correct format
    correct_data = {
        "queries": [
            {"search_query": "first query in correct format"},
            {"search_query": "second query in correct format"}
        ]
    }
    
    print("\n5. Testing with correctly formatted data (both should work):")
    try:
        # Try with unpatched Queries
        queries = Queries(**correct_data)
        print("SUCCESS: Unpatched Queries accepted correctly formatted data")
        
        # Try with patched Queries
        patched_queries = PatchedQueries(**correct_data)
        print("SUCCESS: Patched Queries accepted correctly formatted data")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    test_queries_validation() 