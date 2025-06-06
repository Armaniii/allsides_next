#!/usr/bin/env python
"""
Test script to verify that the Ollama integration works properly
with langchain-ollama's structured output capabilities.
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
    from langchain_core.prompts import PromptTemplate
    from typing import List
except ImportError:
    print("Please install the required packages with:")
    print("pip install langchain-ollama langchain-core")
    sys.exit(1)

# Define a simple structured output schema using Pydantic v1
class Queries(BaseModel):
    """Search queries for research."""
    queries: List[str] = Field(description="List of search queries")
    
    # Add schema_json method for compatibility
    @classmethod
    def schema_json(cls):
        """Return the JSON schema as a string - compatibility method for both v1 and v2."""
        if hasattr(cls, "model_json_schema"):
            # Pydantic v2
            return json.dumps(cls.model_json_schema())
        elif hasattr(cls, "schema"):
            # Pydantic v1
            return json.dumps(cls.schema())
        else:
            # Fallback
            return json.dumps({
                "title": "Queries",
                "description": "Search queries for research.",
                "type": "object",
                "properties": {
                    "queries": {
                        "title": "Queries",
                        "description": "List of search queries",
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["queries"]
            })

# Test function
def test_ollama_structured_output():
    """Test Ollama's ability to handle structured output."""
    print("\n=== Testing Ollama with Structured Output ===\n")
    
    # Check if OLLAMA_HOST is set
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    print(f"Using Ollama host: {ollama_host}")
    
    # List available models
    try:
        import requests
        models_url = f"{ollama_host}/api/tags"
        response = requests.get(models_url)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("\nAvailable Ollama models:")
            for model in models:
                print(f"- {model.get('name')}")
            
            # Let user choose a model
            selected_model = input("\nEnter model name to test (or press Enter for default 'llama3.2:1b'): ")
            if not selected_model:
                selected_model = "llama3.2:1b"
        else:
            print("Could not retrieve models. Using default 'llama3.2:1b'")
            selected_model = "llama3.2:1b"
    except Exception as e:
        print(f"Error retrieving models: {e}")
        print("Using default 'llama3.2:1b'")
        selected_model = "llama3.2:1b"
    
    # Initialize the parser with proper schema handling
    parser = PydanticOutputParser(pydantic_object=Queries)
    
    # Force format instructions to use our schema method
    format_instructions = f"""The output should be formatted as a JSON instance that conforms to the JSON schema below.

```json
{Queries.schema_json()}
```

For example:
```json
{{
  "queries": [
    "climate change effects on crop yields",
    "agricultural adaptation to rising temperatures",
    "impact of extreme weather on farming"
  ]
}}
```"""
    
    # Initialize the model
    try:
        print(f"\nInitializing ChatOllama with model '{selected_model}'")
        model = ChatOllama(
            model=selected_model,
            temperature=0,
            format="json",  # This is crucial for structured output
            base_url=ollama_host
        )
        
        # Create a prompt template with explicit format instructions
        template = """
        Please generate 3-5 search queries for researching the following topic: {topic}
        
        {format_instructions}
        
        Return ONLY valid JSON that follows the exact schema specified, with no additional text or explanations.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["topic"],
            partial_variables={"format_instructions": format_instructions}
        )
        
        # Create the chain
        topic = "Climate change impact on agriculture"
        print(f"\nGenerating structured output for topic: '{topic}'\n")
        
        # Try alternative approach first - get raw response and parse manually
        from langchain_core.output_parsers import StrOutputParser
        
        raw_chain = prompt | model | StrOutputParser()
        raw_output = raw_chain.invoke({"topic": topic})
        
        print("\nRaw output from model:")
        print(raw_output)
        
        # Try to extract and parse any JSON in the response
        try:
            import re
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                print("\nJSON successfully extracted and parsed:")
                pprint(data)
                
                # Create result object manually
                if "queries" in data:
                    class Result:
                        def __init__(self, queries):
                            self.queries = queries
                    
                    result = Result(data["queries"])
                    
                    print("\n=== SUCCESS: Parsed result ===")
                    print("Queries:")
                    for i, query in enumerate(result.queries, 1):
                        print(f"{i}. {query}")
                    
                    print("\nStructured output was successfully parsed ✓")
                else:
                    print("\nJSON found but missing 'queries' field. Format:")
                    pprint(data)
            else:
                print("\nNo valid JSON found in the response.")
        except Exception as parse_error:
            print(f"\nError parsing JSON: {parse_error}")
        
        # Only try the full parser chain if we need to
        try_full_chain = False
        if try_full_chain:
            try:
                chain = prompt | model | parser
                result = chain.invoke({"topic": topic})
                
                print("\n=== SUCCESS: Parsed result with PydanticOutputParser ===")
                print("Queries:")
                for i, query in enumerate(result.queries, 1):
                    print(f"{i}. {query}")
                    
                print("\nStructured output was properly parsed by the PydanticOutputParser ✓")
            except Exception as chain_error:
                print(f"\n=== ERROR with PydanticOutputParser: {chain_error} ===")
    
    except Exception as e:
        print(f"Error during test: {e}")
    
    print("\n=== Test completed ===\n")

if __name__ == "__main__":
    test_ollama_structured_output() 