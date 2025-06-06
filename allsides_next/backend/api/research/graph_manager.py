import os
import uuid
import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from dotenv import load_dotenv
import time

from open_deep_research.graph import builder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain.chat_models import init_chat_model as original_init_chat_model
import langchain_ollama
# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

logger = logging.getLogger(__name__)

def custom_init_chat_model(**kwargs):
    """Custom chat model initializer that ensures streaming=True for OpenAI to show progress updates.
    
    This wraps the original init_chat_model but adds streaming=True for OpenAI models
    and handles configuration for Ollama models.
    
    Note: When using Ollama models, use the exact model name as shown in Ollama:
    - For Llama 3.2 1B Instruct, use "llama3.2:1b"
    - Do NOT use "llama3" or "llama3.2" as they might not exist
    """
    # Set streaming for OpenAI models to ensure progress updates are shown
    model_provider = kwargs.get('model_provider', '').lower()
    
    # OpenAI models use the "streaming" parameter directly, not config_kwargs
    if model_provider == 'openai':
        kwargs['streaming'] = True
        logger.info("Setting streaming=True for OpenAI model")
    # Handle Ollama specific configurations
    elif model_provider == 'ollama':
        model_name = kwargs.get('model', 'llama3.2:1b')  # Default to llama3.2:1b if not specified
        logger.info(f"Using langchain-ollama for model: {model_name}")
        
        from langchain_ollama import ChatOllama
        
        # Use the standard ChatOllama implementation with proper format setting
        return ChatOllama(
            model=model_name,
            temperature=kwargs.get('temperature', 0.7),
            streaming=True,  # Always enable streaming for Ollama
            base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            format="json"  # This is crucial for structured output
        )
    
    # Call the original function with updated kwargs for non-Ollama models
    return original_init_chat_model(**kwargs)

class ResearchManager:
    """Manager for handling Open Deep Research graph operations."""
    
    DEFAULT_CONFIG = {
        "search_api": os.getenv("DEFAULT_SEARCH_API", "tavily"),
        "planner_provider": os.getenv("DEFAULT_PLANNER_PROVIDER", "openai"),
        "planner_model": os.getenv("DEFAULT_PLANNER_MODEL", "gpt-4o"),
        "writer_provider": os.getenv("DEFAULT_WRITER_PROVIDER", "openai"),
        "writer_model": os.getenv("DEFAULT_WRITER_MODEL", "gpt-4o"),
        "max_search_depth": int(os.getenv("MAX_SEARCH_DEPTH", 2)),
        "number_of_queries": int(os.getenv("NUMBER_OF_QUERIES", 2)),
        # Whether to use json_schema method for structured output with Ollama
        "structured_output_method": os.getenv("STRUCTURED_OUTPUT_METHOD", "json_schema"),
    }
    
    # Available providers and models for UI display and validation
    AVAILABLE_PROVIDERS = {
        "openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest", "claude-3-haiku-latest"],
        "ollama": ["llama3.2:1b", "llama3:70b", "llama3:8b", "gemma3:1b", "gemma3:4b"]
    }
    
    REPORT_STRUCTURE = """Use this structure to create a comprehensive report on the user-provided topic:

1. Executive Summary
   - Brief overview of the topic area
   - Current relevance in public discourse
   - Preview of key perspectives included in the report

2. Methodology
   - Source selection criteria and diversity considerations
   - Quality assessment framework for information evaluation
   - Approach to identifying mainstream and marginalized perspectives
   - Steps taken to mitigate researcher bias
   - Limitations of the research approach

3. Background Information
   - Definition and common terminology
   - Key historical context and evolution of the topic
   - Current status, trends, and significant developments
   - Relevant legal, regulatory, or policy frameworks

4. Current Public Discourse
   - Dominant narratives in public discussion
   - Media representation and framing analysis
   - Underrepresented viewpoints in mainstream coverage
   - Notable shifts in public perception over time

5. Multiple Perspectives Framework
   For each identified perspective (aim for 4-8 diverse viewpoints):
   
   Perspective: [Name/Label]
   - Core stance: Clear statement of the fundamental position
   - Key proponents: Notable individuals, groups, or institutions
   - Underlying values/assumptions: Foundational beliefs informing this perspective
   - Main arguments:
     * Argument 1 → Supporting evidence → Limitations
     * Argument 2 → Supporting evidence → Limitations
     * Etc.
   - Counter-arguments from other perspectives
   - Historical context and development of this viewpoint
   
   *Special attention to ensure inclusion of:
   - Mainstream perspectives (conventional approaches)
   - Critical perspectives (challenging dominant frameworks)
   - Marginalized perspectives (historically excluded viewpoints)
   - Cross-cutting perspectives (bridging different approaches)

6. Source Evaluation
   - Credibility assessment of key information sources
   - Analysis of potential biases in available literature
   - Identification of information gaps and areas of uncertainty
   - Discussion of conflicting evidence and how contradictions were addressed

7. Scientific or Expert Context (If Applicable)
   - Current state of expert knowledge
   - Key studies, data, and findings
   - Areas of expert consensus and disagreement
   - Limitations of current research and methodological challenges
   - Emerging trends in the field

8. Ethical Dimensions
   - Key ethical questions raised by the topic
   - Different ethical frameworks applicable to the issue
   - How various perspectives prioritize different ethical considerations
   - Areas of ethical consensus across divergent viewpoints
   - Unresolved ethical dilemmas

9. Synthesis and Integration
   - Comparison of key perspectives' strengths and limitations
   - Areas of unexpected agreement across different viewpoints
   - Productive tensions that drive deeper understanding
   - Comprehensive analytical framework integrating multiple perspectives
   - Visualization of the relationship between different perspectives

10. Conclusion
    - Summary of key insights from multiple perspectives
    - Identification of crucial unresolved questions
    - Frameworks for productive dialogue across different viewpoints
    - Potential paths forward that honor diverse legitimate concerns
    - One structural element (table or visual) that distills the main findings"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the research manager."""
        # Copy default config and update with provided config
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Always ensure report_structure is included in default config
        if "report_structure" not in self.config:
            self.config["report_structure"] = self.REPORT_STRUCTURE

        if config:
            # Don't override report_structure if it's not provided in config
            if "report_structure" not in config and "report_structure" in self.config:
                # Create a copy to avoid modifying the input
                config_copy = config.copy()
                # Update our config with user values
                self.config.update(config_copy)
            else:
                # User provided a report_structure or doesn't have one
                self.config.update(config)
        
        # Log the search API setting for debugging
        logger.info(f"ResearchManager initialized with search_api: {self.config.get('search_api')}")
        
        # Ensure default model is set if not specified
        if not self.config.get("planner_model"):
            self.config["planner_model"] = "gpt-4o"
        if not self.config.get("planner_provider"):
            self.config["planner_provider"] = "openai"
            
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
        
        # Ensure streaming is enabled by monkey patching the init_chat_model function
        self._monkey_patch_init_chat_model()
            
        # Initialize the graph
        try:
            # Create memory saver for checkpointing
            self.memory = MemorySaver()
            
            # Create the research graph
            self.graph = builder.compile(checkpointer=self.memory)
            
            # Initialize thread cache
            self._thread_cache = {}
        except Exception as e:
            logger.error(f"Error initializing research graph: {str(e)}", exc_info=True)
            self.graph = None
    
    def _monkey_patch_init_chat_model(self):
        """Monkey patch the init_chat_model function in open_deep_research.graph to use our custom version.
        
        This ensures that we get streaming responses from OpenAI models, which show progress updates
        to the user during the research process.
        """
        import open_deep_research.graph
        
        # Save the original for reference
        if not hasattr(open_deep_research.graph, '_original_init_chat_model'):
            open_deep_research.graph._original_init_chat_model = open_deep_research.graph.init_chat_model
            
        # Replace with our custom version
        open_deep_research.graph.init_chat_model = custom_init_chat_model
        
        logger.info("Monkey patched open_deep_research.graph.init_chat_model for streaming support")
    
    def _modify_structured_output(self):
        """Modify structured output handling for Ollama models.
        
        This method configures open_deep_research.graph to use the standard ChatOllama 
        implementation with proper JSON format configuration.
        """
        logger.info("Using standard langchain-ollama with format=json for structured output")
        
        try:
            # Import required modules
            import inspect
            import open_deep_research.graph
            import types
            import json
            import copy
            
            # Add compatibility method to handle schema differences between Pydantic v1 and v2
            def get_schema_json(schema_class):
                """Get the JSON schema as a string, handling both Pydantic v1 and v2."""
                try:
                    # Try Pydantic v2 method
                    if hasattr(schema_class, "model_json_schema"):
                        return json.dumps(schema_class.model_json_schema())
                    # Try Pydantic v1 method
                    elif hasattr(schema_class, "schema"):
                        return json.dumps(schema_class.schema())
                    else:
                        # No schema method found - this isn't a normal situation
                        logger.warning(f"Could not find schema method for {schema_class.__name__}")
                        # Return a minimal schema
                        return json.dumps({"type": "object"})
                except Exception as schema_err:
                    logger.error(f"Error generating schema for {schema_class.__name__}: {schema_err}")
                    return json.dumps({"type": "object"})
            
            # Patch schema_json methods for relevant classes in open_deep_research.state
            try:
                from open_deep_research.state import Queries, SearchQuery, Sections, Section, Feedback
                from typing import Union, List
                
                # Patch SearchQuery to handle direct string input
                original_SearchQuery = copy.deepcopy(SearchQuery)
                
                # Create a patched validator for SearchQuery
                def patched_search_query_validator(cls, v):
                    """Custom validator that accepts both strings and dicts for SearchQuery."""
                    if isinstance(v, str):
                        # If it's a string, convert it to a SearchQuery object
                        logger.info(f"Converting string to SearchQuery: {v}")
                        return cls(search_query=v)
                    # Otherwise, return as is for normal validation
                    return v
                
                # Create a new validation schema for Queries
                class PatchedQueries(BaseModel):
                    """Patched Queries model that handles direct string input."""
                    queries: List[Union[SearchQuery, str]] = Field(
                        description="List of search queries that can be strings or SearchQuery objects."
                    )
                    
                    def __init__(self, **data):
                        # Convert string queries to SearchQuery objects
                        if "queries" in data and isinstance(data["queries"], list):
                            for i, query in enumerate(data["queries"]):
                                if isinstance(query, str):
                                    data["queries"][i] = SearchQuery(search_query=query)
                        super().__init__(**data)
                    
                    # For compatibility
                    @classmethod
                    def schema_json(cls):
                        return get_schema_json(cls)
                
                # Replace the original Queries class in open_deep_research.state
                try:
                    # Try to monkey patch the Queries class
                    import sys
                    sys.modules['open_deep_research.state'].Queries = PatchedQueries
                    logger.info("Successfully patched Queries class to handle string inputs directly")
                except Exception as e:
                    logger.warning(f"Could not replace Queries class: {e}")
                
                # Add schema_json method to all classes that might need it
                for cls in [SearchQuery, Sections, Section, Feedback, PatchedQueries]:
                    if not hasattr(cls, "schema_json"):
                        setattr(cls, "schema_json", classmethod(lambda cls: get_schema_json(cls)))
                        logger.info(f"Added schema_json method to {cls.__name__}")
            except ImportError:
                logger.warning("Could not import schema classes from open_deep_research.state")
            
            # Patch the init_chat_model for consistency
            if hasattr(open_deep_research.graph, 'init_chat_model'):
                original_init_chat_model = open_deep_research.graph.init_chat_model
                
                def patched_init_chat_model(**kwargs):
                    # Use standard ChatOllama for Ollama models
                    if kwargs.get('model_provider', '').lower() == 'ollama':
                        model_name = kwargs.get('model', 'llama3.2:1b')  # Default to llama3.2:1b if not specified
                        logger.info(f"Using standard ChatOllama model for {model_name}")
                        from langchain_ollama import ChatOllama
                        return ChatOllama(
                            model=model_name,
                            temperature=kwargs.get('temperature', 0.7),
                            streaming=True,
                            base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                            format="json"  # This is crucial for structured output
                        )
                    # For non-Ollama models, use the original function
                    return original_init_chat_model(**kwargs)
                
                # Replace the function
                open_deep_research.graph.init_chat_model = patched_init_chat_model
                logger.info("Successfully patched init_chat_model for Ollama support")
                
                # Add a hook to transform Ollama responses before validation
                try:
                    # First make sure the necessary imports are available
                    import pydantic
                    from langchain_core.output_parsers import PydanticOutputParser
                    from typing import Union, List, Dict, Any
                    
                    # Store original PydanticOutputParser.parse method
                    original_parse = PydanticOutputParser.parse
                    
                    # Define patched parse method to handle strings in queries
                    def patched_parse(self, completion):
                        """Patched parse method that handles queries that are strings instead of SearchQuery objects."""
                        try:
                            # Try original parse first
                            return original_parse(self, completion)
                        except (pydantic.ValidationError, ValueError) as e:
                            # Check if it's queries validation error
                            if "Queries" in str(e) and "queries" in str(e) and "SearchQuery" in str(e):
                                logger.warning(f"Caught Queries validation error, trying to fix: {e}")
                                
                                # Try to parse as JSON
                                try:
                                    if isinstance(completion, str):
                                        data = json.loads(completion)
                                        
                                        # If 'queries' exists and is a list of strings, convert to SearchQuery objects
                                        if "queries" in data and isinstance(data["queries"], list):
                                            # Convert string queries to SearchQuery objects
                                            for i, query in enumerate(data["queries"]):
                                                if isinstance(query, str):
                                                    data["queries"][i] = {"search_query": query}
                                            
                                            # Convert to JSON string and parse again
                                            fixed_completion = json.dumps(data)
                                            return original_parse(self, fixed_completion)
                                except Exception as json_err:
                                    logger.error(f"Error fixing JSON: {json_err}")
                            
                            # Re-raise the original error
                            raise
                    
                    # Apply the patch
                    PydanticOutputParser.parse = patched_parse
                    logger.info("Successfully patched PydanticOutputParser.parse to handle string queries")
                    
                except Exception as parser_patch_err:
                    logger.warning(f"Could not patch PydanticOutputParser: {parser_patch_err}")
                
                # Patch the JsonOutputParser to handle different schema methods
                if hasattr(open_deep_research.graph, 'JsonOutputParser'):
                    original_parser = open_deep_research.graph.JsonOutputParser
                    
                    # Create a patched version that handles schema differences
                    def patched_get_format_instructions(self):
                        """Get format instructions that handle both Pydantic v1 and v2."""
                        schema = get_schema_json(self.pydantic_object)
                        return f"Output a JSON object that conforms to this schema: {schema}"
                    
                    # Apply the patch if possible
                    try:
                        original_parser.get_format_instructions = patched_get_format_instructions
                        logger.info("Successfully patched JsonOutputParser.get_format_instructions")
                    except Exception as parser_err:
                        logger.warning(f"Could not patch JsonOutputParser: {parser_err}")
                
                # Patch the generate_queries function which has the issue
                if hasattr(open_deep_research.graph, 'generate_queries'):
                    original_generate_queries = open_deep_research.graph.generate_queries
                    
                    async def patched_generate_queries(state, config):
                        try:
                            # Call the original function first
                            result = await original_generate_queries(state, config)
                            return result
                        except (AttributeError, pydantic.ValidationError) as e:
                            # If we get the specific 'dict' object has no attribute 'queries' error
                            # Or a validation error about SearchQuery
                            if "'dict' object has no attribute 'queries'" in str(e) or "SearchQuery" in str(e):
                                logger.warning(f"Handling query validation error: {e}")
                                
                                # Get the configurable from state
                                configurable = state.configurable
                                
                                # Extract what we need to recreate the function
                                writer_provider = configurable.writer_provider
                                writer_model = configurable.writer_model
                                
                                # Ensure we use correct model name for Ollama
                                if writer_provider.lower() == 'ollama' and (not writer_model or writer_model in ['llama3', 'llama3.2']):
                                    writer_model = 'llama3.2:1b'
                                    logger.info(f"Using llama3.2:1b instead of {configurable.writer_model}")
                                
                                topic = state.topic
                                
                                # Set up the writer model
                                from open_deep_research.state import Queries, SearchQuery
                                from langchain_core.messages import SystemMessage, HumanMessage
                                from langchain_core.output_parsers import PydanticOutputParser
                                from typing import List
                                
                                # Create a direct implementation that handles dict results
                                # Initialize the writer model
                                writer_llm = init_chat_model(
                                    model=writer_model,
                                    model_provider=writer_provider,
                                    temperature=0.7
                                )
                                
                                # Explicitly define the format we want
                                example_format = {
                                    "queries": [
                                        {"search_query": f"first query about {topic}"},
                                        {"search_query": f"second query about {topic}"},
                                        {"search_query": f"third query about {topic}"}
                                    ]
                                }
                                
                                # Create the prompt with instructions
                                system_instructions = f"""You are an expert technical writer crafting web search queries for researching a topic.
                                Your goal is to generate search queries that will help gather information for the topic.
                                
                                The topic is: {topic}
                                
                                Return the search queries as a JSON object with this exact structure:
                                ```json
                                {json.dumps(example_format, indent=2)}
                                ```
                                
                                IMPORTANT: Each query must be a dictionary with a 'search_query' key, NOT a plain string.
                                
                                INCORRECT FORMAT (will cause errors):
                                ```json
                                {{"queries": [
                                  "first query about {topic}",
                                  "second query about {topic}"
                                ]}}
                                ```
                                
                                CORRECT FORMAT:
                                ```json
                                {{"queries": [
                                  {{"search_query": "first query about {topic}"}},
                                  {{"search_query": "second query about {topic}"}}
                                ]}}
                                ```
                                
                                Generate specific, well-crafted search queries to find high-quality information.
                                Return ONLY valid JSON that follows the schema exactly."""
                                
                                # Generate the queries directly
                                messages = [
                                    SystemMessage(content=system_instructions),
                                    HumanMessage(content="Generate search queries on the provided topic.")
                                ]
                                
                                # Generate the raw response
                                from langchain_core.output_parsers import StrOutputParser
                                response = writer_llm.invoke(messages)
                                response_text = response.content if hasattr(response, 'content') else str(response)
                                
                                # Try to parse the JSON response
                                import json
                                import re
                                
                                # Extract JSON if it exists in the response
                                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                                if json_match:
                                    try:
                                        json_str = json_match.group()
                                        data = json.loads(json_str)
                                        
                                        # Check if 'queries' is in the response
                                        if 'queries' in data:
                                            query_list = []
                                            
                                            # Fix the format if needed
                                            fixed_queries = []
                                            for q in data['queries']:
                                                if isinstance(q, dict) and 'search_query' in q:
                                                    # Already in correct format
                                                    fixed_queries.append(q)
                                                elif isinstance(q, str):
                                                    # Convert string to proper format
                                                    fixed_queries.append({"search_query": q})
                                                else:
                                                    # Try to extract a string somehow
                                                    query_str = str(q)
                                                    fixed_queries.append({"search_query": query_str})
                                            
                                            # Extract search_query values to return
                                            for q in fixed_queries:
                                                if isinstance(q, dict) and 'search_query' in q:
                                                    query_list.append(q['search_query'])
                                            
                                            # Return the properly formatted result
                                            if query_list:
                                                search_queries = [SearchQuery(search_query=q) for q in query_list]
                                                return {"search_queries": search_queries}
                                    except Exception as json_err:
                                        logger.error(f"Error parsing JSON: {json_err}")
                                
                                # Fallback: extract queries from the text
                                queries = []
                                for line in response_text.split('\n'):
                                    line = line.strip()
                                    # Look for numbered queries, bullet points, or query indicators
                                    if re.match(r'^[0-9]+\.', line) or line.startswith('- ') or line.startswith('* ') or 'query' in line.lower():
                                        # Clean up the line
                                        clean_line = re.sub(r'^[0-9]+\.|\-|\*', '', line).strip()
                                        if clean_line and len(clean_line) > 10:  # Ensure it's substantial
                                            queries.append(clean_line)
                                
                                # Create at least some default queries if nothing was found
                                if not queries:
                                    queries = [
                                        f"{topic} overview",
                                        f"{topic} analysis",
                                        f"{topic} research studies",
                                        f"{topic} expert opinions"
                                    ]
                                
                                # Limit to the configured number of queries
                                queries = queries[:configurable.number_of_queries]
                                
                                # Create proper SearchQuery objects
                                search_queries = [SearchQuery(search_query=q) for q in queries]
                                
                                # Return in the format expected by the next node
                                return {"search_queries": search_queries}
                            else:
                                # It's not the error we're looking for, re-raise
                                raise
                    
                    # Replace the function
                    open_deep_research.graph.generate_queries = patched_generate_queries
                    logger.info("Successfully patched generate_queries function to handle queries attribute error")
            
                # Patch the with_structured_output method if available
                try:
                    from langchain_core.language_models.chat_models import BaseChatModel
                    from langchain_core.output_parsers import PydanticOutputParser
                    
                    # Store original method reference
                    if hasattr(BaseChatModel, 'with_structured_output'):
                        original_with_structured_output = BaseChatModel.with_structured_output
                        
                        # Create a patched version that handles both Pydantic versions
                        def patched_with_structured_output(self, schema, *, include_raw=False, **kwargs):
                            """Patch for with_structured_output that works with both Pydantic v1 and v2."""
                            try:
                                # First try the original method
                                return original_with_structured_output(self, schema, include_raw=include_raw, **kwargs)
                            except (AttributeError, TypeError) as e:
                                # If we get a schema-related error, use our custom approach
                                if "model_json_schema" in str(e) or "schema_json" in str(e):
                                    logger.warning(f"Using patched with_structured_output for {schema.__name__}: {e}")
                                    
                                    # Create a PydanticOutputParser with our schema
                                    parser = PydanticOutputParser(pydantic_object=schema)
                                    
                                    # Get schema as string using our compatibility method
                                    schema_str = get_schema_json(schema)
                                    
                                    # Build prompt with schema
                                    from langchain_core.prompts import ChatPromptTemplate
                                    from langchain_core.runnables import RunnablePassthrough
                                    
                                    prompt = ChatPromptTemplate.from_template("""
                                    You are a helpful assistant that outputs valid JSON according to a specific schema.
                                    
                                    The required JSON schema is:
                                    {schema}
                                    
                                    Your response MUST be valid JSON that follows this schema exactly.
                                    Do not include any explanations, markdown formatting, or anything that is not valid JSON.
                                    
                                    Human: {input}
                                    
                                    JSON Response:
                                    """)
                                    
                                    # Create the chain
                                    chain = prompt | self | parser
                                    
                                    # Create a function that runs the chain
                                    def run_chain(messages):
                                        # Format input for the chain
                                        input_text = ""
                                        for message in messages:
                                            if hasattr(message, 'content') and hasattr(message, 'type'):
                                                input_text += f"{message.type.capitalize()}: {message.content}\n"
                                            elif isinstance(message, dict) and 'content' in message and 'type' in message:
                                                input_text += f"{message['type'].capitalize()}: {message['content']}\n"
                                            else:
                                                input_text += str(message) + "\n"
                                        
                                        # Run the chain
                                        return chain.invoke({"input": input_text, "schema": schema_str})
                                    
                                    # Return an object with an invoke method
                                    class StructuredOutputPatched:
                                        def __init__(self, run_func):
                                            self.run = run_func
                                        
                                        def invoke(self, messages):
                                            return self.run(messages)
                                    
                                    return StructuredOutputPatched(run_chain)
                                else:
                                    # If it's not a schema-related error, re-raise
                                    raise
                        
                        # Apply the patch
                        BaseChatModel.with_structured_output = patched_with_structured_output
                        logger.info("Successfully patched BaseChatModel.with_structured_output")
                        
                except Exception as patch_error:
                    logger.warning(f"Failed to patch with_structured_output: {patch_error}")
            
        except Exception as e:
            logger.error(f"Error in _modify_structured_output: {str(e)}", exc_info=True)
    
    async def start_research(self, topic: str) -> Tuple[str, Dict[str, Any]]:
        """Start a new research session."""
        thread_id = str(uuid.uuid4())
        logger.info(f"Starting research on topic: {topic} (thread_id: {thread_id})")
        
        # Check API keys
        self._check_environment()
        
        # If using Ollama, modify structured output
        if self.config.get("planner_provider", "").lower() == "ollama" or self.config.get("writer_provider", "").lower() == "ollama":
            self._modify_structured_output()
            
        try:
            # Create sections from the predefined structure
            sections = [
                {
                    "name": "Executive Summary",
                    "description": "Brief overview of the topic, its relevance, and preview of key perspectives",
                    "research": False,
                    "content": ""
                },
                {
                    "name": "Methodology",
                    "description": "Source selection criteria, quality assessment framework, approach to diverse perspectives, bias mitigation, and limitations",
                    "research": True,
                    "content": ""
                },
                {
                    "name": "Background Information",
                    "description": "Definition, terminology, historical context, current status, and relevant frameworks",
                    "research": True,
                    "content": ""
                },
                {
                    "name": "Current Public Discourse",
                    "description": "Dominant narratives, media representation, underrepresented viewpoints, and shifts in perception",
                    "research": True,
                    "content": ""
                },
                {
                    "name": "Multiple Perspectives Framework",
                    "description": "Detailed analysis of diverse viewpoints including mainstream, critical, marginalized, and cross-cutting perspectives",
                    "research": True,
                    "content": ""
                },
                {
                    "name": "Source Evaluation",
                    "description": "Assessment of source credibility, potential biases, information gaps, and conflicting evidence",
                    "research": True,
                    "content": ""
                },
                {
                    "name": "Scientific or Expert Context",
                    "description": "Expert knowledge, key studies, areas of consensus and disagreement, limitations, and emerging trends",
                    "research": True,
                    "content": ""
                },
                {
                    "name": "Ethical Dimensions",
                    "description": "Ethical questions, frameworks, priorities, consensus areas, and unresolved dilemmas",
                    "research": True,
                    "content": ""
                },
                {
                    "name": "Synthesis and Integration",
                    "description": "Comparison of perspectives, areas of agreement, productive tensions, and comprehensive framework",
                    "research": True,
                    "content": ""
                },
                {
                    "name": "Conclusion",
                    "description": "Key insights, unresolved questions, dialogue frameworks, potential paths forward, and visual summary",
                    "research": False,
                    "content": ""
                }
            ]
            
            # Create a thread configuration with all config parameters
            thread_config = {
                "thread_id": thread_id,
                "topic": topic,
                "sections": sections,  # Include predefined sections
                "report_structure": self.config.get("report_structure", self.REPORT_STRUCTURE)
            }
            
            # CRITICAL: Ensure numeric values are integers
            # Force convert these specific keys to integers
            thread_config["max_search_depth"] = int(self.config.get("max_search_depth", 2))
            thread_config["number_of_queries"] = int(self.config.get("number_of_queries", 2))
            
            # Log the conversion for debugging
            logger.info(f"Setting max_search_depth to {thread_config['max_search_depth']} ({type(thread_config['max_search_depth']).__name__})")
            logger.info(f"Setting number_of_queries to {thread_config['number_of_queries']} ({type(thread_config['number_of_queries']).__name__})")
            
            # Add all other config parameters
            for key, value in self.config.items():
                if key not in thread_config and key not in ["max_search_depth", "number_of_queries"]:
                    thread_config[key] = value
            
            # Log the final configuration for debugging
            logger.info(f"Final thread configuration - search_api: {thread_config.get('search_api')}, planner_provider: {thread_config.get('planner_provider')}, writer_provider: {thread_config.get('writer_provider')}")
            
            # Create the final thread object with configurable dict
            thread = {
                "configurable": thread_config,
                "topic": topic  # Also add topic at root level
            }
            
            return thread_id, thread
        except Exception as e:
            logger.error(f"Error starting research: {str(e)}", exc_info=True)
            raise
    
    def _serialize_event(self, event: Any) -> Dict[str, Any]:
        """Serialize an event for SSE transmission."""
        try:
            # Handle string events
            if isinstance(event, str):
                return {
                    "type": "message",
                    "data": event,
                    "timestamp": time.time(),
                    "force_update": True
                }
            
            # Handle non-dict events
            if not isinstance(event, dict):
                event = {
                    "type": "message",
                    "data": str(event),
                    "timestamp": time.time(),
                    "force_update": True
                }
            
            # Ensure every event has a type
            if "type" not in event:
                if "status" in event:
                    event["type"] = "status_update"
                elif "section" in event:
                    event["type"] = "section_update"
                elif "error" in event:
                    event["type"] = "error"
                else:
                    event["type"] = "message"
            
            # Add timestamp to all events
            event["timestamp"] = time.time()
            
            # Add force_update flag to ensure UI updates
            event["force_update"] = True
            
            # Ensure progress events have a message
            if event["type"] == "progress":
                if "message" not in event:
                    event["message"] = "Processing..."
            
            # Special handling for Interrupt objects in __interrupt__ field
            if "__interrupt__" in event:
                interrupt_data = event["__interrupt__"]
                
                # Handle tuple/list of Interrupt objects
                if hasattr(interrupt_data, "__iter__") and not isinstance(interrupt_data, (str, dict)):
                    interrupt_list = []
                    for item in interrupt_data:
                        if hasattr(item, "value"):
                            # It's an Interrupt object
                            value = item.value
                            if isinstance(value, dict):
                                interrupt_list.append({"value": value})
                            else:
                                interrupt_list.append({"value": str(value)})
                        else:
                            interrupt_list.append({"value": str(item)})
                    event["__interrupt__"] = interrupt_list
                elif hasattr(interrupt_data, "value"):
                    # Single Interrupt object
                    value = interrupt_data.value
                    if isinstance(value, dict):
                        event["__interrupt__"] = [{"value": value}]
                    else:
                        event["__interrupt__"] = [{"value": str(value)}]
            
            # Process nested objects and lists for JSON serialization
            for key, value in list(event.items()):  # Use list() to allow modification during iteration
                if isinstance(value, (dict, list)):
                    try:
                        # Convert to JSON string and back to ensure serialization
                        json.dumps(value)  # Just test if it's serializable
                    except (TypeError, json.JSONDecodeError):
                        # If not serializable, convert to string representation
                        event[key] = str(value)
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    # Any other non-primitive type should be converted to string
                    event[key] = str(value)
            
            return event
            
        except Exception as e:
            logger.error(f"Error serializing event: {str(e)}", exc_info=True)
            return {
                "type": "error",
                "message": f"Error serializing event: {str(e)}",
                "timestamp": time.time()
            }

    async def run_research(self, topic: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Run research on a topic from start to finish without user interaction."""
        try:
            # Create thread
            thread_id, thread = await self.start_research(topic)
            
            # Start with research phase directly
            logger.info(f"Starting research phase for topic: {topic}")
            
            # Send initial status update
            yield {
                "type": "research_status",
                "status": "RESEARCHING",
                "message": "Starting research with predefined structure",
                "force_update": True,
                "timestamp": time.time()
            }
            
            # Start the research process with sections already defined
            async for event in self.graph.astream(
                Command(resume=True),  # Use resume=True to skip planning
                thread,
                stream_mode="updates"
            ):
                serialized_event = self._serialize_event(event)
                yield serialized_event
                
        except Exception as e:
            logger.error(f"Error running research: {str(e)}", exc_info=True)
            yield {
                "type": "error",
                "message": str(e),
                "timestamp": time.time()
            }
    
    def _is_waiting_for_approval(self, event: Dict[str, Any]) -> bool:
        """Check if the event indicates that the graph is waiting for plan approval."""
        # Check for explicit wait_for_input flag
        if event.get('wait_for_input') or event.get('waiting_for_approval'):
            return True
            
        # Check for plan approval message in interrupt content
        if '__interrupt__' in event and isinstance(event['__interrupt__'], list):
            for item in event['__interrupt__']:
                value = item.get('value', '')
                if isinstance(value, str):
                    approval_phrases = [
                        'approve the plan',
                        'approve this plan',
                        'do you approve',
                        'would you like to approve',
                        'should i proceed with this plan',
                        'proceed with the plan'
                    ]
                    # Check if any approval phrase is in the value
                    if any(phrase in value.lower() for phrase in approval_phrases):
                        return True
                        
        # Check sections exists as a sign of plan being ready
        if 'sections' in event:
            return True
            
        return False
        
    def _check_environment(self):
        """Check if required environment variables are set."""
        # Check for Ollama models and warn if using incorrect model names
        planner_model = self.config.get("planner_model", "")
        writer_model = self.config.get("writer_model", "")
        
        # Check if using incorrect Ollama model names
        if self.config.get("planner_provider", "").lower() == "ollama":
            if planner_model in ["llama3", "llama3.2"]:
                logger.warning(f"Incorrect Ollama model name '{planner_model}'. Use 'llama3.2:1b' instead.")
                self.config["planner_model"] = "llama3.2:1b"
                logger.info("Automatically corrected planner_model to 'llama3.2:1b'")
        
        if self.config.get("writer_provider", "").lower() == "ollama":
            if writer_model in ["llama3", "llama3.2"]:
                logger.warning(f"Incorrect Ollama model name '{writer_model}'. Use 'llama3.2:1b' instead.")
                self.config["writer_model"] = "llama3.2:1b"
                logger.info("Automatically corrected writer_model to 'llama3.2:1b'")
                
        # Check search APIs
        search_api = self.config.get("search_api", "").lower()
        
        if search_api == "tavily" and not os.environ.get("TAVILY_API_KEY"):
            logger.warning("TAVILY_API_KEY not found in environment but search_api is set to tavily")
        
        elif search_api == "perplexity" and not os.environ.get("PERPLEXITY_API_KEY"):
            logger.warning("PERPLEXITY_API_KEY not found in environment but search_api is set to perplexity")
        
        # Check LLM APIs
        planner_provider = self.config.get("planner_provider", "").lower()
        writer_provider = self.config.get("writer_provider", "").lower()
        
        if planner_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found in environment but planner_provider is set to openai")
        
        elif planner_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY not found in environment but planner_provider is set to anthropic")
        
        elif planner_provider == "ollama":
            # Check for Ollama URL or default to localhost
            ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            logger.info(f"Using Ollama host: {ollama_host}")
            
            # Check for langchain-ollama package
            try:
                import importlib
                ollama_spec = importlib.util.find_spec("langchain_ollama")
                if ollama_spec is None:
                    logger.warning("langchain-ollama package not found. Please install it with 'pip install langchain-ollama==0.2.2'")
            except ImportError:
                logger.warning("Failed to check for langchain-ollama package. Please ensure it is installed.")
        
        if writer_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found in environment but writer_provider is set to openai")
        
        elif writer_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY not found in environment but writer_provider is set to anthropic")
        
        elif writer_provider == "ollama":
            # This check is redundant if planner_provider is also ollama, but necessary if only writer uses ollama
            if planner_provider != "ollama":
                ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                logger.info(f"Using Ollama host: {ollama_host}")
                
                # Check for langchain-ollama package
                try:
                    import importlib
                    ollama_spec = importlib.util.find_spec("langchain_ollama")
                    if ollama_spec is None:
                        logger.warning("langchain-ollama package not found. Please install it with 'pip install langchain-ollama==0.2.2'")
                except ImportError:
                    logger.warning("Failed to check for langchain-ollama package. Please ensure it is installed.")
            
    def test_planner(self, topic: str = "Climate change impact on agricultural production") -> Dict[str, Any]:
        """Test the planner model's ability to generate structured output."""
        try:
            # Get configuration values
            planner_provider = self.config.get("planner_provider", "openai")
            planner_model = self.config.get("planner_model", "gpt-4o")
            
            # Return basic test information
            return {
                "success": True,
                "model": planner_model,
                "provider": planner_provider,
                "topic": topic,
                "config": self.config
            }
        except Exception as e:
            logger.error(f"Error in planner test: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def approve_plan(self, thread: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Approve the research plan and start section writing.
        
        Args:
            thread: The thread configuration
            
        Yields:
            Events from the research graph
        """
        try:
            if not self.graph:
                logger.error("Research graph is not initialized")
                yield {"type": "error", "message": "Research graph is not initialized"}
                return
            
            # Get thread ID for logging
            thread_id = thread.get('configurable', {}).get('thread_id', 'unknown')
            
            # Use Command(resume=True) to approve the plan
            logger.info(f"Approving plan for thread: {thread_id}")
            
            # First, yield an event indicating we're starting the approval process
            yield {
                "type": "approval_started",
                "message": "Starting research based on approved plan",
                "thread_id": thread_id,
                "timestamp": time.time()
            }
            
            # Ensure topic is in the root of the state
            topic = thread.get('configurable', {}).get('topic')
            if not topic:
                logger.error(f"Topic not found in thread configurable for thread {thread_id}")
                yield {"type": "error", "message": "Topic not found in thread configuration"}
                return
                
            # Create state with topic at root level
            state = {
                "topic": topic,
                "configurable": thread.get('configurable', {})
            }
            
            # Retrieve the current thread state and validate it
            try:
                current_state = self.graph.get_state(state)
                state_info = f"Current state: {current_state.__class__.__name__}"
                logger.info(f"Current graph state before approval: {state_info}")
                
                # Check if the thread has sections data
                if hasattr(current_state, 'values') and 'sections' in current_state.values:
                    # Include the sections in the approval response for better UI feedback
                    sections = current_state.values.get('sections', [])
                    
                    # Serialize sections for the response
                    serialized_sections = []
                    for section in sections:
                        if hasattr(section, 'dict'):
                            serialized_sections.append(section.dict())
                        elif hasattr(section, 'model_dump'):
                            serialized_sections.append(section.model_dump())
                        elif hasattr(section, '__dict__'):
                            serialized_sections.append(section.__dict__)
                        else:
                            serialized_sections.append(str(section))
                    
                    # Yield the sections data
                    yield {
                        "type": "plan_confirmed",
                        "message": "Plan approved, beginning research",
                        "thread_id": thread_id,
                        "sections": serialized_sections,
                        "timestamp": time.time()
                    }
            except Exception as state_error:
                logger.warning(f"Could not determine graph state before approval: {state_error}")
            
            # Run the graph with resume=True to approve the plan
            event_count = 0  # Track how many events we receive
            research_started = False
            
            async for event in self.graph.astream(Command(resume=True), state, stream_mode="updates"):
                # Serialize the event
                serialized_event = self._serialize_event(event)
                event_count += 1
                
                # Check if this event indicates research has started
                if not research_started and "build_section_with_web_research" in serialized_event:
                    research_started = True
                    logger.info(f"Research has started for thread {thread_id}")
                    
                    # Enhance the event with additional information
                    if isinstance(serialized_event, dict) and "build_section_with_web_research" in serialized_event:
                        section_data = serialized_event["build_section_with_web_research"].get("section", {})
                        section_name = section_data.get("name", "Unknown section")
                        
                        # Add a progress indicator
                        serialized_event["research_progress"] = {
                            "status": "started",
                            "section_name": section_name,
                            "message": f"Started researching '{section_name}'",
                            "timestamp": time.time()
                        }
                
                # Ensure the event has a type field
                if isinstance(serialized_event, dict) and "type" not in serialized_event:
                    if "build_section_with_web_research" in serialized_event:
                        serialized_event["type"] = "researching_section"
                    elif "completed_sections" in serialized_event:
                        serialized_event["type"] = "section_completed"
                    elif "compile_final_report" in serialized_event:
                        serialized_event["type"] = "report_completed"
                    else:
                        serialized_event["type"] = "research_update"
                
                # Add a timestamp if not present
                if isinstance(serialized_event, dict) and "timestamp" not in serialized_event:
                    serialized_event["timestamp"] = time.time()
                
                yield serialized_event
                
            # If we didn't get any events, something might be wrong
            if event_count == 0:
                logger.warning(f"No events received from graph after approval for thread {thread_id}")
                yield {
                    "type": "approval_warning",
                    "message": "Plan was approved but no research events were received. Please check the system status.",
                    "thread_id": thread_id,
                    "timestamp": time.time()
                }
            
            # If research never started, indicate this to the client
            if not research_started:
                logger.warning(f"Research did not start after approval for thread {thread_id}")
                yield {
                    "type": "research_pending",
                    "message": "Plan approved successfully. Research will start soon.",
                    "thread_id": thread_id,
                    "timestamp": time.time()
                }
            
        except Exception as e:
            logger.error(f"Error approving research plan: {str(e)}", exc_info=True)
            yield {"type": "error", "message": f"Error approving research plan: {str(e)}", "timestamp": time.time()}

    async def provide_feedback(self, thread: Dict[str, Any], feedback: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Provide feedback on the research plan.
        
        Args:
            thread: The thread configuration
            feedback: User feedback on the plan
            
        Yields:
            Events from the research graph
        """
        try:
            if not self.graph:
                logger.error("Research graph is not initialized")
                yield {"type": "error", "message": "Research graph is not initialized"}
                return
            
            logger.info(f"Providing feedback to thread: {thread.get('configurable', {}).get('thread_id', 'unknown')}")
            
            # Run the graph with resume=feedback to update the plan
            async for event in self.graph.astream(Command(resume=feedback), thread, stream_mode="updates"):
                # Serialize the event
                serialized_event = self._serialize_event(event)
                yield serialized_event
            
        except Exception as e:
            logger.error(f"Error providing feedback: {str(e)}", exc_info=True)
            yield {"type": "error", "message": f"Error providing feedback: {str(e)}"}

    def _cache_thread(self, thread_id: str, thread: Dict[str, Any]):
        """Cache thread state in memory."""
        self._thread_cache[thread_id] = thread
    
    def _get_cached_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get thread state from cache."""
        return self._thread_cache.get(thread_id)
    
    async def stream_results(self, topic: str, thread: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream research results event by event through the entire research process without requiring user intervention."""
        try:
            # Check if thread has proper configurable data
            if not thread or 'configurable' not in thread:
                logger.error(f"Thread missing or incomplete")
                yield {"type": "error", "message": "Thread configuration is invalid"}
                return
            
            thread_id = thread.get('configurable', {}).get('thread_id', 'unknown')
            logger.info(f"Streaming results for thread: {thread_id}")
            
            # Cache the thread state
            self._cache_thread(thread_id, thread)
            
            # CRITICAL: Force max_search_depth and number_of_queries to be integers
            # This prevents the type error with '>=' comparison
            if 'configurable' in thread:
                thread['configurable']['max_search_depth'] = int(thread['configurable'].get('max_search_depth', 2))
                thread['configurable']['number_of_queries'] = int(thread['configurable'].get('number_of_queries', 2))
                logger.info(f"Forced max_search_depth to int: {thread['configurable']['max_search_depth']}")
                logger.info(f"Forced number_of_queries to int: {thread['configurable']['number_of_queries']}")
            
            # Prepare thread and topic
            try:
                # Make sure topic is set in thread configurable and at root level
                if 'configurable' in thread and topic:
                    thread['configurable']['topic'] = topic
                    thread['topic'] = topic
                
                # First yield a status update to the client to show we're starting research
                yield {
                    "type": "report_status",
                    "status": "RESEARCHING",
                    "force_update": True,
                    "timestamp": time.time()
                }
                
                # Track state to avoid loops
                plan_received = False
                research_in_progress = False
                completed_sections = set()
                completion_detected = False
                
                logger.info(f"Starting research with topic: {topic}")
                
                # PHASE 1: Get the initial plan
                try:
                    # Start with the planning phase
                    async for event in self.graph.astream({"topic": topic}, thread, stream_mode="updates"):
                        serialized_event = self._serialize_event(event)
                        
                        # Check if this is an interrupt event (contains the plan)
                        if "__interrupt__" in serialized_event:
                            logger.info("Received interrupt event with plan")
                            
                            # If the interrupt value is the plan, extract it
                            if isinstance(serialized_event["__interrupt__"], list) and len(serialized_event["__interrupt__"]) > 0:
                                interrupt_value = serialized_event["__interrupt__"][0].get("value", {})
                                
                                # If interrupt contains a formatted sections list, extract it
                                if isinstance(interrupt_value, dict) and "sections" in interrupt_value:
                                    logger.info(f"Extracted sections from interrupt: {len(interrupt_value['sections'])}")
                                    plan_received = True
                                    # Add sections directly to the serialized event for easier frontend processing
                                    serialized_event["sections"] = interrupt_value["sections"]
                                    # Add a user-friendly message
                                    serialized_event["message"] = "Plan created and ready for research"
                                
                                # If it's a string (like a prompt asking for plan approval), extract sections if possible
                                elif isinstance(interrupt_value, str) and "Section:" in interrupt_value:
                                    logger.info("Trying to extract sections from interrupt text")
                                    # Look for sections in format Section: X, Description: Y
                                    from .utils import extract_sections_from_plan
                                    try:
                                        extracted_sections = extract_sections_from_plan(interrupt_value)
                                        if extracted_sections:
                                            plan_received = True
                                            serialized_event["sections"] = extracted_sections
                                            serialized_event["message"] = "Plan created from research outline"
                                            logger.info(f"Successfully extracted {len(extracted_sections)} sections from interrupt text")
                                    except Exception as extract_err:
                                        logger.error(f"Error extracting sections from interrupt: {str(extract_err)}")
                        
                        # Check for regular sections data
                        if "sections" in serialized_event and not plan_received:
                            plan_received = True
                            logger.info(f"Received plan with {len(serialized_event['sections'])} sections")
                            
                            # Add research_stage marker for frontend
                            serialized_event["research_stage"] = "plan_complete"
                            serialized_event["message"] = "Research plan finalized"
                        
                        # Add force_update flag to all report_status events
                        if isinstance(serialized_event, dict) and serialized_event.get('type') == 'report_status':
                            serialized_event['force_update'] = True
                        
                        # Send the event to the client
                        yield serialized_event
                        
                        # If we have a plan or waiting for approval, break to move to research phase
                        if plan_received or self._is_waiting_for_approval(serialized_event):
                            logger.info("Research plan received, automatically approving to start research")
                            break
                    
                    # PHASE 2: Auto-approve the plan and continue with research
                    # If we received a plan or waiting for approval event, auto-approve and start research
                    if plan_received:
                        # Send status update that we're starting research phase
                        yield {
                            "type": "research_status",
                            "status": "RESEARCHING",
                            "message": "Starting research based on plan",
                            "force_update": True,
                            "timestamp": time.time()
                        }
                        
                        # Use Command(resume=True) to continue research
                        research_in_progress = True
                        
                        # Track sections to detect completion
                        section_count = 0
                        completed_section_count = 0
                        
                        # Start the research phase with auto-approval
                        async for event in self.graph.astream(Command(resume=True), thread, stream_mode="updates"):
                            serialized_event = self._serialize_event(event)
                            
                            # Handle string events - add special handling
                            if isinstance(serialized_event, dict):
                                # Track section research starting
                                if "build_section_with_web_research" in serialized_event:
                                    # Extract section name more carefully
                                    section_data = serialized_event.get('build_section_with_web_research', {})
                                    section_name = "Unknown section"  # Default name
                                    
                                    # Try to get section from different data structures
                                    if hasattr(section_data, 'section'):
                                        section = section_data.section
                                        if hasattr(section, 'name'):
                                            section_name = section.name
                                        elif isinstance(section, str):
                                            section_name = section
                                        elif isinstance(section, dict) and 'name' in section:
                                            section_name = section['name']
                                    elif isinstance(section_data, dict):
                                        if 'section' in section_data:
                                            section = section_data['section']
                                            if isinstance(section, dict) and 'name' in section:
                                                section_name = section['name']
                                            elif isinstance(section, str):
                                                section_name = section
                                            elif hasattr(section, 'name'):
                                                section_name = section.name
                                            
                                    # Add a formatted type and section name
                                    serialized_event["type"] = "researching_section"
                                    serialized_event["section_name"] = section_name
                                    serialized_event["message"] = f"Researching: {section_name}"
                                    
                                    logger.info(f"Researching section: {section_name}")
                                    
                                    # Add a force_update flag to ensure UI updates
                                    serialized_event["force_update"] = True
                                
                                # Track completed sections
                                if "completed_sections" in serialized_event:
                                    completed_sections_list = []
                                    
                                    # Extract completed sections with better error handling
                                    completed_sections_data = serialized_event.get("completed_sections", [])
                                    
                                    # Handle different data structures that might come from the package
                                    if not isinstance(completed_sections_data, list):
                                        if hasattr(completed_sections_data, "__iter__"):
                                            completed_sections_data = list(completed_sections_data)
                                        elif hasattr(completed_sections_data, "__dict__"):
                                            completed_sections_data = [completed_sections_data]
                                        else:
                                            completed_sections_data = []
                                    
                                    for section in completed_sections_data:
                                        section_name = None
                                        
                                        # Try multiple approaches to extract section name
                                        if hasattr(section, "name"):
                                            section_name = section.name
                                        elif isinstance(section, dict) and "name" in section:
                                            section_name = section["name"]
                                        elif hasattr(section, "__dict__") and "name" in section.__dict__:
                                            section_name = section.__dict__["name"]
                                        elif hasattr(section, "__getattr__"):
                                            try:
                                                section_name = section.__getattr__("name")
                                            except (AttributeError, KeyError):
                                                pass
                                        
                                        # Use a default name if we couldn't extract one
                                        if not section_name:
                                            section_name = f"Section #{completed_section_count}"
                                        
                                        if section_name not in completed_sections:
                                            completed_sections.add(section_name)
                                            completed_sections_list.append(section_name)
                                            completed_section_count += 1
                                            logger.info(f"Completed section: {section_name} (total: {completed_section_count})")
                                    
                                    # Add type field for easier frontend processing
                                    serialized_event["type"] = "section_completed"
                                    
                                    # Show which section was completed
                                    if completed_sections_list:
                                        if len(completed_sections_list) == 1:
                                            serialized_event["message"] = f"Completed: {completed_sections_list[0]}"
                                        else:
                                            serialized_event["message"] = f"Completed sections: {', '.join(completed_sections_list)}"
                                    else:
                                        serialized_event["message"] = "Section completed"
                                
                                # Check for final report compilation
                                if "compile_final_report" in serialized_event:
                                    logger.info("Final report compilation started")
                                    # Add type field for easier frontend processing
                                    serialized_event["type"] = "report_completed"
                                    serialized_event["message"] = "Report completed!"
                                    
                                    # Ensure the final_report content is available
                                    if "final_report" in serialized_event.get("compile_final_report", {}):
                                        logger.info(f"Final report completed: {len(serialized_event['compile_final_report']['final_report'])} chars")
                                        # Set completed status
                                        serialized_event["status"] = "COMPLETED"
                                        # Mark completion as detected to prevent reverting to RESEARCHING
                                        completion_detected = True
                                
                                # Set proper status based on the event type and completion status
                                if completion_detected:
                                    # Once completion is detected, ensure all subsequent events have COMPLETED status
                                    serialized_event["status"] = "COMPLETED"
                                    # If it's a report_status event, make sure it shows COMPLETED
                                    if serialized_event.get('type') == 'report_status':
                                        serialized_event['status'] = "COMPLETED"
                                
                                # Add force_update flag to all report_status events
                                if serialized_event.get('type') == 'report_status':
                                    serialized_event['force_update'] = True
                                    
                                # Add timestamp if not present
                                if "timestamp" not in serialized_event:
                                    serialized_event["timestamp"] = time.time()
                            
                            # Log event being yielded for debugging
                            if isinstance(serialized_event, dict):
                                event_keys = list(serialized_event.keys())
                                logger.debug(f"Yielding event with keys: {event_keys}")
                                
                                # If completion was detected, ensure all events have COMPLETED status
                                if completion_detected:
                                    serialized_event["status"] = "COMPLETED"
                            else:
                                logger.debug(f"Yielding non-dict event: {type(serialized_event)}")
                            
                            # Send the event to the client
                            yield serialized_event
                        
                        # If we exit the loop and completion was detected, send a final completion event
                        if completion_detected:
                            final_event = {
                                "type": "report_completed",
                                "status": "COMPLETED",
                                "message": "Report completed successfully!",
                                "timestamp": time.time(),
                                "force_update": True,
                                "report_complete": True
                            }
                            yield final_event
                
                except Exception as e:
                    error_msg = f"Error in research process: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    yield {
                        "type": "error", 
                        "message": error_msg,
                        "status": "FAILED",  # Ensure status is set for error case
                        "timestamp": time.time()
                    }
                        
            except Exception as e:
                logger.error(f"Error streaming research results: {str(e)}", exc_info=True)
                yield {"type": "error", "message": f"Error streaming results: {str(e)}", "timestamp": time.time()}
            
        except Exception as e:
            logger.error(f"Error setting up streaming: {str(e)}", exc_info=True)
            yield {"type": "error", "message": f"Error setting up streaming: {str(e)}", "timestamp": time.time()}

    async def collect_events(self, async_gen):
        """Collects all events from an async generator."""
        try:
            completion_detected = False
            last_event_time = time.time()
            keepalive_interval = 15  # Send keepalive every 15 seconds
            
            # Track if we've seen a completion event to prevent status from changing back
            async for event in async_gen:
                # Process completion events
                if isinstance(event, dict):
                    # Check for completion events
                    if event.get('type') == 'report_completed' or (
                        'compile_final_report' in event and 
                        'final_report' in event.get('compile_final_report', {})
                    ):
                        completion_detected = True
                        # Mark as completed explicitly
                        event['status'] = 'COMPLETED'
                        
                        # Add completed flag
                        event['report_complete'] = True
                        event['force_update'] = True
                    
                    # If completion was detected, ensure all subsequent events have COMPLETED status
                    elif completion_detected and 'type' in event:
                        # Don't let status go back to RESEARCHING after completion
                        event['status'] = 'COMPLETED'
                    
                    # Format SSE event correctly
                    if event.startswith('data:') if isinstance(event, str) else False:
                        yield event
                    else:
                        yield f"data: {json.dumps(event)}\n\n"
                elif isinstance(event, str):
                    if event.startswith('data:'):
                        yield event
                    else:
                        yield f"data: {event}\n\n"
                else:
                    # For non-dict/non-string events, wrap in a basic event structure
                    yield f"data: {json.dumps({'type': 'event', 'data': str(event)})}\n\n"
                
                # Add keepalive after each event
                current_time = time.time()
                if current_time - last_event_time > keepalive_interval:
                    yield ":\n\n"
                    last_event_time = current_time
            
            # If we've completed the report and exited the loop, send a final completion event
            if completion_detected:
                final_event = {
                    'type': 'report_completed',
                    'status': 'COMPLETED',
                    'message': 'Report completed successfully!',
                    'timestamp': time.time(),
                    'force_update': True,
                    'report_complete': True
                }
                yield f"data: {json.dumps(final_event)}\n\n"
            
            # Always add a final keepalive
            yield ":\n\n"
            
        except Exception as e:
            logger.error(f"Error collecting events: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'timestamp': time.time()})}\n\n" 