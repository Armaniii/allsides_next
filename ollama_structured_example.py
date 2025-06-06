from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from typing import List

# Define your output structure
class Research(BaseModel):
    """Information from research."""
    findings: List[str] = Field(description="List of research findings")
    summary: str = Field(description="Summary of the research")

# Initialize the parser
parser = PydanticOutputParser(pydantic_object=Research)

# Initialize the model
model = ChatOllama(
    model="your_model_name",  # Replace with your actual model name
    temperature=0,
    format="json"  # This is crucial for structured output
)

# Create a prompt template that includes instructions for formatting
template = """
Please conduct research on the following topic: {topic}

{format_instructions}

Return the results in the requested JSON format.
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create the chain
chain = prompt | model | parser

# Example usage
try:
    result = chain.invoke({"topic": "Climate change effects on agriculture"})
    print("Successfully parsed result:")
    print(f"Summary: {result.summary}")
    print(f"Findings: {result.findings}")
except Exception as e:
    print(f"Error: {e}")
    
    # If you need to debug, run without the parser to see the raw output
    simple_chain = prompt | model | StrOutputParser()
    raw_output = simple_chain.invoke({"topic": "Climate change effects on agriculture"})
    print("\nRaw output from model:")
    print(raw_output) 