# AllSides Deep Research Module

This module integrates with the Open Deep Research library to generate comprehensive research reports on various topics using LLMs like GPT-4o.

## Setup

### Environment Variables

The research module requires several API keys to function properly. These should be set in the `.env` file in the backend directory. The required keys depend on which search and LLM providers you plan to use.

#### Required API Keys:

1. **LLM Provider Keys** (at least one):
   - `OPENAI_API_KEY` - For OpenAI models (GPT-4o, etc.)
   - `ANTHROPIC_API_KEY` - For Anthropic models (Claude)
   - `GROQ_API_KEY` - For Groq models

2. **Search API Keys** (at least one):
   - `TAVILY_API_KEY` - For Tavily search
   - `LINKUP_API_KEY` - For LinkUp search
   - `GOOGLE_API_KEY` - For Google search
   - `PERPLEXITY_API_KEY` - For Perplexity search
   - `EXA_API_KEY` - For Exa search

**Important Note**: LangGraph itself does not require a separate API key. It uses the model and search API keys specified above.

#### Example .env configuration:

```
# Research API Keys
OPENAI_API_KEY="your-openai-api-key"
TAVILY_API_KEY="your-tavily-api-key"

# Model Configuration
DEFAULT_PLANNER_PROVIDER="openai"
DEFAULT_PLANNER_MODEL="gpt-4o"
DEFAULT_WRITER_PROVIDER="openai" 
DEFAULT_WRITER_MODEL="gpt-4o"

# Search Configuration
DEFAULT_SEARCH_API="tavily"
MAX_SEARCH_DEPTH=2
NUMBER_OF_QUERIES=2
```

### Installation

The module dependencies should be automatically installed when you install the backend requirements:

```bash
pip install -r requirements.txt
```

## Authentication

All research API endpoints require authentication. The frontend service uses JWT tokens for authentication:

- The token is stored in localStorage as `accessToken` 
- All API requests must include the token in the Authorization header: `Authorization: Bearer <token>`
- If you encounter 401 Unauthorized errors, check:
  1. That you're properly logged in
  2. That the token is correctly stored and retrieved
  3. That the token hasn't expired

## Usage

The research module exposes several REST endpoints:

- `POST /api/research/reports/` - Create a new research report
- `GET /api/research/reports/` - List all research reports
- `GET /api/research/reports/{id}/` - Get a specific research report
- `POST /api/research/reports/{id}/feedback/` - Provide feedback on a research plan
- `POST /api/research/reports/{id}/approve/` - Approve a research plan
- `GET /api/research/reports/{id}/stream/` - Stream updates for a research report

For detailed API documentation, refer to the API documentation pages in the frontend. 