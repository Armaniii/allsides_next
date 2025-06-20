# AllSides Next Project Overview

AllSides Next is a multi-tier web application that generates diverse political perspectives on user queries. It's designed to provide arguments from multiple political stances (Left, Lean Left, Center, Lean Right, Right), promoting comprehensive understanding of complex issues.

## Core Purpose
- Generate diverse political perspectives on any topic using AI
- Present balanced viewpoints across the political spectrum
- Allow users to rate arguments and provide feedback
- Track user engagement through an "AllStars" leaderboard system

## Architecture Overview
- **Frontend**: Next.js React application with TypeScript
- **Backend**: Django REST API with PostgreSQL database
- **Infrastructure**: Docker Compose orchestration with:
  - Nginx reverse proxy
  - Redis cache
  - pgAdmin for database management
  - Ollama for local LLM capabilities
  - vLLM for model serving
- **Logging**: LangFuse integration for API and LLM call tracking
- **Research Module**: LangGraph integration for deep research reports

## Key Features
1. **Query Formatting**: Transforms topics into well-formed questions
2. **Follow-up Questions**: Generates 4-5 intelligent follow-up questions
3. **Core Argument Summarization**: Creates 2-3 sentence summaries for each position
4. **Source Analysis**: Comprehensive source credibility and bias detection
5. **User Account System**: Personalized experience with query history
6. **Rating System**: Users can rate arguments and provide feedback
7. **Responsive Design**: Works on desktop and mobile devices

## Development Status
The project has recently completed implementation of all local LLM enhancement features (as per main_v3_progress.md). The system is production-ready with comprehensive error handling and fallback mechanisms.