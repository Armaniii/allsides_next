# AllSides Next

## Overview

AllSides Next is an application designed to provide diverse perspectives on various topics through AI-generated arguments. It presents users with a range of viewpoints across the political spectrum (from Left to Right) on any given query, promoting a more comprehensive understanding of complex issues.

## Key Features

- **Diverse Perspective Generation**: Generates arguments from multiple political stances (Left, Lean Left, Center, Lean Right, Right)
- **Local LLM Enhancement**: Query formatting, follow-up questions, core argument summarization, and source analysis
- **User Account System**: Personalized experience with query history and usage limits
- **Rating System**: Users can rate arguments and provide feedback
- **Leaderboard**: Tracks user engagement through an "AllStars" system
- **Source Credibility Analysis**: AI-powered source trust scoring and bias detection
- **Responsive Design**: Works on both desktop and mobile devices

## Project Structure

The project follows a modern microservices architecture:

- **Frontend**: Next.js application with React and TypeScript
- **Backend**: Django REST API with Gunicorn
- **Database**: PostgreSQL with pgAdmin management
- **Caching**: Redis for performance optimization
- **AI/ML Services**: Ollama and vLLM for local LLM capabilities
- **Infrastructure**: Docker Compose orchestration with Nginx reverse proxy
- **Monitoring**: LangFuse for LLM usage tracking

### Directory Structure

```
allsides_next_project/
├── allsides_next/
│   ├── frontend/          # Next.js React application
│   └── backend/           # Django REST API
├── nginx/                 # Nginx reverse proxy configuration
├── backups/              # Database backup storage
├── docker-compose.yml    # Main Docker orchestration
├── docker-compose.aurora.yml # AWS Aurora deployment option
├── CLAUDE.md             # AI assistant instructions
├── TODO.md               # Active task tracking
└── LOCAL_LLM_INTEGRATION.md # LLM setup documentation
```

## Getting Started

### Prerequisites

- Docker Engine 20.10+ and Docker Compose 2.0+
- 16GB+ RAM recommended (for AI/ML services)
- 50GB+ free disk space
- Node.js 16+ (for local frontend development)
- Python 3.11+ (for local backend development)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd allsides_next_project
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start all services**
   ```bash
   docker compose up -d
   ```

4. **Verify deployment**
   ```bash
   docker compose ps
   # Check service health
   curl http://localhost:11434/api/tags  # Ollama
   curl http://localhost:8001/v1/models  # vLLM
   ```

5. **Access the application**
   - Main application: http://localhost:9000
   - pgAdmin: http://localhost:5050

## Development

For detailed information about the frontend and backend components, see their respective README files:

- [Frontend Documentation](./allsides_next/frontend/README.md)
- [Backend Documentation](./allsides_next/backend/README.md)

## License

Proprietary - All rights reserved 