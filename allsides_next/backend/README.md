# AllSides Next - Backend

## Overview

The backend of AllSides Next is built with Django and Django REST Framework, providing a robust API that powers the application's perspective generation, user authentication, and data persistence features.

## Architecture

The backend follows a standard Django project structure with the following key components:

- **API App**: Handles requests related to queries, arguments, and user interactions
- **Core**: Contains project-wide settings, URLs, and middleware
- **Authentication**: JWT-based authentication system for secure user sessions

## Key Components

### Models (`api/models.py`)

- **User**: Extended Django user model with bias ratings and query limits
- **Query**: Stores user queries and their response data
- **ArgumentRating**: Tracks user ratings of arguments on the political spectrum
- **ThumbsRating**: Captures thumbs up/down feedback on arguments
- **CacheStatistics**: Monitors performance of the caching system

### Views (`api/views.py`)

Implements API endpoints for:
- User authentication and profile management
- Query submission and retrieval
- Argument rating and feedback
- User statistics

### API Endpoints

The main API endpoints include:

- `/api/queries/`: Submit and retrieve queries
- `/api/auth/`: User authentication (login, register, refresh)
- `/api/ratings/`: Submit argument ratings
- `/api/stats/`: Get user statistics
- `/api/leaderboard/`: View user leaderboard

## Development Setup

### Prerequisites

- Python 3.9+
- Poetry (recommended) or pip for dependency management

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables (see `.env.example`)

4. Run migrations:
```bash
python manage.py migrate
```

5. Start the development server:
```bash
python manage.py runserver
```

## Docker Deployment

The backend is containerized using Docker for easy deployment:

```bash
# Build and run the backend container
docker build -t allsides-backend .
docker run -p 8000:8000 allsides-backend
```

## API Documentation

The backend includes automatic API documentation available at `/api/docs/` when running the server. 