# AllSides Next

## Overview

AllSides Next is an application designed to provide diverse perspectives on various topics through AI-generated arguments. It presents users with a range of viewpoints across the political spectrum (from Left to Right) on any given query, promoting a more comprehensive understanding of complex issues.

## Key Features

- **Diverse Perspective Generation**: Generates arguments from multiple political stances (Left, Lean Left, Center, Lean Right, Right)
- **User Account System**: Personalized experience with query history and usage limits
- **Rating System**: Users can rate arguments and provide feedback
- **Leaderboard**: Tracks user engagement through an "AllStars" system
- **Responsive Design**: Works on both desktop and mobile devices

## Project Structure

The project follows a modern microservices architecture:

- **Frontend**: Next.js application with React and TypeScript
- **Backend**: Django REST API
- **Database**: PostgreSQL
- **Infrastructure**: Docker-based deployment with Nginx

### Directory Structure

```
allsides_next/
├── frontend/      # Next.js frontend application
├── backend/       # Django backend API
├── nginx/         # Nginx configuration
├── certbot/       # SSL certificate management
└── docker-compose.yml  # Docker configuration
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Node.js (for local frontend development)
- Python 3.9+ (for local backend development)

### Installation

1. Clone the repository
2. Configure environment variables in `.env` files
3. Run the application using Docker Compose:

```bash
docker-compose up -d
```

## Development

For detailed information about the frontend and backend components, see their respective README files:

- [Frontend Documentation](./allsides_next/frontend/README.md)
- [Backend Documentation](./allsides_next/backend/README.md)

## License

Proprietary - All rights reserved 