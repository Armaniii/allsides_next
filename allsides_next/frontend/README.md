# AllSides Next - Frontend

## Overview

The frontend of AllSides Next is built with Next.js, React, TypeScript, and Tailwind CSS. It provides an intuitive user interface for interacting with the diverse perspective generation system and visualizing the arguments across the political spectrum.

## Architecture

The application follows the Next.js App Router architecture with a focus on:

- Type-safe development with TypeScript
- Component-based UI with React
- Responsive styling with Tailwind CSS
- Animation with Framer Motion
- API client with Axios

## Directory Structure

```
frontend/
├── src/
│   ├── app/               # Next.js app router pages
│   ├── components/        # Reusable UI components
│   ├── contexts/          # React context providers
│   ├── lib/               # Utility functions and type definitions
│   ├── services/          # API client and services
│   └── utils/             # Helper utilities
├── public/                # Static assets
└── tailwind.config.js     # Tailwind configuration
```

## Key Components

### Page Components

- **HomePage (`src/app/page.tsx`)**: Main interface for query submission and viewing arguments
- **Login (`src/app/login/page.tsx`)**: User authentication page
- **Dashboard (`src/app/dashboard/page.tsx`)**: User statistics and profile management

### UI Components

- **ArgumentCard**: Displays individual arguments with their political stance
- **SupportingArgumentsModal**: Shows detailed supporting arguments for a given perspective
- **QueryHistoryCard**: Displays past user queries and their results
- **LeaderboardModal**: Shows top users based on their engagement

### Services and Hooks

- **API Client (`src/services/api.ts`)**: Handles communication with the backend API
- **Authentication Context (`src/contexts/AuthContext.tsx`)**: Manages user authentication state

## Features

1. **Query Submission**: Users can submit topics to get diverse perspectives on
2. **Diversity Control**: Slider to control how diverse the returned perspectives should be
3. **Argument Visualization**: Clear display of arguments across the political spectrum
4. **Query History**: Access to previous queries and their results
5. **Argument Rating**: Ability to rate arguments on the political spectrum
6. **Responsive Design**: Works on desktop and mobile devices

## Development Setup

### Prerequisites

- Node.js 16+
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Configure environment variables (create `.env.local` file):
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

3. Start the development server:
```bash
npm run dev
# or
yarn dev
```

4. Build for production:
```bash
npm run build
# or
yarn build
```

## Docker Deployment

The frontend is containerized using Docker for easy deployment:

```bash
# Build and run the frontend container
docker build -t allsides-frontend .
docker run -p 3000:3000 allsides-frontend
```

## UI Components Library

The application uses a combination of custom components and the following libraries:

- **Tailwind CSS**: Utility-first CSS framework
- **Heroicons**: Icon library
- **Framer Motion**: Animation library
- **React Query**: Data fetching and state management
