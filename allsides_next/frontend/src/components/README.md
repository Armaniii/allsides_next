# AllSides Next - Frontend Components

## Overview

This directory contains the React components used throughout the AllSides Next application. The components follow a modular design pattern, with reusable UI elements that can be composed to build complex interfaces.

## Component Architecture

The components are organized into the following categories:

- **Page Components**: Main interfaces for different pages
- **Functional Components**: Components with specific business logic
- **UI Components**: Reusable UI elements and controls
- **Layout Components**: Components for structuring the page layout

## Key Components

### Argument Display Components

| Component | Description |
|-----------|-------------|
| `ArgumentCard.tsx` | Displays a single argument with its political stance and core content |
| `ArgumentsDisplay.tsx` | Container component that renders multiple ArgumentCards |
| `StreamingArgumentCard.tsx` | Variant of ArgumentCard that supports streaming content with animations |

### Modal Components

| Component | Description |
|-----------|-------------|
| `SupportingArgumentsModal.tsx` | Displays detailed supporting arguments for a core argument |
| `LeaderboardModal.tsx` | Shows top users based on their AllStars points |

### Input Components

| Component | Description |
|-----------|-------------|
| `QueryInput.tsx` | Input field for submitting queries with validation |

### History Components

| Component | Description |
|-----------|-------------|
| `QueryHistoryCard.tsx` | Displays past queries and provides options to view or delete them |

### UI Components

The `ui/` subdirectory contains basic UI elements built on Tailwind CSS:

- `Button.tsx`: Various button styles
- `Card.tsx`: Container component with consistent styling
- `Input.tsx`: Form input elements
- `Slider.tsx`: Range slider component
- `Toaster.tsx` and related components: Toast notification system

## Component Interfaces

### ArgumentCard

```typescript
interface ArgumentCardProps {
  stance: string;
  coreArgument: string;
  supportingArguments: string[];
  onRateClick?: (stance: string, coreArgument: string) => void;
  onThumbsUp?: () => void;
  onThumbsDown?: () => void;
  expanded?: boolean;
  onToggleExpand?: () => void;
}
```

### SupportingArgumentsModal

```typescript
interface SupportingArgumentsModalProps {
  isOpen: boolean;
  onClose: () => void;
  stance: string;
  coreArgument: string;
  supportingArguments: string[];
}
```

### QueryHistoryCard

```typescript
interface QueryHistoryCardProps {
  query: Query;
  onDelete: (id: number) => void;
  onSelect: (query: Query) => void;
  searchTerm?: string;
  isExpanded: boolean;
  onToggleExpand: () => void;
}
```

## Component Styling

Components use a combination of:

- **Tailwind CSS**: Utility classes for responsive styling
- **CSS Modules**: Component-specific styles when needed
- **Framer Motion**: For animations and transitions

### Color Scheme

The application follows a consistent color scheme for political stances:

- **Left**: Blue shades
- **Lean Left**: Light blue
- **Center**: Purple
- **Lean Right**: Light red
- **Right**: Red shades

## Component Structure Example

Most components follow this structure:

```tsx
// Imports
import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

// Interface definition
interface ExampleComponentProps {
  // Props definition
}

// Component implementation
export const ExampleComponent: React.FC<ExampleComponentProps> = ({
  // Props destructuring
}) => {
  // Local state and hooks
  
  // Handler functions
  
  // Render
  return (
    <motion.div
      // Animation properties
      className={cn(
        // Base styles
        'common-styles',
        // Conditional styles
        {
          'active-style': isActive,
        }
      )}
    >
      {/* Component content */}
    </motion.div>
  );
};
```

## Usage Examples

### Basic Usage

```tsx
import { ArgumentCard } from '@/components/ArgumentCard';

const MyComponent = () => {
  return (
    <ArgumentCard
      stance="Left"
      coreArgument="Government should provide universal healthcare."
      supportingArguments={[
        "Healthcare is a human right.",
        "Universal coverage reduces overall costs.",
        "No one should go bankrupt due to medical bills."
      ]}
      onRateClick={(stance, argument) => {
        // Handle rating
      }}
    />
  );
};
```

### With Container Components

```tsx
import { ArgumentsDisplay } from '@/components/ArgumentsDisplay';

const MyPage = () => {
  const arguments = [
    // Array of arguments
  ];
  
  return (
    <ArgumentsDisplay 
      arguments={arguments}
      onRateClick={handleRateClick}
    />
  );
};
```

## Development Guidelines

When creating or modifying components:

1. **Maintain TypeScript Type Safety**: All components should have proper TypeScript interfaces
2. **Ensure Reusability**: Components should be designed for reuse with appropriate props
3. **Follow Accessibility Best Practices**: Use semantic HTML and ARIA attributes
4. **Optimize for Performance**: Use memoization and avoid unnecessary re-renders
5. **Maintain Consistent Styling**: Follow the established design patterns and color schemes 