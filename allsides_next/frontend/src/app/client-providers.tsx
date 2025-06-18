'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from '@/contexts/AuthContext';
import { useState } from 'react';

export default function ClientProviders({
  children,
}: {
  children: React.ReactNode;
}) {
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        // Increase timeout to match backend processing time (5 minutes)
        staleTime: 1000 * 60 * 5, // 5 minutes
        // Prevent automatic retries for failed requests to avoid multiple long requests
        retry: 1,
        // Increase timeout for individual queries
        refetchOnWindowFocus: false,
        // Increase query timeout significantly
        gcTime: 1000 * 60 * 10, // 10 minutes garbage collection
      },
      mutations: {
        // Set mutation timeout to 5 minutes to match backend processing
        retry: 0, // Don't retry mutations automatically
        // Force mutations to use network (no cache interference)
        networkMode: 'always',
        // Increase mutation timeout
        gcTime: 1000 * 60 * 10, // 10 minutes garbage collection
      }
    }
  }));

  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        {children}
      </AuthProvider>
    </QueryClientProvider>
  );
} 