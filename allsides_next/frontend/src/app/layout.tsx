'use client';

import localFont from 'next/font/local';
import './globals.css';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from '@/contexts/AuthContext';
import { useState } from 'react';

const etBook = localFont({
  src: [
    {
      path: '../../public/fonts/et-book/et-book-roman.woff',
      weight: '400',
      style: 'normal',
    },
    {
      path: '../../public/fonts/et-book/et-book-bold.woff',
      weight: '700',
      style: 'normal',
    },
    {
      path: '../../public/fonts/et-book/et-book-italic.woff',
      weight: '400',
      style: 'italic',
    },
  ],
  display: 'swap',
  variable: '--font-et-book',
});

// export const metadata = {
//   title: 'AllSides',
//   description: 'Explore diverse perspectives on important topics',
// };

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        retry: 1,
        refetchOnWindowFocus: false,
      },
    },
  }));

  return (
    <html lang="en" className={etBook.variable} suppressHydrationWarning>
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className={etBook.className} suppressHydrationWarning>
        <QueryClientProvider client={queryClient}>
          <AuthProvider>
            <main className="min-h-screen">
              {children}
            </main>
          </AuthProvider>
        </QueryClientProvider>
      </body>
    </html>
  );
}
