'use client';

import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { nord } from 'react-syntax-highlighter/dist/cjs/styles/prism';

interface ResearchMarkdownViewerProps {
  content: string;
  className?: string;
}

const ResearchMarkdownViewer: React.FC<ResearchMarkdownViewerProps> = ({ 
  content,
  className = ''
}) => {
  // Custom renderers for markdown elements
  const components = useMemo(() => ({
    h1: ({ node, ...props }: any) => (
      <h1 className="text-3xl font-bold mt-8 mb-4 text-gray-800 dark:text-white" {...props} />
    ),
    h2: ({ node, ...props }: any) => (
      <h2 className="text-2xl font-bold mt-6 mb-3 text-gray-800 dark:text-white" {...props} />
    ),
    h3: ({ node, ...props }: any) => (
      <h3 className="text-xl font-bold mt-5 mb-2 text-gray-800 dark:text-white" {...props} />
    ),
    p: ({ node, ...props }: any) => (
      <p className="my-4 text-gray-700 dark:text-gray-300 leading-relaxed" {...props} />
    ),
    a: ({ node, ...props }: any) => (
      <a 
        className="text-blue-600 dark:text-blue-400 hover:underline" 
        target="_blank" 
        rel="noopener noreferrer"
        {...props} 
      />
    ),
    ul: ({ node, ...props }: any) => (
      <ul className="list-disc pl-8 my-4 text-gray-700 dark:text-gray-300" {...props} />
    ),
    ol: ({ node, ...props }: any) => (
      <ol className="list-decimal pl-8 my-4 text-gray-700 dark:text-gray-300" {...props} />
    ),
    li: ({ node, ...props }: any) => (
      <li className="my-1 text-gray-700 dark:text-gray-300" {...props} />
    ),
    blockquote: ({ node, ...props }: any) => (
      <blockquote 
        className="border-l-4 border-gray-300 dark:border-gray-600 pl-4 italic my-4 text-gray-600 dark:text-gray-400" 
        {...props} 
      />
    ),
    table: ({ node, ...props }: any) => (
      <div className="overflow-x-auto my-6">
        <table className="min-w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700" {...props} />
      </div>
    ),
    thead: ({ node, ...props }: any) => (
      <thead className="bg-gray-100 dark:bg-gray-700" {...props} />
    ),
    th: ({ node, ...props }: any) => (
      <th className="py-2 px-4 border-b border-gray-300 dark:border-gray-600 text-left font-semibold text-gray-700 dark:text-gray-300" {...props} />
    ),
    td: ({ node, ...props }: any) => (
      <td className="py-2 px-4 border-b border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300" {...props} />
    ),
    hr: ({ node, ...props }: any) => (
      <hr className="my-6 border-gray-300 dark:border-gray-700" {...props} />
    ),
    img: ({ node, ...props }: any) => (
      <img className="max-w-full h-auto my-4 rounded-md" {...props} alt={props.alt || 'Research image'} />
    ),
    code: ({ node, inline, className, children, ...props }: any) => {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter
          style={nord}
          language={match[1]}
          PreTag="div"
          className="rounded-md my-4"
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code
          className={`${className} bg-gray-200 dark:bg-gray-700 rounded px-1 py-0.5 text-sm font-mono`}
          {...props}
        >
          {children}
        </code>
      );
    }
  }), []);

  return (
    <div className={`research-markdown-container prose dark:prose-invert max-w-none ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default ResearchMarkdownViewer; 