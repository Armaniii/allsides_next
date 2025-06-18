import React from 'react';
import { LinkIcon } from '@heroicons/react/24/outline';
import { formatUrlForDisplay } from '@/utils/url';

interface FormattedLinkProps {
  url: string;
  displayText?: string;
  className?: string;
}

/**
 * A link component that displays a clean domain name but routes to the full URL.
 * Handles URL normalization and provides consistent styling.
 */
export const FormattedLink: React.FC<FormattedLinkProps> = ({ 
  url, 
  displayText, 
  className = "" 
}) => {
  // Ensure URL has protocol for proper routing - handle http, https, and ftp
  const normalizedUrl = /^(https?|ftp):\/\//.test(url) ? url : `https://${url}`;
  
  // Use provided display text or format the URL for display
  const displayContent = displayText || formatUrlForDisplay(url);
  
  const baseClasses = "inline-flex items-center gap-1 text-purple-600 hover:text-purple-800 font-medium group";
  const finalClasses = className ? `${baseClasses} ${className}` : baseClasses;

  return (
    <a
      href={normalizedUrl}
      target="_blank"
      rel="noopener noreferrer"
      className={finalClasses}
    >
      <LinkIcon className="h-3.5 w-3.5 group-hover:scale-110 transition-transform" />
      <span className="group-hover:underline transition-all">
        {displayContent}
      </span>
    </a>
  );
};