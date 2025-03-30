import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDownIcon, ChevronUpIcon, TrashIcon } from '@heroicons/react/24/outline';
import { Query } from '@/lib/api';

interface QueryHistoryCardProps {
  query: Query;
  onDelete?: (id: number) => void;
  searchTerm?: string;
  isExpanded?: boolean;
  onToggleExpand?: (id: number) => void;
  onArgumentClick?: (argument: any) => void;
}

const QueryHistoryCard: React.FC<QueryHistoryCardProps> = ({
  query,
  onDelete,
  searchTerm = '',
  isExpanded = false,
  onToggleExpand,
  onArgumentClick,
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const highlightText = (text: string) => {
    if (!searchTerm?.trim()) return text;
    const escapedSearchTerm = searchTerm.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const parts = text.split(new RegExp(`(${escapedSearchTerm})`, 'gi'));
    return parts.map((part, index) => 
      part.toLowerCase() === searchTerm.trim().toLowerCase() ? 
        <mark key={index} className="bg-amber-100 text-amber-900 not-italic rounded px-0.5">{part}</mark> : 
        part
    );
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="card card-hover overflow-hidden mb-4"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Header */}
      <div className="p-6 border-b border-gray-100">
        <div className="flex justify-between items-start gap-4">
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold text-gray-900 mb-2 line-clamp-2">
              {highlightText(query.query_text)}
            </h3>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-500">
                {formatDate(query.created_at)}
              </span>
              <span className="badge badge-purple">
                Score: {query.diversity_score}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {onDelete && (
              <button
                onClick={() => onDelete(query.id)}
                className={`btn-secondary p-2 !rounded-full transition-all duration-200 ${
                  isHovered ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
                }`}
                aria-label="Delete query"
              >
                <TrashIcon className="h-5 w-5 text-gray-500 hover:text-rose-600" />
              </button>
            )}
            {onToggleExpand && (
              <button
                onClick={() => onToggleExpand(query.id)}
                className="btn-secondary p-2 !rounded-full"
                aria-label={isExpanded ? 'Collapse' : 'Expand'}
              >
                {isExpanded ? (
                  <ChevronUpIcon className="h-5 w-5 text-gray-500" />
                ) : (
                  <ChevronDownIcon className="h-5 w-5 text-gray-500" />
                )}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Arguments */}
      <AnimatePresence>
        {isExpanded && query.response && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="p-6 space-y-6">
              {query.response.arguments.map((argument, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="space-y-4"
                >
                  {/* Stance */}
                  <div className="flex items-center gap-2">
                    <h4 className="text-lg font-semibold text-gray-900">
                      {highlightText(argument.stance)}
                    </h4>
                    <span className="badge badge-blue">Stance</span>
                  </div>

                  {/* Core Argument */}
                  <div
                    onClick={() => onArgumentClick?.(argument)}
                    className="p-4 bg-indigo-50 rounded-lg border-l-4 border-indigo-500 
                             hover:bg-indigo-100 transition-colors cursor-pointer"
                  >
                    <p className="text-gray-700 font-medium">
                      {highlightText(argument.core_argument)}
                    </p>
                  </div>

                  {/* Supporting Arguments */}
                  <div className="ml-4 space-y-3">
                    {argument.supporting_arguments.map((supporting, sIndex) => (
                      <motion.div
                        key={sIndex}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: (index + sIndex) * 0.1 }}
                        className="group p-4 bg-gray-50 rounded-lg border-l-4 border-gray-300 
                                 hover:bg-gray-100 hover:border-indigo-400 transition-all duration-200 
                                 cursor-pointer"
                        onClick={() => onArgumentClick?.({ ...argument, selected_supporting: supporting })}
                      >
                        <p className="text-gray-600 group-hover:text-gray-900 transition-colors">
                          {highlightText(supporting)}
                        </p>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default QueryHistoryCard; 