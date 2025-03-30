import React, { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { XMarkIcon, LinkIcon, StarIcon } from '@heroicons/react/24/outline';
import { useAuth } from '@/contexts/AuthContext';
import api from '@/services/api';
import { toast } from 'react-hot-toast';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  stance: string;
  coreArgument: string;
  supportingArguments: string[];
  queryId: number;
  onAllStarsUpdate: (count: number) => void;
}

interface RatingButtonsProps {
  argumentText: string;
}

interface RatingOption {
  value: string;
  label: string;
  tooltip: string;
}

interface RatedSection {
  argumentText: string;
  isRated: boolean;
}

interface ExistingRating {
  id: number;
  user: number;
  query: number;
  stance: string;
  core_argument: string;
  rating: string;
  created_at: string;
  argument_source: string;
}

interface RatingResponse {
  rating_id: number;
  allstars: number;
  created_at: string;
  argument_source: string;
}

interface ThumbsRatingResponse {
  id: number;
  allstars: number;
  created_at: string;
  user: number;
  query: number;
  core_argument: string;
  stance: string;
  rating: 'UP' | 'DOWN';
}

const ratingOptions: RatingOption[] = [
  { value: 'L', label: 'L', tooltip: 'Reflects positions typically associated with progressive/liberal thought' },
  { value: 'LL', label: 'LL', tooltip: 'Moderately aligns with progressive/liberal thought' },
  { value: 'C', label: 'C', tooltip: 'Represents balanced or nonpartisan viewpoints' },
  { value: 'LR', label: 'LR', tooltip: 'Moderately aligns with conservative/right-wing thought' },
  { value: 'R', label: 'R', tooltip: 'Reflects positions typically associated with conservative/right-wing thought' }
];

// Utility function to format URLs in text
const formatTextWithUrls = (text: string): React.ReactNode[] => {
  // Regular expression to match URLs with or without @ symbol
  const urlRegex = /(?:@)?(https?:\/\/[^\s]+)/g;
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match;

  while ((match = urlRegex.exec(text)) !== null) {
    // Add text before the URL
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const fullUrl = match[1];
    try {
      const url = new URL(fullUrl);
      // Create a link component with improved styling
      parts.push(
        <a
          key={match.index}
          href={fullUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-purple-600 hover:text-purple-800 font-medium group"
        >
          <LinkIcon className="h-3.5 w-3.5 group-hover:scale-110 transition-transform" />
          <span className="group-hover:underline transition-all">
            {url.hostname.replace(/^www\./, '')}
          </span>
        </a>
      );
    } catch (e) {
      // If URL parsing fails, just show the original text
      parts.push(match[0].replace('@', ''));
    }
    lastIndex = match.index + match[0].length;
  }

  // Add remaining text after the last URL
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
};

export const SupportingArgumentsModal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  stance,
  coreArgument,
  supportingArguments,
  queryId,
  onAllStarsUpdate
}) => {
  const { refreshStats } = useAuth();
  const [ratedSections, setRatedSections] = useState<Record<string, boolean>>({});
  const [selectedRatings, setSelectedRatings] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [thumbsRatings, setThumbsRatings] = useState<Record<string, 'UP' | 'DOWN' | null>>({});
  const persistedRatingsRef = useRef<Record<string, 'UP' | 'DOWN' | null>>({});

  // Load persisted ratings from localStorage on mount
  useEffect(() => {
    const loadPersistedRatings = () => {
      try {
        const storedRatings = localStorage.getItem(`thumbs_ratings_${queryId}`);
        if (storedRatings) {
          const parsedRatings = JSON.parse(storedRatings);
          setThumbsRatings(parsedRatings);
          persistedRatingsRef.current = parsedRatings;
        }
      } catch (error) {
        console.error('Error loading persisted ratings:', error);
      }
    };

    loadPersistedRatings();
  }, [queryId]);

  // Initialize rated sections and thumbs ratings when modal opens
  useEffect(() => {
    if (isOpen) {
      // Initialize with both persisted ratings and current state
      const initialRatedSections: Record<string, boolean> = {
        [coreArgument]: false,
        ...Object.fromEntries(supportingArguments.map(arg => [arg, false]))
      };

      const initialThumbsRatings: Record<string, 'UP' | 'DOWN' | null> = {
        [coreArgument]: persistedRatingsRef.current[coreArgument] || null,
        ...Object.fromEntries(supportingArguments.map(arg => [
          arg,
          persistedRatingsRef.current[arg] || null
        ]))
      };

      setRatedSections(initialRatedSections);
      setSelectedRatings({});
      setThumbsRatings(prev => ({
        ...prev,
        ...initialThumbsRatings
      }));
      setLoading({});
      setError(null);
    }
  }, [isOpen, coreArgument, supportingArguments]);

  // Fetch existing ratings when modal opens or queryId changes
  useEffect(() => {
    const fetchExistingRatings = async () => {
      try {
        // Fetch thumbs ratings
        const thumbsResponse = await api.get(`/thumbs-ratings/?query_id=${queryId}`);
        const existingThumbsRatings = thumbsResponse.data;
        
        if (Array.isArray(existingThumbsRatings)) {
          const updatedThumbsRatings = { ...persistedRatingsRef.current };
          existingThumbsRatings.forEach(rating => {
            updatedThumbsRatings[rating.core_argument] = rating.rating as 'UP' | 'DOWN';
          });
          setThumbsRatings(updatedThumbsRatings);
          persistedRatingsRef.current = updatedThumbsRatings;
          
          // Persist to localStorage
          localStorage.setItem(`thumbs_ratings_${queryId}`, JSON.stringify(updatedThumbsRatings));
        }
      } catch (error) {
        console.error('Error fetching existing ratings:', error);
        setError('Failed to load existing ratings');
      }
    };

    fetchExistingRatings();
  }, [queryId]);

  // Update persisted ratings whenever thumbsRatings changes
  useEffect(() => {
    try {
      localStorage.setItem(`thumbs_ratings_${queryId}`, JSON.stringify(thumbsRatings));
      persistedRatingsRef.current = thumbsRatings;
    } catch (error) {
      console.error('Error persisting ratings:', error);
    }
  }, [thumbsRatings, queryId]);

  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [onClose]);

  const handleRate = async (argumentText: string, rating: string) => {
    if (loading[argumentText] || ratedSections[argumentText]) {
      return;
    }

    try {
      setLoading(prev => ({ ...prev, [argumentText]: true }));
      setError(null);

      // Validate inputs
      if (!argumentText || !rating || !queryId || !stance) {
        console.error('Missing required fields:', { argumentText, rating, queryId, stance });
        throw new Error('Missing required fields for rating');
      }

      // Ensure queryId is a number
      const numericQueryId = Number(queryId);
      if (isNaN(numericQueryId)) {
        throw new Error('Invalid query ID');
      }

      const requestData = {
        query: numericQueryId,
        stance: stance,
        core_argument: argumentText,
        rating: rating
      };

      try {
        const response = await api.post<RatingResponse>('/ratings/', requestData);
        
        // Update the rated state and selected rating
        setRatedSections(prev => ({
          ...prev,
          [argumentText]: true
        }));
        setSelectedRatings(prev => ({
          ...prev,
          [argumentText]: rating
        }));

        // Update AllStars in parent component
        onAllStarsUpdate(response.data.allstars);

        await refreshStats();
      } catch (apiError: any) {
        if (apiError.response?.data?.core_argument) {
          throw new Error(Array.isArray(apiError.response.data.core_argument) 
            ? apiError.response.data.core_argument[0] 
            : apiError.response.data.core_argument);
        } else if (apiError.response?.data?.detail) {
          throw new Error(apiError.response.data.detail);
        } else if (apiError.response?.data) {
          throw new Error(JSON.stringify(apiError.response.data));
        } else {
          throw new Error('Failed to submit rating. Please try again.');
        }
      }
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to submit rating';
      console.error('Rating submission error:', errorMessage);
      setError(errorMessage);
    } finally {
      setLoading(prev => ({ ...prev, [argumentText]: false }));
    }
  };

  const handleThumbsRating = async (argumentText: string, rating: 'UP' | 'DOWN') => {
    if (thumbsRatings[argumentText] || loading[argumentText]) return;

    try {
      setLoading(prev => ({ ...prev, [argumentText]: true }));
      setError(null);

      const response = await api.post<ThumbsRatingResponse>('/thumbs-ratings/', {
        query: queryId,
        core_argument: argumentText,
        rating: rating,
        stance: stance
      });

      const updatedRatings = {
        ...thumbsRatings,
        [argumentText]: rating
      };

      setThumbsRatings(updatedRatings);
      persistedRatingsRef.current = updatedRatings;
      
      // Persist to localStorage
      localStorage.setItem(`thumbs_ratings_${queryId}`, JSON.stringify(updatedRatings));

      // Ensure allstars is a number and handle it properly
      const newAllStars = typeof response.data.allstars === 'number' ? response.data.allstars : parseInt(response.data.allstars);
      if (!isNaN(newAllStars)) {
        onAllStarsUpdate(newAllStars);
        await refreshStats(); // Refresh stats to ensure consistency
      } else {
        console.error('Invalid allstars value received:', response.data.allstars);
      }

      // Show success toast
      toast.success('Rating submitted successfully!', {
        position: 'bottom-right',
        duration: 3000,
      });

    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'Failed to submit rating';
      setError(errorMessage);
      toast.error(errorMessage, {
        position: 'bottom-right',
        duration: 3000,
      });
    } finally {
      setLoading(prev => ({ ...prev, [argumentText]: false }));
    }
  };

  const RatingButtons = React.memo<RatingButtonsProps>(({ argumentText }) => {
    const isLoading = loading[argumentText] === true;
    const currentRating = thumbsRatings[argumentText];

    return (
      <div className="mt-4 flex flex-col items-center gap-4 relative">
        {/* Original rating buttons - hidden but preserved */}
        <div className="hidden">
          {ratingOptions.map((option) => (
            <motion.button
              key={option.value}
              disabled={true}
              className="hidden"
            >
              {option.label}
            </motion.button>
          ))}
        </div>

        {/* New Thumbs Up/Down buttons */}
        <div className="flex items-center justify-center gap-6 w-full">
          {/* Thumbs Up Button */}
          <div className={`relative flex-shrink-0 ${currentRating ? 'pointer-events-none' : ''}`}>
            <motion.button
              initial={false}
              animate={{ 
                scale: 1,
                backgroundColor: currentRating === 'UP' ? 'rgb(220 252 231)' : 'rgb(243 244 246)'
              }}
              whileHover={currentRating ? {} : { scale: 1.1 }}
              whileTap={currentRating ? {} : { scale: 0.95 }}
              onClick={() => !currentRating && handleThumbsRating(argumentText, 'UP')}
              disabled={!!currentRating || isLoading}
              className={`
                p-3 rounded-full transition-all duration-300
                ${currentRating === 'UP' 
                  ? 'bg-green-100 text-green-600 shadow-lg ring-2 ring-green-500/50 transform scale-110' 
                  : currentRating 
                    ? 'bg-gray-100 text-gray-400 opacity-40 transform scale-90'
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-600 hover:shadow-md'}
                ${isLoading ? 'opacity-50 cursor-not-allowed animate-pulse' : ''}
                group focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500
              `}
            >
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                className={`h-8 w-8 transition-all duration-200 ${currentRating ? '' : 'group-hover:scale-110'}`}
                fill={currentRating === 'UP' ? 'currentColor' : 'none'}
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" 
                />
              </svg>
              {/* Tooltip */}
              {!currentRating && !isLoading && (
                <div className="absolute -top-2 left-1/2 -translate-x-1/2 -translate-y-full opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                  <div className="bg-white text-gray-700 text-sm rounded-lg shadow-lg border border-gray-200 px-3 py-2 whitespace-nowrap">
                    Agree with this argument
                    <div className="absolute bottom-[-8px] left-1/2 -translate-x-1/2 w-2 h-2 bg-white border-r border-b border-gray-200 transform rotate-45" />
                  </div>
                </div>
              )}
            </motion.button>
          </div>

          {/* Thumbs Down Button */}
          <div className={`relative flex-shrink-0 ${currentRating ? 'pointer-events-none' : ''}`}>
            <motion.button
              initial={false}
              animate={{ 
                scale: 1,
                backgroundColor: currentRating === 'DOWN' ? 'rgb(254 226 226)' : 'rgb(243 244 246)'
              }}
              whileHover={currentRating ? {} : { scale: 1.1 }}
              whileTap={currentRating ? {} : { scale: 0.95 }}
              onClick={() => !currentRating && handleThumbsRating(argumentText, 'DOWN')}
              disabled={!!currentRating || isLoading}
              className={`
                p-3 rounded-full transition-all duration-300
                ${currentRating === 'DOWN' 
                  ? 'bg-red-100 text-red-600 shadow-lg ring-2 ring-red-500/50 transform scale-110' 
                  : currentRating 
                    ? 'bg-gray-100 text-gray-400 opacity-40 transform scale-90'
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-600 hover:shadow-md'}
                ${isLoading ? 'opacity-50 cursor-not-allowed animate-pulse' : ''}
                group focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500
              `}
            >
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                className={`h-8 w-8 transform rotate-180 transition-all duration-200 ${currentRating ? '' : 'group-hover:scale-110'}`}
                fill={currentRating === 'DOWN' ? 'currentColor' : 'none'}
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" 
                />
              </svg>
              {/* Tooltip */}
              {!currentRating && !isLoading && (
                <div className="absolute -top-2 left-1/2 -translate-x-1/2 -translate-y-full opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                  <div className="bg-white text-gray-700 text-sm rounded-lg shadow-lg border border-gray-200 px-3 py-2 whitespace-nowrap">
                    Disagree with this argument
                    <div className="absolute bottom-[-8px] left-1/2 -translate-x-1/2 w-2 h-2 bg-white border-r border-b border-gray-200 transform rotate-45" />
                  </div>
                </div>
              )}
            </motion.button>
          </div>
        </div>

        {/* Success animation when rated */}
        <AnimatePresence>
          {currentRating && (
            <motion.div
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none z-20"
            >
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ 
                  scale: [1, 1.2, 1],
                  opacity: [0, 1, 0]
                }}
                transition={{
                  duration: 0.5,
                  times: [0, 0.2, 1],
                  ease: "easeInOut"
                }}
                className={`text-4xl ${currentRating === 'UP' ? 'text-green-500' : 'text-red-500'}`}
              >
                {currentRating === 'UP' ? '👍' : '👎'}
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  });

  const ArgumentDisplay: React.FC<{ text: string }> = ({ text }) => {
    return (
      <div className="text-gray-700 mb-4">
        {formatTextWithUrls(text)}
      </div>
    );
  };

  const renderCoreArgument = React.useCallback(() => {
    return (
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2 text-purple-600">
          Core Argument
        </h3>
        <div className="relative">
          <ArgumentDisplay text={coreArgument} />
          <div className="h-px bg-gradient-to-r from-purple-500/30 to-indigo-500/30 mb-4" />
        </div>
        <RatingButtons argumentText={coreArgument} />
      </div>
    );
  }, [coreArgument, ratedSections[coreArgument]]);

  const renderSupportingArgument = React.useCallback((argument: string, index: number) => {
    return (
      <div key={index} className="mb-6">
        <div className="relative">
          <ArgumentDisplay text={argument} />
        </div>
        <RatingButtons argumentText={argument} />
      </div>
    );
  }, [ratedSections]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-white/50 backdrop-blur-sm cursor-pointer z-40"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 350 }}
            className="fixed inset-4 sm:inset-8 md:inset-16 z-50 flex items-start justify-center pointer-events-none"
          >
            <div 
              className="relative bg-white/80 backdrop-blur-sm rounded-2xl shadow-2xl w-full max-w-4xl max-h-[85vh] pointer-events-auto overflow-hidden border border-white/20"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Decorative gradient blur */}
              <div className="absolute inset-0 bg-gradient-to-br from-purple-600/10 to-indigo-600/10 pointer-events-none" />
              
              {/* Content container */}
              <div className="relative h-full flex flex-col" onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div className="flex-shrink-0 p-4 sm:p-6 border-b border-purple-100">
                  <div className="flex justify-between items-center">
                    <h3 className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
                      {stance}
                    </h3>
                    <button
                      onClick={onClose}
                      className="text-gray-400 hover:text-gray-500 transition-colors p-2 hover:bg-gray-100 rounded-lg"
                    >
                      <XMarkIcon className="h-6 w-6" />
                    </button>
                  </div>
                </div>
                
                {/* Scrollable Content */}
                <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
                  {/* Core Argument */}
                  {renderCoreArgument()}
                  
                  {/* Supporting Arguments */}
                  <div className="space-y-4">
                    <h4 className="text-lg font-semibold text-purple-600 flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600" />
                      Supporting Arguments
                    </h4>
                    <motion.div className="space-y-4">
                      {supportingArguments.map((arg, index) => renderSupportingArgument(arg, index))}
                    </motion.div>
                  </div>

                  {/* Error Message */}
                  <AnimatePresence>
                    {error && (
                      <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="mt-4 p-3 rounded-md bg-red-50 text-red-500 text-sm"
                      >
                        {error}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}; 