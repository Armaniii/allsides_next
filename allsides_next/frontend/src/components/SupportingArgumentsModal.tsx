import React, { useEffect, useState, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { XMarkIcon, LinkIcon, StarIcon, ChevronDownIcon } from '@heroicons/react/24/outline';
import { useAuth } from '@/contexts/AuthContext';
import api from '@/services/api';
import { toast } from 'react-hot-toast';
import { FormattedLink } from './FormattedLink';
import { normalizeDomain } from '@/utils/url';

interface EvidenceItem {
  quote: string;
  citation_id: number;
  reasoning: string;
  stance: 'supports' | 'refutes';
  url: string;
  domain: string;
  formatted: string;
}

interface EvidenceMetadata {
  supporting_evidence_count: number;
  refuting_evidence_count: number;
  total_evidence_count: number;
  primary_sources: number;
  secondary_sources: number;
}

interface DetailedEvidence {
  supporting: EvidenceItem[];
  refuting: EvidenceItem[];
}

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  stance: string;
  supportingArguments: string[];
  references?: any[];
  queryId: number;
  onAllStarsUpdate: (count: number) => void;
  evidenceMetadata?: EvidenceMetadata;
  detailedEvidence?: DetailedEvidence;
  core_argument_summary?: string;
  source_analysis?: {
    dialectical_summary?: string;
    supporting_profile?: any;
    refuting_profile?: any;
    key_perspectives?: string[];
    // Legacy fields for backward compatibility
    average_trust?: number;
    distribution?: Record<string, { count: number; percentage: number }>;
    trust_distribution?: { high: number; medium: number; low: number };
    biases?: string[];
    enhanced_sources?: Array<{
      url: string;
      domain: string;
      category: string;
      trust_score: number;
      credibility_indicators: string[];
    }>;
  };
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

// Utility function to format URLs and domain citations in text
const formatTextWithUrls = (text: string, referenceMap?: Map<string, any>): React.ReactNode[] => {
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  
  // Combined regex to match both [domain.com] citations and URLs
  const combinedRegex = /(\[([^\]]+\.[^\]]+)\])|(?:@)?(https?:\/\/[^\s]+)/g;
  let match;
  
  while ((match = combinedRegex.exec(text)) !== null) {
    // Add text before the match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    
    if (match[1]) {
      // This is a [domain.com] citation
      const domain = match[2];
      if (!domain || domain === 'unknown') {
        // Don't render as a link if domain is empty/invalid
        parts.push(domain || '');
        lastIndex = match.index + match[0].length;
        continue;
      }
      let fullUrl = `https://${domain}`; // Default fallback
      let foundRef = false;
      if (referenceMap && referenceMap.size > 0) {
        const normalizedTargetDomain = normalizeDomain(domain);
        const ref = referenceMap.get(normalizedTargetDomain);
        if (ref && ref.url && ref.domain) {
          foundRef = true;
          fullUrl = ref.url.startsWith('http') ? ref.url : `https://${ref.url}`;
        }
      }
      if (foundRef) {
        parts.push(
          <FormattedLink
            key={match.index}
            url={fullUrl}
            displayText={domain}
          />
        );
      } else {
        // Log a warning in dev
        if (typeof window !== 'undefined' && window.console) {
          // eslint-disable-next-line no-console
          console.warn(`No reference found for domain [${domain}], not rendering as a link.`);
        }
        parts.push(domain);
      }
    } else {
      // This is a full URL - display domain but link to full URL
      const fullUrl = match[3];
      parts.push(
        <FormattedLink
          key={match.index}
          url={fullUrl}
        />
      );
    }
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  
  // If no matches found, return the original text
  if (parts.length === 0) {
    parts.push(text);
  }
  
  return parts;
};

export const SupportingArgumentsModal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  stance,
  supportingArguments,
  references,
  queryId,
  onAllStarsUpdate,
  evidenceMetadata,
  detailedEvidence,
  core_argument_summary,
  source_analysis
}) => {
  const { refreshStats } = useAuth();

  const effectiveCoreSummary = core_argument_summary;
  const effectiveSourceAnalysis = source_analysis;
  const [ratedSections, setRatedSections] = useState<Record<string, boolean>>({});
  const [selectedRatings, setSelectedRatings] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [thumbsRatings, setThumbsRatings] = useState<Record<string, 'UP' | 'DOWN' | null>>({});
  const persistedRatingsRef = useRef<Record<string, 'UP' | 'DOWN' | null>>({});
  const [showAllEvidence, setShowAllEvidence] = useState<boolean>(false);
  const [showDialecticalAnalysis, setShowDialecticalAnalysis] = useState<boolean>(false);

  // Performance optimization: Pre-compute reference map for O(1) lookups
  const referenceMap = useMemo(() => {
    if (!references || references.length === 0) {
      return new Map();
    }
    const map = new Map();
    references.forEach(ref => {
      if (ref.domain) {
        const normalizedDomain = normalizeDomain(ref.domain);
        map.set(normalizedDomain, ref);
      }
      if (ref.url) {
        try {
          const urlObj = new URL(ref.url.startsWith('http') ? ref.url : `https://${ref.url}`);
          const normalizedUrlDomain = normalizeDomain(urlObj.hostname);
          map.set(normalizedUrlDomain, ref);
        } catch (e) {
          // Ignore malformed URLs
        }
      }
    });
    return map;
  }, [references]);

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
        ...Object.fromEntries(supportingArguments.map(arg => [arg, false]))
      };

      const initialThumbsRatings: Record<string, 'UP' | 'DOWN' | null> = {
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
  }, [isOpen, supportingArguments]);

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
        {/* All rating buttons - temporarily hidden */}
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

        {/* Thumbs Up/Down buttons - temporarily hidden */}
        <div className="hidden flex items-center justify-center gap-6 w-full">
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

  const ArgumentDisplay: React.FC<{ text: string; referenceMap?: Map<string, any> }> = ({ text, referenceMap }) => {
    return (
      <div className="text-[#3D3748] mb-4">
        {formatTextWithUrls(text, referenceMap)}
      </div>
    );
  };

  // Removed renderCoreArgument function as we no longer display core argument

  // Parse arguments to separate supporting and refuting
  // Use detailed evidence if available, otherwise fall back to text parsing
  const parseArgumentsByStance = React.useMemo(() => {
    // Debug: Log evidence structure (remove in production)
    // console.log('🔍 Modal Debug - detailedEvidence:', detailedEvidence);
    // console.log('🔍 Modal Debug - evidenceMetadata:', evidenceMetadata);
    
    if (detailedEvidence && (detailedEvidence.supporting || detailedEvidence.refuting)) {
      // console.log('✅ Using detailed evidence structure');
      // console.log('Supporting items:', detailedEvidence.supporting?.length || 0);
      // console.log('Refuting items:', detailedEvidence.refuting?.length || 0);
      
      // Use the structured evidence data from backend
      return {
        supporting: detailedEvidence.supporting?.map(item => item.formatted) || [],
        refuting: detailedEvidence.refuting?.map(item => item.formatted) || [],
        supportingItems: detailedEvidence.supporting || [],
        refutingItems: detailedEvidence.refuting || []
      };
    }
    
    // Fallback to text parsing for older data
    // console.log('⚠️ Using fallback text parsing');
    const supporting: string[] = [];
    const refuting: string[] = [];
    
    supportingArguments.forEach((argument, index) => {
      // Check if argument contains stance indicators
      // Look for various patterns that might indicate refuting evidence
      const refutingPatterns = [
        /\(refutes?\)/i,
        /\(opposes?\)/i,
        /\(contradicts?\)/i,
        /\(challenges?\)/i,
        /\(disputes?\)/i,
        /\(argues? against\)/i,
        /\(disagrees?\)/i,
        /however,/i,
        /but /i,
        /contrary to/i,
        /in contrast/i,
        /on the other hand/i,
        /critics argue/i,
        /opponents claim/i,
        /studies show that this is not/i,
        /evidence suggests otherwise/i
      ];
      
      const isRefuting = refutingPatterns.some(pattern => pattern.test(argument));
      
      if (isRefuting) {
        // console.log(`📍 Found refuting argument ${index}: ${argument.substring(0, 50)}...`);
        refuting.push(argument);
      } else {
        supporting.push(argument);
      }
    });
    
    // console.log(`📊 Final parsing result - Supporting: ${supporting.length}, Refuting: ${refuting.length}`);
    return { supporting, refuting, supportingItems: [], refutingItems: [] };
  }, [supportingArguments, detailedEvidence]);

  const renderArgument = React.useCallback((argument: string, index: number, isSupporting: boolean) => {
    return (
      <motion.div 
        key={`${isSupporting ? 'support' : 'refute'}-${index}`}
        initial={{ opacity: 0, x: isSupporting ? -20 : 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: index * 0.1 }}
        className="mb-4 relative"
      >
        {/* Colored indicator bar */}
        <div className={`absolute left-0 top-0 bottom-0 w-1 rounded-full ${
          isSupporting ? 'bg-gradient-to-b from-green-400 to-green-600' : 'bg-gradient-to-b from-red-400 to-red-600'
        }`} />
        
        {/* Content */}
        <div className="pl-6 pr-2">
          <div className={`p-4 rounded-xl border-l-4 transition-all duration-300 hover:shadow-md ${
            isSupporting 
              ? 'bg-[#FDFCFB] border-l-green-500 hover:bg-[#FDFCFB]/90' 
              : 'bg-[#FDFCFB] border-l-red-500 hover:bg-[#FDFCFB]/90'
          }`}>
            {/* Stance indicator */}
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-2 h-2 rounded-full ${
                isSupporting ? 'bg-green-500' : 'bg-red-500'
              }`} />
              <span className={`text-xs font-medium uppercase tracking-wide ${
                isSupporting ? 'text-green-700' : 'text-red-700'
              }`}>
                {isSupporting ? 'Supporting' : 'Refuting'}
              </span>
            </div>
            
            {/* Argument content */}
            <div className="relative">
              <ArgumentDisplay text={argument} referenceMap={referenceMap} />
              
              {/* Dialectical Tags for Evidence Items */}
              <div className="mt-3 flex flex-wrap gap-2">
                {/* Mock dialectical tags based on argument content for demonstration */}
                <div className="flex gap-1">
                  <span className="inline-flex items-center gap-1 px-2 py-1 bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-700 rounded border border-blue-100 text-xs">
                    <span>🌐</span>
                    <span>Mainstream</span>
                  </span>
                  <span className="inline-flex items-center gap-1 px-2 py-1 bg-gradient-to-r from-green-50 to-emerald-50 text-green-700 rounded border border-green-100 text-xs">
                    <span>✅</span>
                    <span>{isSupporting ? 'Arg. from Authority' : 'Arg. from Consequence'}</span>
                  </span>
                </div>
                <span className="inline-flex items-center gap-1 px-2 py-1 bg-gradient-to-r from-purple-50 to-violet-50 text-purple-700 rounded border border-purple-100 text-xs">
                  <span>📄</span>
                  <span>Academic Research Summary</span>
                </span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    );
  }, [references]);

  // No longer needed since we parse arguments by stance
  // const visibleArguments = supportingArguments;
  // const hasMoreArguments = false;

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
              className="relative bg-[#FDFCFB]/80 backdrop-blur-sm rounded-2xl shadow-2xl w-full max-w-4xl max-h-[85vh] pointer-events-auto overflow-hidden border border-white/20"
              onClick={(e) => e.stopPropagation()}
            >
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
                <div className="flex-1 p-4 sm:p-6 max-h-[calc(85vh-120px)] overflow-y-auto">
                  {/* Core Argument Summary */}
                  {effectiveCoreSummary && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mb-6 p-4 bg-gradient-to-r from-purple-50/80 to-indigo-50/80 rounded-xl border border-purple-100/50"
                    >
                      <h4 className="text-sm font-semibold text-purple-700 mb-2 flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-purple-500" />
                        Core Argument
                      </h4>
                      <p className="text-gray-700 leading-relaxed">{effectiveCoreSummary}</p>
                    </motion.div>
                  )}

                  {/* Dialectical Analysis - Collapsible */}
                  {effectiveSourceAnalysis && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 }}
                      className="mb-6"
                    >
                      {/* Collapsible Header */}
                      <motion.button
                        onClick={() => setShowDialecticalAnalysis(!showDialecticalAnalysis)}
                        className="w-full p-4 bg-gradient-to-r from-indigo-50/80 to-purple-50/80 hover:from-indigo-100/80 hover:to-purple-100/80 rounded-xl border border-indigo-200/50 hover:border-indigo-300/50 transition-all duration-300 group"
                        whileHover={{ scale: 1.01 }}
                        whileTap={{ scale: 0.99 }}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className="p-2 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg shadow-sm">
                              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                              </svg>
                            </div>
                            <div className="text-left">
                              <h4 className="text-sm font-semibold text-indigo-700 group-hover:text-indigo-800 transition-colors">
                                Dialectical Analysis
                              </h4>
                              <p className="text-xs text-gray-600 group-hover:text-gray-700 transition-colors">
                                View argument structure, discourse positioning & transparency insights
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-indigo-600 font-medium opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                              {showDialecticalAnalysis ? 'Hide' : 'Show'} Details
                            </span>
                            <motion.div
                              animate={{ rotate: showDialecticalAnalysis ? 180 : 0 }}
                              transition={{ duration: 0.3 }}
                              className="p-1"
                            >
                              <ChevronDownIcon className="h-5 w-5 text-indigo-600 group-hover:text-indigo-700 transition-colors" />
                            </motion.div>
                          </div>
                        </div>
                      </motion.button>

                      {/* Collapsible Content */}
                      <AnimatePresence>
                        {showDialecticalAnalysis && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.3, ease: 'easeInOut' }}
                            className="overflow-hidden"
                          >
                            <div className="mt-3 p-5 bg-white/60 backdrop-blur-sm rounded-xl border border-gray-200/50 space-y-6">
                              
                              {/* Dialectical Summary */}
                              {effectiveSourceAnalysis.dialectical_summary && (
                                <div className="p-4 bg-gradient-to-r from-indigo-50/50 to-purple-50/50 rounded-lg border border-indigo-100/50">
                                  <div className="flex items-center gap-2 mb-3">
                                    <div className="w-2 h-2 rounded-full bg-indigo-500" />
                                    <span className="text-sm font-medium text-indigo-700">Intellectual Structure</span>
                                  </div>
                                  <p className="text-sm text-gray-700 leading-relaxed">
                                    {effectiveSourceAnalysis.dialectical_summary}
                                  </p>
                                </div>
                              )}
                              
                              {/* Profile Statistics */}
                              {(effectiveSourceAnalysis.supporting_profile || effectiveSourceAnalysis.refuting_profile) && (
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                  {effectiveSourceAnalysis.supporting_profile && (
                                    <div className="p-4 bg-green-50/50 rounded-lg border border-green-100/50">
                                      <div className="flex items-center gap-2 mb-3">
                                        <div className="w-2 h-2 rounded-full bg-green-500" />
                                        <h5 className="text-sm font-semibold text-green-700">
                                          Supporting Evidence ({effectiveSourceAnalysis.supporting_profile.count})
                                        </h5>
                                      </div>
                                      <div className="space-y-2">
                                        {Object.entries(effectiveSourceAnalysis.supporting_profile.discourse_positions || {}).slice(0, 3).map(([position, count]) => (
                                          <div key={position} className="flex items-center justify-between">
                                            <span className="text-xs text-gray-700 font-medium">{position}</span>
                                            <span className="text-xs text-green-600 font-semibold bg-green-100 px-2 py-1 rounded-full">{count as number}</span>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {effectiveSourceAnalysis.refuting_profile && (
                                    <div className="p-4 bg-red-50/50 rounded-lg border border-red-100/50">
                                      <div className="flex items-center gap-2 mb-3">
                                        <div className="w-2 h-2 rounded-full bg-red-500" />
                                        <h5 className="text-sm font-semibold text-red-700">
                                          Refuting Evidence ({effectiveSourceAnalysis.refuting_profile.count})
                                        </h5>
                                      </div>
                                      <div className="space-y-2">
                                        {Object.entries(effectiveSourceAnalysis.refuting_profile.discourse_positions || {}).slice(0, 3).map(([position, count]) => (
                                          <div key={position} className="flex items-center justify-between">
                                            <span className="text-xs text-gray-700 font-medium">{position}</span>
                                            <span className="text-xs text-red-600 font-semibold bg-red-100 px-2 py-1 rounded-full">{count as number}</span>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}

                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  )}

                  {/* Sticky Evidence Header */}
                  <div className="sticky top-0 bg-white/95 backdrop-blur-sm py-3 -mx-4 px-4 rounded-lg mb-6 border-b border-purple-100 z-10">
                    <h4 className="text-lg font-semibold text-purple-600 flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600" />
                      Evidence ({evidenceMetadata?.total_evidence_count || supportingArguments.length} items)
                      {evidenceMetadata && (
                        <span className="text-sm font-normal text-gray-500 ml-2">
                          {evidenceMetadata.primary_sources} primary, {evidenceMetadata.secondary_sources} secondary
                        </span>
                      )}
                    </h4>
                  </div>

                  {/* Two Column Layout for Supporting vs Refuting */}
                  {parseArgumentsByStance.refuting.length > 0 ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {/* Supporting Arguments Column */}
                      <div className="space-y-4">
                        <div className="flex items-center gap-2 mb-4">
                          <div className="w-3 h-3 rounded-full bg-green-500" />
                          <h5 className="text-base font-semibold text-green-700">
                            Supporting ({parseArgumentsByStance.supporting.length})
                          </h5>
                        </div>
                        <div className="max-h-[calc(85vh-280px)] overflow-y-auto pr-2 space-y-4">
                          {parseArgumentsByStance.supporting.map((arg, index) => 
                            renderArgument(arg, index, true)
                          )}
                          {parseArgumentsByStance.supporting.length === 0 && (
                            <div className="text-gray-500 text-sm italic p-4 bg-gray-50 rounded-lg">
                              No supporting evidence found
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Refuting Arguments Column */}
                      <div className="space-y-4">
                        <div className="flex items-center gap-2 mb-4">
                          <div className="w-3 h-3 rounded-full bg-red-500" />
                          <h5 className="text-base font-semibold text-red-700">
                            Refuting ({parseArgumentsByStance.refuting.length})
                          </h5>
                        </div>
                        <div className="max-h-[calc(85vh-280px)] overflow-y-auto pr-2 space-y-4">
                          {parseArgumentsByStance.refuting.map((arg, index) => 
                            renderArgument(arg, index, false)
                          )}
                        </div>
                      </div>
                    </div>
                  ) : (
                    /* Single Column Layout when no refuting arguments */
                    <div className="space-y-4">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-3 h-3 rounded-full bg-green-500" />
                        <h5 className="text-base font-semibold text-green-700">
                          Supporting Evidence ({parseArgumentsByStance.supporting.length})
                        </h5>
                      </div>
                      <div className="max-h-[calc(85vh-280px)] overflow-y-auto pr-2 space-y-4">
                        {parseArgumentsByStance.supporting.map((arg, index) => 
                          renderArgument(arg, index, true)
                        )}
                      </div>
                    </div>
                  )}

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