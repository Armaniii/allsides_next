'use client';

import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { GPTResponse, Argument, TextObject, stats, StreamingArgument, UserStats, LeaderboardUser, leaderboard } from '@/lib/api';
import type { Query } from '@/lib/api';
import { queries as apiQueries } from '@/lib/api';
import api, { auth } from '@/services/api';
import { useAuth } from '@/contexts/AuthContext';
import { useRouter } from 'next/navigation';
import { MagnifyingGlassIcon, TrashIcon, ChevronDownIcon, ChevronUpIcon, XMarkIcon, LinkIcon, StarIcon, Cog6ToothIcon } from '@heroicons/react/24/outline';
import { createPortal } from 'react-dom';
import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion';
import QueryHistoryCard from '@/components/QueryHistoryCard';
import { SupportingArgumentsModal } from '@/components/SupportingArgumentsModal';
import { StreamingArgumentCard } from '@/components/StreamingArgumentCard';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { useToast } from '@/components/ui/use-toast';
import { ArgumentsDisplay } from '@/components/ArgumentsDisplay';
import { Toaster } from '@/components/ui/toaster';
import { LeaderboardModal } from '@/components/LeaderboardModal';
import { cn } from '@/lib/utils';
import { Loader2, Send } from 'lucide-react';
import { formatTimeRemaining } from '@/utils/time';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  stance: string;
  coreArgument: string;
  supportingArguments: string[];
}

const SupportingArgumentsModalMain: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  stance,
  coreArgument,
  supportingArguments
}) => {
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [onClose]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-gray-300 cursor-pointer z-40"
          />
          
          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: "spring", damping: 20, stiffness: 300 }}
            className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none p-4"
          >
            <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl pointer-events-auto overflow-hidden">
              {/* Header */}
              <div className="p-6 border-b border-gray-100">
                <div className="flex justify-between items-center">
                  <h3 className="text-2xl font-bold text-gray-900">{stance}</h3>
                  <button
                    onClick={onClose}
                    className="text-gray-400 hover:text-gray-500 transition-colors"
                  >
                    <XMarkIcon className="h-6 w-6" />
                  </button>
                </div>
              </div>
              
              {/* Content */}
              <div className="p-6 space-y-6">
                {/* Core Argument */}
                <div className="space-y-2">
                  <h4 className="text-lg font-semibold text-gray-900">Core Argument</h4>
                  <p className="text-gray-700 select-text">{coreArgument}</p>
                  <div className="h-0.5 bg-purple-500 w-full"></div>
                </div>
                
                {/* Supporting Arguments */}
                <div className="space-y-4">
                  <h4 className="text-lg font-semibold text-gray-900">Supporting Arguments</h4>
                  <motion.div className="space-y-3">
                    {supportingArguments.map((arg, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="p-4 bg-gray-50 rounded-lg border-l-4 border-purple-500"
                      >
                        <p className="text-gray-700 select-text">{arg}</p>
                      </motion.div>
                    ))}
                  </motion.div>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

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

interface Stance {
  stance: string;
  supporting_arguments: string[];
  references?: any[];
  evidence_metadata?: EvidenceMetadata;
  detailed_evidence?: DetailedEvidence;
  core_argument_summary?: string;
  source_analysis?: {
    average_trust: number;
    distribution: Record<string, { count: number; percentage: number }>;
    trust_distribution: { high: number; medium: number; low: number };
    biases: string[];
  };
}

interface QueryResponse {
  arguments: Stance[];
  references?: any[]; // Add references to match backend response
  follow_up_questions?: string[]; // Add follow-up questions
}

interface QueryData {
  id: number;
  query_text: string;
  diversity_score: number;
  response: QueryResponse;
  created_at: string;
  is_active: boolean;
}

export default function HomePage() {
  const [queryText, setQueryText] = useState('');
  const [diversityScore, setDiversityScore] = useState([0.5]);
  const [showHistory, setShowHistory] = useState(false);
  const [currentQuery, setCurrentQuery] = useState<QueryData | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [hiddenQueries, setHiddenQueries] = useState<Set<number>>(new Set());
  const [selectionPopup, setSelectionPopup] = useState({ show: false, x: 0, y: 0, text: '' });
  const { user, userStats, loading, isAuthenticated, refreshStats, logout } = useAuth();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [selectedGroups, setSelectedGroups] = useState<Set<string>>(new Set());
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [expandedCards, setExpandedCards] = useState<{ [key: number]: boolean }>({});
  const [selectedArgument, setSelectedArgument] = useState<any>(null);
  const [queryList, setQueryList] = useState<Query[]>([]);
  const { toast } = useToast();
  const [showAllStarsAnimation, setShowAllStarsAnimation] = useState(false);
  const [showLeaderboard, setShowLeaderboard] = useState(false);
  const [leaderboardUsers, setLeaderboardUsers] = useState<LeaderboardUser[]>([]);
  const [queryLimitAnimation, setQueryLimitAnimation] = useState(false);
  const [showStanceSelector, setShowStanceSelector] = useState(false);
  const [stanceCount, setStanceCount] = useState(3); // Default to 3 stances
  const [lastMouseActivity, setLastMouseActivity] = useState<number>(Date.now());
  const [isLimitReached, setIsLimitReached] = useState(false);
  const [progressMessages, setProgressMessages] = useState<string[]>([]);
  const [processingStartTime, setProcessingStartTime] = useState<number | null>(null);

  const topicGroups = [
    "Politics",
    "Social Issues",
    "Technology",
    "Environment",
    "Healthcare"
  ];

  // Topic associations mapping
  const topicAssociations: Record<string, string[]> = {
    'gmo': ['Technology'],
    'climate change': ['Environment'],
    'global warming': ['Environment'],
    'abortion': ['Healthcare', 'Social Issues'],
    'elon musk': ['Technology', 'Social Issues'],
    'tesla': ['Technology'],
    'twitter': ['Technology', 'Social Issues'],
    'healthcare': ['Healthcare'],
    'medical': ['Healthcare'],
    'vaccine': ['Healthcare', 'Social Issues'],
    'election': ['Politics'],
    'democrat': ['Politics'],
    'republican': ['Politics'],
    'immigration': ['Politics', 'Social Issues'],
    'education': ['Social Issues'],
    'gun': ['Politics', 'Social Issues'],
  };

  const getQueryTopics = (queryText: string): string[] => {
    const topics = new Set<string>();
    const lowerText = queryText.toLowerCase();
    
    // Check for topic associations
    Object.entries(topicAssociations).forEach(([key, groups]) => {
      if (lowerText.includes(key.toLowerCase())) {
        groups.forEach(group => topics.add(group));
      }
    });
    
    return Array.from(topics);
  };

  useEffect(() => {
    if (!loading && !user) {
      router.replace('/login');
    }
  }, [loading, user, router]);

  const { data: userQueries = [], isLoading: queriesLoading } = useQuery<Query[]>({
    queryKey: ['queries'],
    queryFn: () => apiQueries.list(),
    enabled: !!user,
    staleTime: 30000, // Consider data fresh for 30 seconds
  });

  const mutation = useMutation<Query, Error, { query_text: string; diversity_score: number; num_stances: number }>({
    // Set mutation timeout to 10 minutes for extended processing
    networkMode: 'always',
    gcTime: 600000, // 10 minutes garbage collection time
    mutationFn: async (data: { query_text: string; diversity_score: number; num_stances: number }): Promise<Query> => {
      // Check if we have remaining queries before attempting
      if (user && user.daily_query_count >= user.daily_query_limit) {
        throw new Error('Daily query limit reached');
      }

      try {
        console.log('📤 Sending query:', { ...data });
        
        // Start progress tracking
        setProcessingStartTime(Date.now());
        setProgressMessages(['🔍 Initializing search...']);
        setProcessingQuery(data.query_text);
        
        // Use streaming API instead of hardcoded messages
        return new Promise<Query>((resolve, reject) => {
          apiQueries.createStream(
            { ...data },
            // onProgress callback
            (message: string) => {
              setProgressMessages(prev => [...prev, message]);
            },
            // onComplete callback
            (query: Query) => {
              console.log('📥 Received streaming response:', query);
              
              // Validate and normalize response structure
              if (!query || !query.response) {
                console.error('Invalid response structure:', query);
                reject(new Error('Invalid response from server'));
                return;
              }

              // Ensure arguments array exists and is properly structured
              if (!Array.isArray(query.response.arguments)) {
                console.error('Invalid arguments structure:', query.response);
                reject(new Error('Invalid arguments structure from server'));
                return;
              }

              // Update user data immediately after successful query
              if (user) {
                user.daily_query_count = (user.daily_query_count || 0) + 1;
              }

              resolve(query);
            },
            // onError callback
            (error: string) => {
              console.error('❌ Streaming query error:', error);
              reject(new Error(error));
            }
          );
        });
      } catch (error) {
        console.error('❌ Query error:', error);
        const errorMessage = error instanceof Error 
          ? error.message 
          : 'An error occurred while processing your request';
        throw new Error(errorMessage);
      }
    },
    onSuccess: async (data: Query) => {
      console.log('✅ Query successful:', data);
    
      // Clear progress messages
      setProgressMessages([]);
      setProcessingStartTime(null);
      setProcessingQuery('');
    
      if (!data.response?.arguments) {
        console.error('Missing arguments in response:', data);
        toast({
          title: 'Error',
          description: 'Invalid response format from server',
          variant: 'destructive' as const,
        });
        return;
      }

      // Update UI state
      setCurrentQuery(data);
      setQueryText('');
      setExpandedCards({});
      setQueryList(prev => [data, ...prev]);
      
      // Trigger query limit animation
      setQueryLimitAnimation(true);
      setTimeout(() => setQueryLimitAnimation(false), 1000);
      
      // Get fresh user data to update query count
      try {
        const userData = await auth.getUser();
        if (userData.daily_query_count >= userData.daily_query_limit) {
          const resetTime = new Date();
          resetTime.setHours(24, 0, 0, 0); // Next midnight
          const timeLeft = resetTime.getTime() - Date.now();
          const hoursLeft = Math.ceil(timeLeft / (1000 * 60 * 60));
          
          toast({
            title: 'Daily query limit reached',
            description: `Please try again in ${hoursLeft} hours when your limit resets.`,
            variant: 'warning' as const,
          });
        }
      } catch (error) {
        console.error('Failed to refresh user data:', error);
      }
    },
    onError: (error: any) => {
      console.error('❌ Query failed:', error);
      
      // Clear progress messages
      setProgressMessages([]);
      setProcessingStartTime(null);
      setProcessingQuery('');
      
      if (error.message === 'Daily query limit reached') {
        const resetTime = new Date();
        resetTime.setHours(24, 0, 0, 0); // Next midnight
        const timeLeft = resetTime.getTime() - Date.now();
        const hoursLeft = Math.ceil(timeLeft / (1000 * 60 * 60));
        
        toast({
          title: 'Daily query limit reached',
          description: `Please try again in ${hoursLeft} hours when your limit resets.`,
          variant: 'destructive' as const,
        });
      } else if (error.response?.status === 401) {
        toast({
          title: 'Authentication required',
          description: 'Please log in to continue.',
          variant: 'destructive' as const,
        });
        router.push('/login');
      } else {
        toast({
          title: 'Error',
          description: error.message || 'An error occurred while processing your query. Please try again.',
          variant: 'destructive' as const,
        });
      }
    }
  });

  const submitQuery = (text: string) => {
    if (!text.trim()) return;
    
    // Prevent submission if mutation is in progress or query limit reached
    if (mutation.isPending || (user && user.daily_query_count >= user.daily_query_limit)) {
      if (user && user.daily_query_count >= user.daily_query_limit) {
        toast({
          title: 'Daily query limit reached',
          description: `Please try again in ${formatTimeRemaining(user?.reset_time ?? null)} when your limit resets.`,
          variant: 'warning' as const,
        });
      }
      return;
    }
    
    console.log('🚀 Submitting query:', text);
    
    // Clear previous query results to show a blank screen for new progress updates
    setCurrentQuery(null);
    
    mutation.mutate({
      query_text: text,
      diversity_score: diversityScore[0],
      num_stances: stanceCount
    });
    
    // Clear any existing text selection
    window.getSelection()?.removeAllRanges();
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryText.trim()) return;
    submitQuery(queryText);
  };

  const handleExampleClick = (example: string) => {
    submitQuery(example);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      // Only proceed if we haven't reached the query limit and not loading
      if (!isQueryLimitReached() && !loading) {
        handleSubmit(e);
      } else {
        toast({
          title: 'Daily query limit reached',
          description: `Please try again in ${formatTimeRemaining(user?.reset_time ?? null)} when your limit resets.`,
          variant: 'warning' as const,
        });
      }
    }
  };

  const highlightText = (text: string, searchTerm: string) => {
    if (!searchTerm?.trim()) return text;
    
    // Escape special characters in the search term
    const escapedSearchTerm = searchTerm.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    
    // Use word boundaries to prevent partial word matches
    const parts = text.split(new RegExp(`(${escapedSearchTerm})`, 'gi'));
    
    return parts.map((part, index) => 
      part.toLowerCase() === searchTerm.trim().toLowerCase() ? 
        <mark 
          key={index} 
          className="bg-gradient-to-r from-yellow-200/70 to-amber-200/70 text-gray-900 not-italic rounded px-1 py-0.5 -mx-1 animate-highlight"
        >
          {part}
        </mark> : 
        part
    );
  };

  const extractUrlAndDomain = (text: string): { text: string; url?: string; domain?: string } => {
    const urlRegex = /https?:\/\/([^\/\s]+)[^\s]*$/;
    const match = text.match(urlRegex);
    
    if (!match) return { text };
    
    const url = match[0];
    const fullDomain = match[1]; // e.g., www.npr.org or npr.org
    const domain = fullDomain.replace(/^www\./, '');
    const cleanText = text.replace(url, '').trim();
    
    return { text: cleanText, url, domain };
  };

  const renderSupportingArgument = (support: string, idx: number) => {
    const { text, url, domain } = extractUrlAndDomain(support);
    
    return (
      <motion.div 
        key={idx}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: idx * 0.1 }}
        className="group flex items-start gap-3 p-3 rounded-lg hover:bg-purple-50/50 transition-all duration-300"
      >
        <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600 group-hover:scale-125 transition-transform duration-300" />
        <div className="flex-1">
          <p className="text-gray-700 text-sm leading-relaxed">
            {highlightText(text, searchTerm)}
          </p>
          {url && domain && (
            <motion.a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 mt-1 text-sm font-medium text-indigo-600 hover:text-indigo-800 group-hover:underline"
              onClick={(e) => e.stopPropagation()}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <LinkIcon className="h-3.5 w-3.5" />
              <span>{domain}</span>
            </motion.a>
          )}
        </div>
      </motion.div>
    );
  };

  const renderArguments = (results: GPTResponse | undefined | null) => {
    if (!results || !results.arguments) return null;

    const argumentsList = results.arguments;
    if (!Array.isArray(argumentsList)) return null;

    return argumentsList.map((argument: any, index: number) => {
      // Validate and normalize argument structure
      const normalizedArgument = {
        stance: String(argument.stance || 'Unknown Stance'),
        core_argument: String(argument.core_argument || argument.text || 'No core argument provided'),
        supporting_arguments: Array.isArray(argument.supporting_arguments) 
          ? argument.supporting_arguments.map((arg: string | TextObject) => 
              typeof arg === 'object' && arg !== null ? String(arg.text || '') : String(arg))
          : ['No supporting arguments provided']
      };

      return (
        <div key={index} className="bg-gray-50 rounded-lg shadow p-4 transition-all duration-300 ease-in-out">
          {/* Stance */}
          <h3 className="text-lg font-bold text-gray-900 mb-2">
            {normalizedArgument.stance}
          </h3>
          
          {/* Core Argument with Purple Line */}
          <div className="mb-2">
            <p className="text-gray-700 font-medium select-text">
              {normalizedArgument.core_argument}
            </p>
            <div className="h-0.5 bg-purple-500 my-2"></div>
          </div>
          
          {/* Learn More and Expand Button */}
          <div 
            className="flex items-center justify-between cursor-pointer hover:bg-gray-100 p-2 rounded-md"
            onClick={() => toggleCard(index)}
          >
            <span className="text-sm text-purple-600">Learn more</span>
            {expandedCards[index] ? (
              <ChevronUpIcon className="h-5 w-5 text-purple-600" />
            ) : (
              <ChevronDownIcon className="h-5 w-5 text-purple-600" />
            )}
          </div>
          
          {/* Supporting Arguments */}
          {expandedCards[index] && normalizedArgument.supporting_arguments && (
            <div className="mt-2 space-y-2 transition-all duration-300 ease-in-out user-select-text">
              {normalizedArgument.supporting_arguments.map((support: string, sIndex: number) => (
                <p 
                  key={sIndex} 
                  className="text-gray-600 text-sm pl-4 border-l-2 border-purple-200 cursor-text whitespace-pre-wrap"
                >
                  {support}
                </p>
              ))}
            </div>
          )}
        </div>
      );
    }).filter(Boolean);  // Remove any null elements from invalid arguments
  };

  // Update query rendering to use response field
  const renderQueryResponse = (query: QueryData) => {
    console.log('Rendering query response:', query);
    
    if (!query.response?.arguments || !Array.isArray(query.response.arguments)) {
      console.error('Invalid query response structure:', query);
      return (
        <div className="w-full text-center py-8">
          <p className="text-gray-600">No arguments available for this query.</p>
        </div>
      );
    }

    const queryTopics = getQueryTopics(query.query_text);
    
    return (
      <div className="flex flex-col items-center w-full">
        {/* Query Title */}
        <div className="w-full mb-8 text-center">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent inline-block">
            Results
          </h2>
          <p className="mt-2 text-gray-600 text-lg italic">"{query.query_text}"</p>
        </div>

        {/* Topic Tags */}
        {queryTopics.length > 0 && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-wrap gap-3 justify-center mb-12"
          >
            {queryTopics.map((topic, index) => (
              <motion.span
                key={index}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ scale: 1.05 }}
                transition={{ 
                  type: "spring",
                  stiffness: 400,
                  damping: 25,
                  delay: index * 0.1 
                }}
                className="px-6 py-2 bg-gradient-to-r from-purple-50 to-indigo-50 text-purple-700 rounded-full text-sm font-medium border border-purple-100 shadow-sm hover:shadow-md transition-shadow duration-300 cursor-default"
              >
                {topic}
              </motion.span>
            ))}
          </motion.div>
        )}
        
        {/* Cards Grid */}
        <div className={`grid gap-8 w-full max-w-7xl mx-auto ${
          query.response.arguments.length <= 2 ? 'grid-cols-1 md:grid-cols-2' :
          query.response.arguments.length <= 4 ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4' :
          'grid-cols-1 md:grid-cols-2 lg:grid-cols-4'
        }`}>
          {query.response.arguments.map((argument, index) => {
            console.log('Rendering argument:', argument);
            return (
              <motion.div 
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ 
                  type: "spring",
                  stiffness: 300,
                  damping: 30,
                  delay: index * 0.1 
                }}
                className="relative group"
              >
                <div className={`relative bg-[#FDFCFB]/95 backdrop-blur-sm rounded-2xl shadow-md hover:shadow-lg transition-all duration-300 p-6 h-full border border-gray-100 ${
                  query.response.arguments.length > 4 ? 'text-sm' : ''
                }`}>
                  {/* Stance Header */}
                  <div className="flex items-center gap-3 mb-4">
                    <h3 className={`font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent ${
                      query.response.arguments.length > 4 ? 'text-lg' : 'text-xl'
                    }`}>
                      {argument.stance || 'Unknown Stance'}
                    </h3>
                    <div className="flex-grow h-px bg-gradient-to-r from-purple-200 to-transparent"></div>
                  </div>

                  {/* Spacer - position is already shown in header */}
                  <div className="mb-6">
                    <div className="h-0.5 bg-gradient-to-r from-purple-500/30 to-indigo-500/30 w-full rounded-full"></div>
                  </div>

                  {/* Learn More Button */}
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => openModal(argument)}
                    className={`w-full mt-auto flex items-center justify-center gap-2 text-white bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 rounded-xl transition-all duration-300 font-medium shadow-lg hover:shadow-xl group ${
                      query.response.arguments.length > 4 ? 'p-2 text-sm' : 'p-3'
                    }`}
                  >
                    <span>Learn more</span>
                    <ChevronDownIcon className="h-5 w-5 transform group-hover:translate-y-0.5 transition-transform duration-300" />
                  </motion.button>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    );
  };

  const deleteQuery = async (queryId: number) => {
    try {
      await apiQueries.delete(queryId);
      // Remove from local state
      setHiddenQueries(prev => new Set([...prev, queryId]));
      // Invalidate queries to refresh the list
      queryClient.invalidateQueries({ queryKey: ['queries'] });
      setQueryList(prev => prev.filter(q => q.id !== queryId));
      toast({
        title: 'Query deleted',
        description: 'The query has been removed from your history.',
      });
    } catch (error) {
      console.error('Failed to delete query:', error);
      toast({
        title: 'Error deleting query',
        description: 'Failed to delete the query. Please try again.',
        variant: 'destructive' as const,
      });
    }
  };

  const clearHistory = async () => {
    try {
      await apiQueries.deleteAll();
      // Invalidate queries to refresh the list
      queryClient.invalidateQueries({ queryKey: ['queries'] });
      setQueryList([]);
      toast({
        title: 'All queries cleared',
        description: 'Your query history has been cleared.',
      });
    } catch (error) {
      console.error('Failed to clear history:', error);
      toast({
        title: 'Error clearing queries',
        description: 'Failed to clear your query history. Please try again.',
        variant: 'destructive' as const,
      });
    }
  };

  const filteredQueries = userQueries.filter((q: Query) => {
    // Filter out current query and hidden queries
    if (currentQuery && q.id === currentQuery.id) return false;
    if (hiddenQueries.has(q.id)) return false;
    
    // Apply group filter
    if (selectedGroups.size > 0) {
      const queryTopics = getQueryTopics(q.query_text);
      return queryTopics.some(topic => selectedGroups.has(topic));
    }

    return true;
  });

  const handleGroupClick = (group: string) => {
    setSelectedGroups(prev => {
      const newGroups = new Set(prev);
      if (newGroups.has(group)) {
        newGroups.delete(group);
      } else {
        newGroups.add(group);
      }
      return newGroups;
    });
  };

  const handleInvestigate = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (!selectionPopup.text.trim() || mutation.isPending) return;
    
    // Update the query text
    setQueryText(selectionPopup.text);
    
    // Use submitQuery instead of directly calling mutation.mutate
    submitQuery(selectionPopup.text);
    
    // Clear the selection and hide popup
    window.getSelection()?.removeAllRanges();
    setSelectionPopup(prev => ({ ...prev, show: false }));
    
    // Prevent the mouseup event from firing
    e.nativeEvent.stopImmediatePropagation();
  }, [selectionPopup.text, mutation.isPending, submitQuery]);

  const SelectionPopup = () => {
    useEffect(() => {
      let isInvestigating = false;

      const updatePopupPosition = () => {
        const selection = window.getSelection();
        if (!selection || selection.isCollapsed || !selectionPopup.show) return;

        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        const contentContainer = document.querySelector('.max-w-7xl');
        
        if (contentContainer) {
          const containerRect = contentContainer.getBoundingClientRect();
          const x = rect.left + (rect.width / 2);
          const y = rect.top - containerRect.top - 45;
          
          setSelectionPopup(prev => ({
            ...prev,
            x,
            y
          }));
        }
      };

      const handleSelection = () => {
        if (isInvestigating) {
          isInvestigating = false;
          return;
        }

        const selection = window.getSelection();
        if (!selection || selection.isCollapsed) {
          setSelectionPopup(prev => ({ ...prev, show: false }));
          return;
        }

        const text = selection.toString().trim();
        if (!text) {
          setSelectionPopup(prev => ({ ...prev, show: false }));
          return;
        }

        // Check if selection is within an argument text
        const range = selection.getRangeAt(0);
        const container = range.commonAncestorContainer;
        let element = container.nodeType === 3 ? container.parentElement : container as Element;
        let isArgumentText = false;
        
        while (element && !isArgumentText) {
          if (element.classList?.contains('argument-text')) {
            isArgumentText = true;
            break;
          }
          element = element.parentElement;
        }

        if (!isArgumentText) {
          setSelectionPopup(prev => ({ ...prev, show: false }));
          return;
        }

        const rect = range.getBoundingClientRect();
        const contentContainer = document.querySelector('.max-w-7xl');
        
        if (contentContainer) {
          const containerRect = contentContainer.getBoundingClientRect();
          const x = rect.left + (rect.width / 2);
          const y = rect.top - containerRect.top - 45;

          setSelectionPopup({
            show: true,
            x,
            y,
            text
          });
        }
      };

      // Only hide popup on mousedown outside the popup
      const handleMouseDown = (e: MouseEvent) => {
        const target = e.target as HTMLElement;
        if (target.closest('.investigate-button')) {
          isInvestigating = true;
        } else if (!target.closest('.investigate-button')) {
          setSelectionPopup(prev => ({ ...prev, show: false }));
        }
      };

      // Update position on scroll
      const handleScroll = () => {
        requestAnimationFrame(updatePopupPosition);
      };

      document.addEventListener('mouseup', handleSelection);
      document.addEventListener('mousedown', handleMouseDown);
      window.addEventListener('scroll', handleScroll, { passive: true });

      return () => {
        document.removeEventListener('mouseup', handleSelection);
        document.removeEventListener('mousedown', handleMouseDown);
        window.removeEventListener('scroll', handleScroll);
      };
    }, []);

    if (!selectionPopup.show) return null;

    return createPortal(
      <div
        className="investigate-button fixed z-50"
        style={{
          left: `${selectionPopup.x}px`,
          top: `${window.scrollY + selectionPopup.y}px`,
          transform: 'translateX(-50%)'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={handleInvestigate}
          className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-full transition-colors duration-150 flex items-center space-x-2 cursor-pointer select-none shadow-lg"
        >
          <MagnifyingGlassIcon className="h-4 w-4" />
          <span>Investigate</span>
        </button>
      </div>,
      document.querySelector('.max-w-7xl') || document.body
    );
  };

  const { data: queryUserStats, isLoading: isLoadingStats, error: statsError } = useQuery({
    queryKey: ['user-stats'],
    queryFn: async () => {
      try {
        return await stats.get();
      } catch (error) {
        console.error('Stats fetch error:', error);
        throw error;
      }
    },
    enabled: !!user,
    staleTime: 60000, // Consider data fresh for 1 minute
    retry: 0 // Prevent automatic retries
  });

  // Add history sidebar state
  const [isHistorySidebarOpen, setIsHistorySidebarOpen] = useState(true);

  // Add effect for auto-closing sidebar
  useEffect(() => {
    if (!isHistorySidebarOpen) return;

    const checkInactivity = () => {
      const now = Date.now();
      if (now - lastMouseActivity > 2000) { // 2 seconds
        setIsHistorySidebarOpen(false);
      }
    };

    const timer = setInterval(checkInactivity, 500);
    return () => clearInterval(timer);
  }, [isHistorySidebarOpen, lastMouseActivity]);

  // Add mouse activity handler
  const handleMouseMove = () => {
    setLastMouseActivity(Date.now());
  };

  const toggleCard = (index: number) => {
    setExpandedCards(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const openModal = (argument: any) => {
    // Debug: Log argument structure (remove in production)
    // console.log('🔍 Opening modal with argument:', argument);
    // console.log('🔍 Evidence metadata:', argument.evidence_metadata);
    // console.log('🔍 Detailed evidence:', argument.detailed_evidence);
    
    // Ensure supporting_arguments is always an array and include references
    const normalizedArgument = {
      stance: argument.stance || '',
      supporting_arguments: Array.isArray(argument.supporting_arguments) 
        ? argument.supporting_arguments 
        : argument.supporting_arguments 
          ? [argument.supporting_arguments]
          : [],
      references: argument.references || [],
      evidence_metadata: argument.evidence_metadata,
      detailed_evidence: argument.detailed_evidence
    };
    // console.log('🔍 Normalized argument:', normalizedArgument);
    setSelectedArgument(normalizedArgument);
  };

  const closeModal = () => {
    setSelectedArgument(null);
  };

  const handleAllStarsUpdate = (newCount: number) => {
    // Update the stats in the cache
    queryClient.setQueryData<UserStats>(['user-stats'], (oldData) => 
      oldData ? {
        ...oldData,
        allstars: newCount
      } : undefined
    );
    
    setShowAllStarsAnimation(true);
    setTimeout(() => setShowAllStarsAnimation(false), 1000);
  };

  // Add this query to fetch leaderboard data
  const { data: leaderboardData } = useQuery({
    queryKey: ['leaderboard'],
    queryFn: leaderboard.get,
    enabled: showLeaderboard,
    staleTime: 60000 // Consider data fresh for 1 minute
  });

  useEffect(() => {
    if (leaderboardData) {
      setLeaderboardUsers(leaderboardData);
    }
  }, [leaderboardData]);

  // Add effect to handle query limit animation
  useEffect(() => {
    if (mutation.isSuccess) {
      setQueryLimitAnimation(true);
      setTimeout(() => setQueryLimitAnimation(false), 1000);
    }
  }, [mutation.isSuccess]);

  // Add effect to update elapsed time counter
  const [elapsedTime, setElapsedTime] = useState(0);
  const [processingQuery, setProcessingQuery] = useState('');
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (processingStartTime) {
      interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - processingStartTime) / 1000));
      }, 1000);
    } else {
      setElapsedTime(0);
    }
    return () => clearInterval(interval);
  }, [processingStartTime]);

  const handleStanceCountChange = (count: number) => {
    setStanceCount(count);
    setShowStanceSelector(false);
  };

  // Update effect to check query limit using stats data
  useEffect(() => {
    // Set limit reached while loading to prevent premature enabling
    if (loading) {
      setIsLimitReached(true);
      return;
    }

    // Only use userStats for checking limits since it's the source of truth
    if (userStats) {
      setIsLimitReached(userStats.remaining_queries <= 0);
    } else {
      // If no stats yet, keep disabled
      setIsLimitReached(true);
    }
  }, [userStats, loading]);

  // Simplify the helper function to only use userStats
  const isQueryLimitReached = () => {
    if (loading || !userStats) return true;
    return userStats.remaining_queries <= 0;
  };

  return (
    <div className="min-h-screen bg-gray-50 relative flex">
      {/* Query History Sidebar */}
      <motion.div 
        initial={{ width: 0 }}
        animate={{ width: isHistorySidebarOpen ? '20rem' : 0 }}
        transition={{ type: "spring", damping: 30, stiffness: 300 }}
        onHoverStart={() => window.innerWidth >= 1024 && setIsHistorySidebarOpen(true)}
        onHoverEnd={() => window.innerWidth >= 1024 && setIsHistorySidebarOpen(false)}
        onMouseMove={handleMouseMove}
        className="fixed left-0 top-0 h-full bg-white/95 backdrop-blur-sm border-r border-purple-100 shadow-lg z-40 overflow-hidden"
      >
        <div className="h-full w-80 overflow-y-auto">
          {/* Sidebar Header */}
          <div className="sticky top-0 bg-white/90 backdrop-blur-sm border-b border-purple-100 p-4 flex justify-between items-center">
            <h2 className="text-lg font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
              History
            </h2>
            <button
              onClick={() => setIsHistorySidebarOpen(false)}
              className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <XMarkIcon className="h-5 w-5 text-gray-500" />
            </button>
          </div>

          {/* History List */}
          <div className="p-4 space-y-4">
            {userQueries.map((query: Query) => (
              <motion.div
                key={query.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="group p-4 rounded-xl border border-purple-100 hover:border-purple-200 transition-all duration-300 cursor-pointer bg-gray-50"
                onClick={() => {
                  setCurrentQuery(query);
                  setQueryText('');
                }}
              >
                <div className="space-y-2">
                  <p className="text-sm text-gray-600 line-clamp-2 group-hover:text-gray-900 transition-colors">
                    {query.query_text}
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-400">
                      {new Date(query.created_at).toLocaleDateString()}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteQuery(query.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-50 rounded transition-all duration-200"
                    >
                      <TrashIcon className="h-4 w-4 text-red-500" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Hover Area for Desktop */}
      <motion.div
        className="fixed left-0 top-0 w-4 h-full z-30 hidden lg:block"
        onHoverStart={() => setIsHistorySidebarOpen(true)}
      />

      {/* Toggle Sidebar Button */}
      <motion.button
        onClick={() => setIsHistorySidebarOpen(true)}
        initial={{ opacity: 0, x: -20 }}
        animate={{ 
          opacity: isHistorySidebarOpen ? 0 : 1,
          x: isHistorySidebarOpen ? -20 : 0,
          pointerEvents: isHistorySidebarOpen ? 'none' : 'auto'
        }}
        className="fixed left-4 top-4 z-30 p-2 bg-white rounded-lg shadow-md hover:shadow-lg transition-all duration-300"
      >
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          className="h-6 w-6 text-gray-600" 
          fill="none" 
          viewBox="0 0 24 24" 
          stroke="currentColor"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
          />
        </svg>
      </motion.button>

      {/* Main Content */}
      <div className={`flex-1 transition-all duration-300 ${isHistorySidebarOpen ? 'ml-80' : 'ml-0'}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-32">
          {/* Welcome Watermark or Progress Messages */}
          {!currentQuery && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <AnimatePresence mode="wait">
                {mutation.isPending && progressMessages.length > 0 ? (
                  <motion.div
                    key="progress"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="text-center flex flex-col items-center gap-6 max-w-2xl mx-auto px-8"
                  >
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="h-16 w-16"
                    >
                      <Loader2 className="h-16 w-16 text-purple-600" />
                    </motion.div>
                    
                    <div className="space-y-3">
                      <h2 className="text-2xl font-semibold text-gray-700">Processing your query...</h2>
                      <p className="text-lg text-gray-600">"{processingQuery || 'Your query'}"</p>
                      
                      {/* Fixed Height Progress Stream */}
                      <div className="mt-6 h-20 overflow-hidden relative bg-gray-50/50 rounded-xl px-4">
                        <div className="flex flex-col justify-end h-full py-2">
                          {progressMessages.slice(-3).map((message, index) => {
                            const totalMessages = progressMessages.slice(-3).length;
                            const messageIndex = totalMessages - 1 - index;
                            
                            // Opacity: most recent (index 0) = 1.0, second recent = 0.8, oldest = 0.3
                            const opacity = messageIndex === 0 ? 1.0 : messageIndex === 1 ? 0.8 : 0.3;
                            
                            return (
                              <motion.div
                                key={progressMessages.length - 3 + index}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ 
                                  opacity: opacity,
                                  y: 0
                                }}
                                exit={{ opacity: 0, y: -8 }}
                                transition={{ 
                                  duration: 0.3,
                                  ease: "easeOut"
                                }}
                                className={`flex items-center gap-2 text-sm transition-all duration-300 ${
                                  messageIndex === 0 
                                    ? 'text-purple-700 font-medium' 
                                    : messageIndex === 1 
                                      ? 'text-gray-600' 
                                      : 'text-gray-400'
                                }`}
                                style={{
                                  transform: `translateY(${messageIndex * -2}px)`,
                                }}
                              >
                                <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                                  messageIndex === 0 
                                    ? 'bg-purple-500 animate-pulse' 
                                    : messageIndex === 1 
                                      ? 'bg-gray-400' 
                                      : 'bg-gray-300'
                                }`} />
                                <span className="leading-tight">{message}</span>
                              </motion.div>
                            );
                          })}
                        </div>
                        
                        {/* Subtle gradient fade effect */}
                        <div className="absolute top-0 left-0 right-0 h-3 bg-gradient-to-b from-gray-50/80 to-transparent pointer-events-none" />
                      </div>
                      
                      {processingStartTime && (
                        <div className="mt-6 text-sm text-gray-500">
                          Elapsed time: {elapsedTime}s
                        </div>
                      )}
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="welcome"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 0.7, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="text-center flex flex-col items-center gap-8"
                  >
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 0.7 }}
                      className="h-24 sm:h-28 md:h-32 lg:h-40 aspect-square relative"
                    >
                      <Image
                        src="/logo.webp"
                        alt="AllStances Logo"
                        fill
                        className="object-contain"
                      />
                    </motion.div>
                    <div>
                      <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-300">
                        Hi, {user?.first_name || user?.username || 'there'}!
                      </h1>
                      <p className="mt-4 text-lg sm:text-xl md:text-2xl text-gray-300">
                        Welcome to AllStances!
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* Top Bar with AllStars and Sign Out */}
          <div className="fixed top-4 right-4 z-50 flex items-center gap-4">
            {/* AllStars Counter */}
            <motion.button
              onClick={() => setShowLeaderboard(true)}
              initial={{ opacity: 0, y: -10 }}
              animate={{ 
                opacity: 1, 
                y: 0,
                scale: showAllStarsAnimation ? [1, 1.2, 1] : 1,
              }}
              transition={{
                duration: 0.5,
                ease: "easeInOut"
              }}
              className="flex items-center gap-2 text-purple-600 font-medium bg-white/80 backdrop-blur-sm px-4 py-2 rounded-lg shadow-sm border border-purple-100 hover:bg-white/90 hover:shadow-md transition-all duration-200"
            >
              <motion.div
                animate={{
                  rotate: showAllStarsAnimation ? [0, 360] : 0,
                  scale: showAllStarsAnimation ? [1, 1.2, 1] : 1,
                }}
                transition={{
                  duration: 0.5,
                  ease: "easeInOut"
                }}
              >
                <StarIcon 
                  className={`h-5 w-5 transition-colors ${
                    showAllStarsAnimation ? 'text-yellow-500' : 'text-purple-600'
                  }`}
                />
              </motion.div>
              <motion.span
                animate={{
                  scale: showAllStarsAnimation ? [1, 1.1, 1] : 1,
                }}
                className={`transition-colors ${
                  showAllStarsAnimation ? 'text-yellow-600' : 'text-purple-600'
                }`}
              >
                {queryUserStats?.allstars || 0} AllStars
              </motion.span>
            </motion.button>

            {/* Sign Out Button */}
            <button
              onClick={() => logout()}
              className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 bg-white/80 hover:bg-white/90 rounded-lg shadow-sm hover:shadow border border-gray-200 transition-all duration-200 flex items-center gap-2"
            >
              <span>Sign Out</span>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M3 3a1 1 0 00-1 1v12a1 1 0 001 1h12a1 1 0 001-1V4a1 1 0 00-1-1H3zm11 4.414l-4.293 4.293a1 1 0 01-1.414 0L4 7.414V15h12V7.414z" clipRule="evenodd" />
              </svg>
            </button>
          </div>

          {/* Results Section */}
          {currentQuery && currentQuery.response && (
            <div className="space-y-6 py-12">
              <ArgumentsDisplay
                query={currentQuery.query_text}
                arguments={currentQuery.response.arguments}
                queryId={currentQuery.id}
                onAllStarsUpdate={handleAllStarsUpdate}
                globalReferences={currentQuery.response.references}
                followUpQuestions={currentQuery.response.follow_up_questions}
                onNewQuery={(question) => submitQuery(question)}
              />
            </div>
          )}

          {/* Query History Section */}
          {showHistory && userQueries && userQueries.length > 0 && (
            <div className="mt-12">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-gray-900">History</h2>
                <button
                  onClick={clearHistory}
                  className="text-red-600 hover:text-red-700"
                >
                  Clear History
                </button>
              </div>
              <div className="space-y-4">
                {userQueries.map((query: Query) => (
                  <QueryHistoryCard
                    key={query.id}
                    query={query}
                    onDelete={() => deleteQuery(query.id)}
                    searchTerm={searchTerm}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Supporting Arguments Modal */}
          {selectedArgument && (
            <SupportingArgumentsModal
              isOpen={!!selectedArgument}
              onClose={() => setSelectedArgument(null)}
              stance={selectedArgument.stance}
              supportingArguments={selectedArgument.supporting_arguments}
              references={selectedArgument.references}
              queryId={currentQuery?.id || 0}
              onAllStarsUpdate={handleAllStarsUpdate}
              evidenceMetadata={selectedArgument.evidence_metadata}
              detailedEvidence={selectedArgument.detailed_evidence}
              core_argument_summary={selectedArgument.core_argument_summary}
              source_analysis={selectedArgument.source_analysis}
            />
          )}

          {/* Selection Popup */}
          <SelectionPopup />

          {/* Leaderboard Modal */}
          <LeaderboardModal
            isOpen={showLeaderboard}
            onClose={() => setShowLeaderboard(false)}
            users={leaderboardUsers}
          />
        </div>
      </div>

      {/* Minimalist Search Bar */}
      <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50 w-full max-w-2xl px-4">
        <form onSubmit={handleSubmit} className="relative group">
          <input
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="What would you like to learn about today?"
            className="w-full pl-4 pr-[7rem] sm:pr-[9rem] py-3 sm:py-4 bg-white/95 backdrop-blur-sm rounded-2xl shadow-lg border border-purple-100/50 focus:border-purple-300/50 focus:ring-2 focus:ring-purple-500/30 transition-all duration-300 outline-none text-gray-700 text-sm sm:text-base"
          />
          {/* Information Message */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 flex items-center whitespace-nowrap">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3 h-3 text-gray-400/70 flex-shrink-0 translate-y-[0.5px]">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clipRule="evenodd" />
            </svg>
            <span className="text-[10px] text-gray-400/70 font-light ml-1.5">AllStances can make mistakes. Please double check all Information</span>
          </div>
          
          <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 sm:gap-3">
            {user && (
              <>

                <motion.div
                  animate={{
                    scale: queryLimitAnimation ? [1, 0.9, 1.1, 1] : 1,
                    opacity: queryLimitAnimation ? [1, 0.7, 1] : 1
                  }}
                  transition={{ duration: 0.4 }}
                  className="hidden sm:block px-2 py-1 bg-purple-50 rounded-lg border border-purple-100"
                >
                  <motion.span 
                    className="text-xs sm:text-sm font-medium bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent"
                    animate={{
                      y: queryLimitAnimation ? [0, -10, 0] : 0
                    }}
                    transition={{ duration: 0.4 }}
                  >
                    {userStats ? (
                      <>
                        {userStats.remaining_queries}/{userStats.daily_query_limit || 20}
                        {userStats.remaining_queries <= 0 && userStats.reset_time && (
                          <span className="ml-2 text-red-500">
                            Reset in {formatTimeRemaining(userStats.reset_time)}
                          </span>
                        )}
                      </>
                    ) : '0/0'}
                  </motion.span>
                </motion.div>
              </>
            )}
            
            <Button 
              type="submit"
              className={cn(
                "h-8 px-3 sm:h-auto sm:px-4",
                queryLimitAnimation && "animate-pulse",
                (isQueryLimitReached() || loading)
                  ? "opacity-50 cursor-not-allowed bg-gray-300 hover:bg-gray-300"
                  : "bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700"
              )}
              disabled={mutation.isPending || isQueryLimitReached() || loading}
            >
              {mutation.isPending ? (
                <Loader2 className="h-4 w-4 sm:mr-2 animate-spin" />
              ) : (
                <Send className={cn(
                  "h-4 w-4 sm:mr-2",
                  isQueryLimitReached() ? "text-gray-500" : "text-white"
                )} />
              )}
              <span className={cn(
                "hidden sm:inline",
                isQueryLimitReached() ? "text-gray-500" : "text-white"
              )}>
                {mutation.isPending ? "Processing..." : 
                 isQueryLimitReached()
                   ? `Resets in ${formatTimeRemaining(user?.reset_time ?? null)}` 
                   : "Submit"}
              </span>
            </Button>
          </div>
        </form>
      </div>

      {/* Toast Notifications */}
      <Toaster />
    </div>
  );
}
