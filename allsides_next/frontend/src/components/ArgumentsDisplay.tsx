import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronDownIcon } from '@heroicons/react/24/outline';
import { ArgumentCard } from './ArgumentCard';
import { SupportingArgumentsModal } from './SupportingArgumentsModal';
import { FollowUpQuestions } from './FollowUpQuestions';

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
  references?: any[]; // Add support for references
  evidence_metadata?: EvidenceMetadata;
  detailed_evidence?: DetailedEvidence;
  core_argument_summary?: string;
  key_perspectives?: string[];
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
  };
}

interface ArgumentsDisplayProps {
  query: string;
  arguments: Stance[];
  queryId: number;
  onAllStarsUpdate: (count: number) => void;
  globalReferences?: any[]; // Add global references from the response
  followUpQuestions?: string[]; // Add follow-up questions from the response
  onNewQuery?: (query: string) => void; // Callback for when user clicks a follow-up question
}

export const ArgumentsDisplay: React.FC<ArgumentsDisplayProps> = ({ query, arguments: stances, queryId, onAllStarsUpdate, globalReferences, followUpQuestions, onNewQuery }) => {
  const [selectedArgument, setSelectedArgument] = useState<Stance | null>(null);
  const [visibleStancesCount, setVisibleStancesCount] = useState<number>(4);

  const handleLearnMore = (argument: Stance) => {
    // Merge individual argument references with global references
    const mergedReferences = [
      ...(argument.references || []),
      ...(globalReferences || [])
    ];
    
    // Remove duplicates based on URL
    const uniqueReferences = mergedReferences.filter((ref, index, self) => 
      index === self.findIndex(r => r.url === ref.url)
    );
    
    setSelectedArgument({
      ...argument,
      references: uniqueReferences
    });
  };

  const handleCloseModal = () => {
    setSelectedArgument(null);
  };

  const handleShowMoreStances = () => {
    setVisibleStancesCount(prev => Math.min(prev + 3, stances.length));
  };

  const handleFollowUpQuestion = (question: string) => {
    if (onNewQuery) {
      onNewQuery(question);
    }
  };

  if (!stances || stances.length === 0) {
    return null;
  }

  // Calculate visible stances and remaining count
  const visibleStances = stances.slice(0, visibleStancesCount);
  const remainingStances = stances.length - visibleStancesCount;
  const hasMoreStances = remainingStances > 0;

  // Determine grid columns based on number of visible stances
  const gridCols = visibleStances.length <= 2 ? 
    'grid-cols-1 md:grid-cols-2' : 
    visibleStances.length <= 3 ? 
    'grid-cols-1 md:grid-cols-3' : 
    'grid-cols-1 md:grid-cols-2 lg:grid-cols-4';

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex flex-col items-center">
        {/* Query Title */}
        <div className="w-full mb-12 text-center">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent inline-block">
            AllStances For "{query}"
          </h2>
          {stances.length > 4 && (
            <p className="text-gray-600 mt-2">
              Showing {visibleStancesCount} of {stances.length} positions
            </p>
          )}
        </div>

        {/* Arguments Grid */}
        <div className={`grid ${gridCols} gap-6 md:gap-8 w-full`}>
          {visibleStances.map((argument, index) => (
            <ArgumentCard
              key={index}
              argument={argument}
              onLearnMore={(enhancedArgument) => handleLearnMore(enhancedArgument || argument)}
            />
          ))}
        </div>

        {/* Show More Stances Button */}
        {hasMoreStances && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full flex justify-center mt-8"
          >
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleShowMoreStances}
              className="flex items-center justify-center gap-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-8 py-4 rounded-2xl font-semibold shadow-lg hover:shadow-xl transition-all duration-300 group"
            >
              <span>Show {Math.min(3, remainingStances)} More Position{remainingStances > 1 ? 's' : ''}</span>
              <ChevronDownIcon className="h-5 w-5 transform group-hover:translate-y-0.5 transition-transform duration-300" />
            </motion.button>
          </motion.div>
        )}

        {/* Follow-Up Questions */}
        <FollowUpQuestions 
          questions={followUpQuestions || []}
          onQuestionClick={handleFollowUpQuestion}
          isVisible={true} // Always show if questions exist
        />

        {/* Supporting Arguments Modal */}
        {selectedArgument && (
          <SupportingArgumentsModal
            isOpen={!!selectedArgument}
            onClose={handleCloseModal}
            stance={selectedArgument.stance}
            supportingArguments={selectedArgument.supporting_arguments}
            references={selectedArgument.references}
            queryId={queryId}
            onAllStarsUpdate={onAllStarsUpdate}
            evidenceMetadata={selectedArgument.evidence_metadata}
            detailedEvidence={selectedArgument.detailed_evidence}
            core_argument_summary={selectedArgument.core_argument_summary}
            source_analysis={selectedArgument.source_analysis}
          />
        )}
      </div>
    </div>
  );
}; 