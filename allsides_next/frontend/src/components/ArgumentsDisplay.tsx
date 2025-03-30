import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ArgumentCard } from './ArgumentCard';
import { SupportingArgumentsModal } from './SupportingArgumentsModal';

interface Stance {
  stance: string;
  core_argument: string;
  supporting_arguments: string[];
}

interface ArgumentsDisplayProps {
  query: string;
  arguments: Stance[];
  queryId: number;
  onAllStarsUpdate: (count: number) => void;
}

export const ArgumentsDisplay: React.FC<ArgumentsDisplayProps> = ({ query, arguments: stances, queryId, onAllStarsUpdate }) => {
  const [selectedArgument, setSelectedArgument] = useState<Stance | null>(null);

  const handleLearnMore = (argument: Stance) => {
    setSelectedArgument(argument);
  };

  const handleCloseModal = () => {
    setSelectedArgument(null);
  };

  if (!stances || stances.length === 0) {
    return null;
  }

  // Determine grid columns based on number of stances
  const gridCols = stances.length <= 2 ? 
    'grid-cols-1 md:grid-cols-2' : 
    stances.length <= 3 ? 
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
        </div>

        {/* Arguments Grid */}
        <div className={`grid ${gridCols} gap-6 md:gap-8 w-full`}>
          {stances.map((argument, index) => (
            <ArgumentCard
              key={index}
              argument={argument}
              onLearnMore={() => handleLearnMore(argument)}
            />
          ))}
        </div>

        {/* Supporting Arguments Modal */}
        {selectedArgument && (
          <SupportingArgumentsModal
            isOpen={!!selectedArgument}
            onClose={handleCloseModal}
            stance={selectedArgument.stance}
            coreArgument={selectedArgument.core_argument}
            supportingArguments={selectedArgument.supporting_arguments}
            queryId={queryId}
            onAllStarsUpdate={onAllStarsUpdate}
          />
        )}
      </div>
    </div>
  );
}; 