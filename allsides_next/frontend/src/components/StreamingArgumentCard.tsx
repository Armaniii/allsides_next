import React from 'react';
import { motion } from 'framer-motion';
import { StreamingArgument } from '@/lib/api';

interface StreamingArgumentCardProps {
  argument: StreamingArgument;
  isStreaming?: boolean;
  onClick?: () => void;
}

export const StreamingArgumentCard: React.FC<StreamingArgumentCardProps> = ({ 
  argument, 
  isStreaming = false, 
  onClick 
}) => {
  const isComplete = argument.isComplete;
  const hasStance = !!argument.stance;
  const hasCoreArgument = !!argument.core_argument;
  const hasSupportingArguments = argument.supporting_arguments && argument.supporting_arguments.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={`p-6 bg-white rounded-lg shadow-sm border ${
        isComplete ? 'border-green-200' : 'border-gray-200'
      } ${onClick ? 'cursor-pointer' : ''}`}
      onClick={onClick}
    >
      {/* Stance */}
      {hasStance && (
        <div className="mb-4">
          <h3 className="text-xl font-semibold text-gray-900">{argument.stance}</h3>
        </div>
      )}

      {/* Core Argument */}
      {hasCoreArgument && (
        <div className="mb-4">
          <h4 className="text-md font-medium text-gray-700 mb-2">Core Argument</h4>
          <p className="text-gray-600">{argument.core_argument}</p>
        </div>
      )}

      {/* Supporting Arguments */}
      {hasSupportingArguments && (
        <div>
          <h4 className="text-md font-medium text-gray-700 mb-2">Supporting Arguments</h4>
          <ul className="space-y-2">
            {argument.supporting_arguments?.map((arg, index) => (
              <motion.li
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="text-gray-600 pl-4 border-l-2 border-purple-200"
              >
                {arg}
              </motion.li>
            ))}
          </ul>
        </div>
      )}

      {/* Loading Indicator */}
      {isStreaming && !isComplete && (
        <div className="mt-4 flex items-center text-sm text-gray-500">
          <div className="animate-pulse flex space-x-2 items-center">
            <div className="h-2 w-2 bg-purple-500 rounded-full"></div>
            <div className="h-2 w-2 bg-purple-500 rounded-full animation-delay-200"></div>
            <div className="h-2 w-2 bg-purple-500 rounded-full animation-delay-400"></div>
          </div>
          <span className="ml-2">Generating response...</span>
        </div>
      )}
    </motion.div>
  );
}; 