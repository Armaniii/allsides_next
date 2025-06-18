import React from 'react';
import { motion } from 'framer-motion';
import { ChevronDownIcon } from '@heroicons/react/24/outline';
import { Card } from '@/components/ui/card';

interface Stance {
  stance: string;
  supporting_arguments: string[];
  references?: any[];
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

interface ArgumentCardProps {
  argument: Stance;
  onLearnMore: (enhancedArgument?: Stance) => void;
}

export const ArgumentCard: React.FC<ArgumentCardProps> = ({ argument, onLearnMore }) => {
  // Calculate source quality score and color
  const getSourceQualityColor = (score: number) => {
    if (score >= 80) return 'from-green-400 to-green-600';
    if (score >= 60) return 'from-yellow-400 to-yellow-600';
    return 'from-red-400 to-red-600';
  };

  const getTrustBadgeColor = (score: number) => {
    if (score >= 80) return 'bg-green-100 text-green-700 border-green-200';
    if (score >= 60) return 'bg-yellow-100 text-yellow-700 border-yellow-200';
    return 'bg-red-100 text-red-700 border-red-200';
  };

  const effectiveSourceAnalysis = argument.source_analysis;
  const effectiveCoreSummary = argument.core_argument_summary;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ 
        type: "spring",
        stiffness: 300,
        damping: 30
      }}
      className="relative group"
    >
      <div className="absolute inset-0 bg-gradient-to-r from-purple-50/50 to-indigo-50/50 rounded-2xl blur opacity-25 group-hover:opacity-30 transition-opacity duration-300" />
      <Card className="relative bg-white/95 backdrop-blur-sm rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 p-6 h-full border border-purple-100/50">

        {/* Stance Header */}
        <div className="flex items-center gap-3 mb-4">
          <h3 className="text-xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
            {argument.stance}
          </h3>
          <div className="flex-grow h-px bg-gradient-to-r from-purple-200/50 to-transparent"></div>
        </div>

        {/* Core Argument Summary or Preview */}
        <div className="mb-6">
          {effectiveCoreSummary ? (
            <div>
              <p className="text-gray-700 text-sm leading-relaxed font-medium mb-2">
                {effectiveCoreSummary}
              </p>
              <div className="h-0.5 bg-gradient-to-r from-purple-500/30 to-indigo-500/30 w-full rounded-full"></div>
            </div>
          ) : (
            <>
              <p className="text-gray-600 text-sm leading-relaxed">
                {argument.supporting_arguments?.[0] 
                  ? `${argument.supporting_arguments[0].substring(0, 120)}${argument.supporting_arguments[0].length > 120 ? '...' : ''}`
                  : 'Click to explore evidence and arguments for this perspective'
                }
              </p>
              <div className="h-0.5 bg-gradient-to-r from-purple-500/20 to-indigo-500/20 w-full mt-4 rounded-full"></div>
            </>
          )}
        </div>

        {/* Key Perspectives Tags */}
        {argument.key_perspectives && argument.key_perspectives.length > 0 && (
          <div className="mb-4">
            <div className="flex items-center gap-1 mb-2">
              <span className="text-xs text-gray-500 uppercase tracking-wide font-medium">Key Perspectives</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {argument.key_perspectives.slice(0, 3).map((perspective, index) => {
                const getIcon = (perspective: string) => {
                  if (perspective.includes('Mainstream')) return '🌐';
                  if (perspective.includes('Critical')) return '🔥';
                  if (perspective.includes('Technical')) return '🔬';
                  if (perspective.includes('Economic')) return '💼';
                  if (perspective.includes('Emerging')) return '🚀';
                  return '💡';
                };
                
                return (
                  <motion.span
                    key={index}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.2 + index * 0.1 }}
                    className="inline-flex items-center gap-1 px-2 py-1 bg-gradient-to-r from-purple-50 to-indigo-50 text-purple-700 rounded-full text-xs font-medium border border-purple-100"
                  >
                    <span className="text-xs">{getIcon(perspective)}</span>
                    <span className="truncate max-w-20">{perspective.split(' ')[0]}</span>
                  </motion.span>
                );
              })}
            </div>
          </div>
        )}

        {/* Learn More Button */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onLearnMore(argument)}
          className="w-full mt-auto flex items-center justify-center gap-2 text-purple-600 bg-purple-50 hover:bg-purple-100 p-3 rounded-xl transition-all duration-300 font-medium shadow-sm hover:shadow group"
        >
          <span>Learn more</span>
          <ChevronDownIcon className="h-5 w-5 transform group-hover:translate-y-0.5 transition-transform duration-300" />
        </motion.button>
      </Card>
    </motion.div>
  );
}; 