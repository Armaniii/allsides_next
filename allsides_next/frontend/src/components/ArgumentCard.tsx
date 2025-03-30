import React from 'react';
import { motion } from 'framer-motion';
import { ChevronDownIcon } from '@heroicons/react/24/outline';
import { Card } from '@/components/ui/card';

interface Stance {
  stance: string;
  core_argument: string;
  supporting_arguments: string[];
}

interface ArgumentCardProps {
  argument: Stance;
  onLearnMore: () => void;
}

export const ArgumentCard: React.FC<ArgumentCardProps> = ({ argument, onLearnMore }) => {
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

        {/* Core Argument */}
        <div className="mb-6">
          <p className="text-gray-700 font-medium leading-relaxed">
            {argument.core_argument}
          </p>
          <div className="h-0.5 bg-gradient-to-r from-purple-500/20 to-indigo-500/20 w-full mt-4 rounded-full"></div>
        </div>

        {/* Learn More Button */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={onLearnMore}
          className="w-full mt-auto flex items-center justify-center gap-2 text-purple-600 bg-purple-50 hover:bg-purple-100 p-3 rounded-xl transition-all duration-300 font-medium shadow-sm hover:shadow group"
        >
          <span>Learn more</span>
          <ChevronDownIcon className="h-5 w-5 transform group-hover:translate-y-0.5 transition-transform duration-300" />
        </motion.button>
      </Card>
    </motion.div>
  );
}; 