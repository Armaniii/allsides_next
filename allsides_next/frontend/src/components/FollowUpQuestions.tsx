import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MagnifyingGlassIcon, LightBulbIcon } from '@heroicons/react/24/outline';

interface FollowUpQuestionsProps {
  questions: string[];
  onQuestionClick: (question: string) => void;
  isVisible: boolean;
}

export const FollowUpQuestions: React.FC<FollowUpQuestionsProps> = ({
  questions,
  onQuestionClick,
  isVisible
}) => {
  if (!questions || questions.length === 0 || !isVisible) {
    return null;
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ delay: 0.5 }}
        className="w-full max-w-4xl mx-auto mt-12 mb-8"
      >
        {/* Header */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
          className="text-center mb-6"
        >
          <div className="flex items-center justify-center gap-2 mb-2">
            <LightBulbIcon className="h-5 w-5 text-purple-600" />
            <h3 className="text-lg font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
              Explore Further
            </h3>
          </div>
          <p className="text-sm text-gray-600">
            Deepen your understanding with these related questions
          </p>
        </motion.div>

        {/* Questions Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {questions.map((question, index) => (
            <motion.button
              key={index}
              initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.8 + index * 0.1 }}
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onQuestionClick(question)}
              className="group relative p-4 bg-white/80 backdrop-blur-sm rounded-xl border border-purple-100/50 hover:border-purple-200 transition-all duration-300 text-left shadow-sm hover:shadow-md"
            >
              {/* Subtle gradient background */}
              <div className="absolute inset-0 bg-gradient-to-br from-purple-50/30 to-indigo-50/30 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              
              {/* Content */}
              <div className="relative flex items-start gap-3">
                <div className="flex-shrink-0 mt-1">
                  <div className="w-6 h-6 rounded-full bg-gradient-to-r from-purple-100 to-indigo-100 flex items-center justify-center group-hover:from-purple-200 group-hover:to-indigo-200 transition-all duration-300">
                    <MagnifyingGlassIcon className="h-3.5 w-3.5 text-purple-600" />
                  </div>
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-700 leading-relaxed group-hover:text-gray-900 transition-colors duration-300">
                    {question}
                  </p>
                </div>
              </div>

              {/* Hover indicator */}
              <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <div className="w-2 h-2 rounded-full bg-gradient-to-r from-purple-500 to-indigo-500" />
              </div>
            </motion.button>
          ))}
        </div>

        {/* Footer hint */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="text-center mt-6"
        >
          <p className="text-xs text-gray-400">
            Click any question to explore that perspective
          </p>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};