import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { XMarkIcon, TrophyIcon } from '@heroicons/react/24/outline';

interface LeaderboardUser {
  first_name: string;
  allstars: number;
}

interface LeaderboardModalProps {
  isOpen: boolean;
  onClose: () => void;
  users: LeaderboardUser[];
}

export const LeaderboardModal: React.FC<LeaderboardModalProps> = ({
  isOpen,
  onClose,
  users
}) => {
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
            className="fixed inset-0 bg-black/50 backdrop-blur-sm cursor-pointer z-50"
          />
          
          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 350 }}
            className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none p-4"
          >
            <div 
              className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-2xl w-full max-w-md pointer-events-auto overflow-hidden border border-purple-100"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="p-6 border-b border-purple-100">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-3">
                    <TrophyIcon className="h-6 w-6 text-yellow-500" />
                    <h3 className="text-xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
                      AllStars Leaderboard
                    </h3>
                  </div>
                  <button
                    onClick={onClose}
                    className="text-gray-400 hover:text-gray-500 transition-colors p-2 hover:bg-gray-100 rounded-lg"
                  >
                    <XMarkIcon className="h-6 w-6" />
                  </button>
                </div>
              </div>
              
              {/* Content */}
              <div className="p-6">
                <div className="space-y-4">
                  {users.map((user, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className={`flex items-center justify-between p-4 rounded-xl border ${
                        index === 0 
                          ? 'bg-gradient-to-r from-yellow-50 to-amber-50 border-yellow-200'
                          : index === 1
                          ? 'bg-gradient-to-r from-gray-50 to-slate-50 border-gray-200'
                          : index === 2
                          ? 'bg-gradient-to-r from-orange-50 to-amber-50 border-orange-200'
                          : 'bg-white border-purple-100'
                      }`}
                    >
                      <div className="flex items-center gap-4">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                          index === 0 
                            ? 'bg-yellow-100 text-yellow-700'
                            : index === 1
                            ? 'bg-gray-100 text-gray-700'
                            : index === 2
                            ? 'bg-orange-100 text-orange-700'
                            : 'bg-purple-100 text-purple-700'
                        }`}>
                          {index + 1}
                        </div>
                        <span className="font-medium text-gray-900">
                          {user.first_name}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-purple-600">
                          {user.allstars}
                        </span>
                        <TrophyIcon className={`h-5 w-5 ${
                          index === 0 
                            ? 'text-yellow-500'
                            : index === 1
                            ? 'text-gray-400'
                            : index === 2
                            ? 'text-orange-500'
                            : 'text-purple-400'
                        }`} />
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}; 