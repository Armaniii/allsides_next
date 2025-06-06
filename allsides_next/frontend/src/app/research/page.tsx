'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import researchService, { ResearchReport } from '@/services/researchService';

const ResearchPage: React.FC = () => {
  const [reports, setReports] = useState<ResearchReport[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  // Use useCallback to memoize the loadReports function
  const loadReports = useCallback(async () => {
    try {
      setLoading(true);
      const data = await researchService.getResearchReports();
      setReports(data);
    } catch (err) {
      console.error('Error loading research reports:', err);
      setError('Failed to load research reports. Please try again later.');
    } finally {
      setLoading(false);
    }
  }, []);

  // Set up polling for reports in PLANNING or RESEARCHING state
  useEffect(() => {
    const pollInterval = 5000; // Poll every 5 seconds
    let pollTimer: NodeJS.Timeout;

    const pollForUpdates = async () => {
      const needsUpdate = reports.some(report => 
        // Only poll for reports that aren't being streamed via SSE
        (report.status === 'PLANNING' || 
         report.status === 'RESEARCHING' ||
         report.status === 'WRITING') &&
        !report.isStreaming  // Add this flag when SSE connection is active
      );

      if (needsUpdate) {
        await loadReports();
        pollTimer = setTimeout(pollForUpdates, pollInterval);
      }
    };

    // Start polling if needed
    if (reports.some(report => 
      (report.status === 'PLANNING' || 
       report.status === 'RESEARCHING' ||
       report.status === 'WRITING') &&
      !report.isStreaming
    )) {
      pollTimer = setTimeout(pollForUpdates, pollInterval);
    }

    // Clean up on unmount
    return () => {
      if (pollTimer) {
        clearTimeout(pollTimer);
      }
    };
  }, [reports, loadReports]);

  // Initial load
  useEffect(() => {
    loadReports();
  }, [loadReports]);

  const handleStartNewResearch = () => {
    router.push('/research/new');
  };

  const handleViewReport = (id: number) => {
    router.push(`/research/${id}`);
  };

  const handleDeleteReport = async (id: number) => {
    if (window.confirm('Are you sure you want to delete this research report?')) {
      try {
        const success = await researchService.deleteResearchReport(id);
        if (success) {
          setReports(prevReports => prevReports.filter(report => report.id !== id));
        } else {
          setError('Failed to delete the report. Please try again.');
        }
      } catch (err) {
        console.error('Error deleting research report:', err);
        setError('Failed to delete the report. Please try again.');
      }
    }
  };

  // Helper function to get status badge styling
  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case 'PLANNING':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300';
      case 'RESEARCHING':
        return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300';
      case 'WRITING':
        return 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300';
      case 'COMPLETED':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300';
      case 'FAILED':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300';
    }
  };

  // Format date with memoization
  const formatDate = useCallback((dateString: string | null) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  }, []);

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold dark:text-white">Research Reports</h1>
        <button
          onClick={handleStartNewResearch}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition"
        >
          Start New Research
        </button>
      </div>

      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
          <p>{error}</p>
        </div>
      )}

      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      ) : reports.length > 0 ? (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {reports.map((report) => (
            <div 
              key={`${report.id}-${report.status}`} // Add status to key to force re-render on status change
              className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden"
            >
              <div className="p-5">
                <div className="flex justify-between items-start mb-3">
                  <h2 className="text-xl font-semibold dark:text-white">{report.topic}</h2>
                  <span 
                    className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBadgeClass(report.status)}`}
                  >
                    {report.status_display}
                  </span>
                </div>
                
                <div className="text-sm mb-4 text-gray-600 dark:text-gray-400">
                  <p>Created: {formatDate(report.created_at)}</p>
                  {report.completed_at && <p>Completed: {formatDate(report.completed_at)}</p>}
                </div>
                
                <div className="flex space-x-2 mt-4">
                  <button
                    onClick={() => handleViewReport(report.id)}
                    className="flex-1 px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition"
                  >
                    View
                  </button>
                  <button
                    onClick={() => handleDeleteReport(report.id)}
                    className="px-3 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 text-center">
          <p className="text-lg text-gray-600 dark:text-gray-400 mb-4">
            You haven&apos;t created any research reports yet.
          </p>
          <button
            onClick={handleStartNewResearch}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition"
          >
            Start Your First Research
          </button>
        </div>
      )}
    </div>
  );
};

export default ResearchPage; 