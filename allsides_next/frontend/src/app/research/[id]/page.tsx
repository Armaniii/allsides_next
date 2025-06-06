'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import researchService, { ResearchReport, ResearchEventHandlers } from '@/services/researchService';
import ResearchMarkdownViewer from '@/components/ResearchMarkdownViewer';
import { toast } from 'react-hot-toast';
import { ClipboardIcon, DocumentArrowDownIcon } from '@heroicons/react/24/outline';

interface ResearchDetailPageProps {
  params: {
    id: string;
  };
}

const ResearchDetailPage: React.FC<ResearchDetailPageProps> = ({ params }) => {
  const reportId = parseInt(params.id);
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [planView, setPlanView] = useState<boolean>(false);
  const [feedback, setFeedback] = useState<string>('');
  const [submittingFeedback, setSubmittingFeedback] = useState<boolean>(false);
  const [approvingPlan, setApprovingPlan] = useState<boolean>(false);
  const [progressEvents, setProgressEvents] = useState<any[]>([]);
  const eventSourceRef = useRef<(() => void) | null>(null);
  const router = useRouter();
  const [isSendingFeedback, setIsSendingFeedback] = useState(false);

  // Load report data
  useEffect(() => {
    loadReport();
    // Clean up event source on unmount
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current();
        eventSourceRef.current = null;
      }
    };
  }, [reportId]);

  // Add effect to respond to report status changes
  useEffect(() => {
    if (report) {
      console.log('[DEBUG:COMPONENT] Report status changed:', report.status);
      
      // If the report is completed, ensure we close any existing event streams
      if (report.status === 'COMPLETED' && eventSourceRef.current) {
        console.log('[DEBUG:COMPONENT] Report status changed to COMPLETED, cleaning up any event streams');
        eventSourceRef.current();
        eventSourceRef.current = null;
      }
      
      // Optionally force a re-render
      const forceUpdate = setTimeout(() => {
        setProgressEvents(prev => [...prev]);
      }, 100);
      
      return () => clearTimeout(forceUpdate);
    }
  }, [report?.status]);

  const loadReport = async () => {
    setLoading(true);
    try {
      const report = await researchService.getResearchReport(reportId);
      console.log("[DEBUG:COMPONENT] Loaded report:", report);
      
      // Ensure report always has a plan
      if (!report.plan || (Array.isArray(report.plan) && report.plan.length === 0)) {
        console.log("[DEBUG:COMPONENT] Report has no plan, adding default section");
        report.plan = [{
          name: "Introduction",
          description: "Brief overview of the topic",
          research: false
        }];
      }
      
      setReport(report);
      setLoading(false);
      
      // Only set up event stream if the report is not already completed
      if (report.status !== 'COMPLETED') {
        setupEventStream(report);
      } else {
        console.log('[DEBUG:COMPONENT] Report already completed, not setting up stream');
      }
    } catch (error) {
      console.error('Error loading report:', error);
      setError('Failed to load research report.');
      setLoading(false);
    }
  };

  const setupEventStream = (report: ResearchReport) => {
    // Skip stream setup if report is already completed
    if (report.status === 'COMPLETED') {
      console.log('[DEBUG:COMPONENT] Report already completed, skipping stream setup');
      return;
    }

    // Clean up any existing event source before setting up a new one
    if (eventSourceRef.current) {
      console.log('[DEBUG:COMPONENT] Cleaning up existing event source before setting up new one');
      eventSourceRef.current();
      eventSourceRef.current = null;
    }

    // Define event handlers for SSE
    console.log('[DEBUG:COMPONENT] Configuring event handlers for report:', report.id);
    
    // Set streaming flag to prevent polling conflicts
    setReport(prev => prev ? {...prev, isStreaming: true} : null);
    
    const handlers: ResearchEventHandlers = {
      onPlan: (plan) => {
        console.log('[DEBUG:COMPONENT] onPlan handler called with plan:', plan);
        try {
          // Process the plan regardless of format
          // The plan could be either a direct array or contained within an object
          let processedPlan = plan;

          // Add debug logging to see what format we're receiving
          if (Array.isArray(plan)) {
            console.log('[DEBUG:COMPONENT] Plan is an array with length:', plan.length);
          } else if (typeof plan === 'object' && plan !== null) {
            console.log('[DEBUG:COMPONENT] Plan is an object with keys:', Object.keys(plan));
            
            // If it's an object with a sections property, use that
            if (Array.isArray(plan.sections)) {
              processedPlan = plan.sections;
              console.log('[DEBUG:COMPONENT] Using plan.sections array with length:', processedPlan.length);
            } else if (Object.keys(plan).length === 0) {
              // Empty object - create a default section to avoid frontend errors
              console.log('[DEBUG:COMPONENT] Received empty object plan, creating default section');
              processedPlan = [{
                name: 'Introduction',
                description: 'Brief overview of the topic',
                research: false
              }];
            }
          }

          // Now processedPlan should be an array we can use
          if (!Array.isArray(processedPlan)) {
            console.log('[DEBUG:COMPONENT] Plan is not an array, creating default section');
            processedPlan = [{
              name: 'Introduction',
              description: 'Brief overview of the topic',
              research: false
            }];
          }
          
          // Final check - ensure we never update state with an empty array
          if (Array.isArray(processedPlan) && processedPlan.length === 0) {
            console.log('[DEBUG:COMPONENT] Processed plan is empty array, creating default section');
            processedPlan = [{
              name: 'Introduction',
              description: 'Brief overview of the topic',
              research: false
            }];
          }
          
          console.log('[DEBUG:COMPONENT] Updating report with processed plan (length:', processedPlan.length, ')');
          
          // Update the report with the new plan and ensure we're in PLANNING state
          setReport(prev => {
            if (!prev) return null;
            
            // Don't change status if already researching or completed
            const newStatus = prev.status === 'RESEARCHING' || prev.status === 'COMPLETED' 
              ? prev.status
              : 'PLANNING';
              
            const updatedReport = {
              ...prev,
              plan: processedPlan,
              status: newStatus
            };
            
            console.log('[DEBUG:COMPONENT] Updated report with plan, now auto-starting research');
            
            // Automatically start research if we're in planning state and plan has auto_approve flag
            if (newStatus === 'PLANNING' && !approvingPlan && 
                (plan.auto_approved === true || plan.type === 'plan_ready')) {
              // Set a short timeout to allow the UI to update before starting research
              setTimeout(() => {
                console.log('[DEBUG:COMPONENT] Auto-approving plan');
                handleApprovePlan();
              }, 500);
            }
            
            return updatedReport;
          });
        } catch (error) {
          console.error('[DEBUG:COMPONENT] Error in onPlan handler:', error);
        }
      },
      onProgress: (event) => {
        console.log('[DEBUG:COMPONENT] onProgress handler called with event type:', event.type);
        try {
          // Protect against plan regression - don't revert status to planning
          // if we've already moved to researching or completion
          if (event.type === 'report_status' && report.status === 'RESEARCHING' && event.status === 'PLANNING') {
            console.log('[DEBUG:COMPONENT] Ignoring status regression from RESEARCHING to PLANNING');
            event.status = 'RESEARCHING';
          }
          
          // Similarly, don't revert from COMPLETED to an earlier state
          if (event.type === 'report_status' && report.status === 'COMPLETED' && 
             (event.status === 'PLANNING' || event.status === 'RESEARCHING')) {
            console.log('[DEBUG:COMPONENT] Ignoring status regression from COMPLETED');
            event.status = 'COMPLETED';
          }
          
          // Add the event to progress updates
          setProgressEvents(prev => [...prev, event]);
          
          // Handle plan_ready events (direct handling of Interrupt events from backend)
          if (event.type === 'plan_ready' && event.sections) {
            console.log('[DEBUG:COMPONENT] Handling plan_ready event with sections');
            // Call the onPlan handler to process the sections
            handlers.onPlan?.(event);
            return;
          }
          
          // Handle different event types
          if (event.type === 'report_status') {
            console.log('[DEBUG:COMPONENT] Processing report_status event with status:', event.status);
            
            // Check if this is a true update that we should process
            const shouldProcess = event.force_update === true || 
                                report.status !== event.status ||
                                event.plan ||
                                event.content;
                                
            if (!shouldProcess) {
              console.log('[DEBUG:COMPONENT] Skipping redundant report_status update');
              return;
            }
            
            console.log('[DEBUG:COMPONENT] Current report state before update:', report);
            
            setReport(prev => {
              if (!prev) return null;
              
              // Create the updated report object
              const updatedReport = {
                ...prev,
                status: event.status || prev.status
              };
              
              // If the event includes a plan, update that too
              if (event.plan) {
                console.log('[DEBUG:COMPONENT] Updating plan from report_status event');
                updatedReport.plan = event.plan;
              }
              
              // If the event includes content and status is COMPLETED, update that too
              if (event.content && event.status === 'COMPLETED') {
                console.log('[DEBUG:COMPONENT] Updating content from report_status completion event');
                updatedReport.content = event.content;
                updatedReport.completed_at = new Date().toISOString();
              }
              
              console.log('[DEBUG:COMPONENT] Updated report state:', updatedReport);
              return updatedReport;
            });
          }
          // Handle research_started event type
          else if (event.type === 'research_started' || 
                   event.type === 'plan_ready' || 
                   (event.type === 'report_status' && event.status === 'PLANNING' && !approvingPlan)) {
            console.log('[DEBUG:COMPONENT] Processing research start/plan ready/planning status, moving to research phase');
            
            // Auto approve plan if we're not already in the process and auto_approved flag is set
            if (event.status === 'PLANNING' && !approvingPlan && 
                (event.auto_approved === true || event.type === 'plan_ready')) {
              // Short timeout to allow UI to update
              setTimeout(() => {
                console.log('[DEBUG:COMPONENT] Auto-approving plan from event:', event.type);
                handleApprovePlan();
              }, 500);
            }
            
            setReport(prev => {
              if (!prev) return null;
              
              // Set status to RESEARCHING regardless of current state
              const updatedReport = {
                ...prev,
                status: 'RESEARCHING'
              };
              
              // Update plan if provided
              if (event.sections) {
                updatedReport.plan = event.sections;
              }
              
              console.log('[DEBUG:COMPONENT] Updated report for research start:', updatedReport);
              return updatedReport;
            });
          }
          // Handle researching_section events
          else if (event.type === 'researching_section') {
            console.log('[DEBUG:COMPONENT] Processing researching_section event:', event.section_name);
            
            // Update the report status if needed
            setReport(prev => {
              if (!prev || prev.status === 'COMPLETED') return prev;
              
              return {
                ...prev,
                status: 'RESEARCHING'
              };
            });
          }
          // Handle section_completed events
          else if (event.type === 'section_completed') {
            console.log('[DEBUG:COMPONENT] Processing section completion');
            // We don't need to update the report state here, just add to progress events
          }
          // Handle report_completed events
          else if (event.type === 'report_completed' && event.content) {
            console.log('[DEBUG:COMPONENT] Processing report completion');
            // Update the report with completed status and content
            setReport(prev => {
              if (!prev) return null;
              
              // Set to COMPLETED state
              return {
                ...prev,
                status: 'COMPLETED',
                content: event.content,
                completed_at: new Date().toISOString()
              };
            });
          }
          // Handle build_section_with_web_research events
          else if (event.build_section_with_web_research) {
            console.log('[DEBUG:COMPONENT] Processing build_section_with_web_research event');
            
            // Update report status
            setReport(prev => {
              if (!prev || prev.status === 'COMPLETED') return prev;
              
              return {
                ...prev,
                status: 'RESEARCHING'
              };
            });
          }
          // Handle direct section data from the backend
          else if (event.completed_sections) {
            console.log('[DEBUG:COMPONENT] Processing completed_sections event');
            // No need to update status, just add to progress events
          }
          // Handle final report directly from backend
          else if (event.compile_final_report && event.compile_final_report.final_report) {
            console.log('[DEBUG:COMPONENT] Processing final report content');
            setReport(prev => {
              if (!prev) return null;
              
              return {
                ...prev,
                status: 'COMPLETED',
                content: event.compile_final_report.final_report,
                completed_at: new Date().toISOString()
              };
            });
          }
        } catch (error) {
          console.error('[DEBUG:COMPONENT] Error in onProgress handler:', error);
        }
      },
      onComplete: (reportContent: any) => {
        console.log('[DEBUG:COMPONENT] onComplete handler called with data:', reportContent);
        try {
          // Extract content string from the response regardless of format
          let contentString = '';
          
          if (typeof reportContent === 'string') {
            // If it's already a string, use it directly
            contentString = reportContent;
          } else if (reportContent && typeof reportContent === 'object') {
            // If it's an object with content property
            if (typeof reportContent.content === 'string') {
              contentString = reportContent.content;
            } else if (reportContent.content) {
              // If content exists but is not a string, stringify it
              contentString = JSON.stringify(reportContent.content);
            } else {
              // If no content property, stringify the whole object
              contentString = JSON.stringify(reportContent);
            }
          } else {
            // Fallback - convert whatever we got to a string
            contentString = String(reportContent || '');
          }
          
          // Only update if we have actual content
          if (!contentString.trim()) {
            console.log('[DEBUG:COMPONENT] Ignoring empty content in onComplete');
            return;
          }
          
          // Clean up the event source to prevent duplicate streams
          if (eventSourceRef.current) {
            console.log('[DEBUG:COMPONENT] Cleaning up event source on completion');
            eventSourceRef.current();
            eventSourceRef.current = null;
          }
          
          // Update the report with the completed content
          setReport(prev => {
            if (!prev) return null;
            
            console.log('[DEBUG:COMPONENT] Updating report with completed content string');
            return {
              ...prev,
              content: contentString,
              status: 'COMPLETED',
              completed_at: new Date().toISOString(),
              isStreaming: false // Mark as not streaming anymore
            };
          });
          
          setApprovingPlan(false);
        } catch (error) {
          console.error('[DEBUG:COMPONENT] Error in onComplete handler:', error);
          setApprovingPlan(false);
        }
      },
      onError: (err) => {
        console.error('[DEBUG:COMPONENT] Error in research stream:', err);
        try {
          // Extract a useful error message regardless of error type
          let errorMessage = 'An unknown error occurred';
          
          // Handle different error types
          if (err instanceof Error) {
            // Standard Error object
            errorMessage = err.message;
          } else if (err instanceof Event) {
            // EventSource error
            errorMessage = 'Connection error with the research stream';
          } else if (typeof err === 'string') {
            // String error
            errorMessage = err;
          } else if (err && typeof err === 'object') {
            // Object with message or description
            errorMessage = err.message || err.description || err.error || JSON.stringify(err);
          }
          
          // Log the error message to the console
          console.error('[DEBUG:COMPONENT] Error message:', errorMessage);
          
          // Don't change report status on connection errors which might resolve themselves
          if (errorMessage.toLowerCase().includes('connection') || 
              errorMessage.toLowerCase().includes('timeout') || 
              errorMessage.toLowerCase().includes('reconnect')) {
            
            // Just show an error toast without changing report state
            toast.error(`Connection issue: ${errorMessage}. Trying to reconnect...`, {
              duration: 5000,
              id: 'connection-error' // Prevent multiple identical toasts
            });
            
            // Try to reconnect by setting up the stream again after a short delay
            setTimeout(() => {
              if (report) {
                console.log('[DEBUG:COMPONENT] Attempting to reconnect SSE stream');
                setupEventStream(report);
              }
            }, 3000);
          } else {
            // For major errors, update the error state and potentially the report status
            setError(`Research error: ${errorMessage}`);
            
            // Only set to FAILED if we're not already COMPLETED
            setReport(prev => {
              if (!prev || prev.status === 'COMPLETED') return prev;
              
              return {
                ...prev,
                status: 'FAILED'
              };
            });
            
            // Show a more detailed toast
            toast.error(`Research process encountered an error: ${errorMessage}`, {
              duration: 5000,
              id: 'research-error'
            });
          }
        } catch (error) {
          console.error('[DEBUG:COMPONENT] Error in onError handler itself:', error);
          setError('An error occurred in the research process');
        }
      }
    };
    
    console.log('[DEBUG:COMPONENT] Starting SSE connection to report:', report.id);
    
    // Set up the event stream
    const cleanup = researchService.streamResearchReport(
      reportId,
      (event) => {
        if (handlers.onProgress) {
          handlers.onProgress(event);
        }
      },
      (report) => {
        if (handlers.onComplete) {
          handlers.onComplete(report);
        }
      },
      (error) => {
        if (handlers.onError) {
          handlers.onError(error);
        }
      }
    );
    console.log('[DEBUG:COMPONENT] SSE connection established, cleanup function received');
    eventSourceRef.current = cleanup;
  };

  const handleProvideFeedback = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!feedback.trim()) {
      setError('Please provide feedback');
      return;
    }
    
    try {
      setSubmittingFeedback(true);
      setError(null);
      
      // Clean up existing event stream
      if (eventSourceRef.current) {
        eventSourceRef.current();
        eventSourceRef.current = null;
      }
      
      // Define new event handlers for feedback
      const handlers: ResearchEventHandlers = {
        onPlan: (plan) => {
          // Update the report with the new plan
          setReport(prev => prev ? {
            ...prev,
            plan: plan,
            status: 'PLANNING'
          } : null);
          setSubmittingFeedback(false);
        },
        onProgress: (event) => {
          // Add the event to progress updates
          setProgressEvents(prev => [...prev, event]);
        },
        onError: (err) => {
          console.error('Error providing feedback:', err);
          setError('Failed to process feedback. Please try again.');
          setSubmittingFeedback(false);
        }
      };
      
      // Send feedback and set up new event stream
      const cleanup = researchService.provideFeedback(reportId, { feedback }, handlers);
      eventSourceRef.current = cleanup;
    } catch (err) {
      console.error('Error providing feedback:', err);
      setError('Failed to submit feedback. Please try again.');
      setSubmittingFeedback(false);
    }
  };

  const handleApprovePlan = async () => {
    try {
      setApprovingPlan(true);
      
      // Always make sure we have a plan
      if (!report?.plan || (Array.isArray(report.plan) && report.plan.length === 0)) {
        console.error('Cannot approve - no plan available');
        toast.error('Cannot approve - plan is not available.');
        setApprovingPlan(false);
        return;
      }
      
      // Start the approval process
      console.log('[DEBUG:COMPONENT] Approving plan...');
      await researchService.approveResearchPlan(report.id, {
        onProgress: (event) => {
          // Add the event to progress updates
          setProgressEvents(prev => [...prev, event]);
          
          // Handle report_status events to update the report status
          if (event.type === 'report_status') {
            console.log('[DEBUG:COMPONENT] Processing report_status event with status:', event.status);
            setReport(prev => {
              if (!prev) return null;
              
              // Create the updated report object
              const updatedReport = {
                ...prev,
                status: event.status || prev.status
              };
              
              // If the event includes a plan, update that too
              if (event.plan) {
                updatedReport.plan = event.plan;
              }
              
              // If the event includes content and status is COMPLETED, update that too
              if (event.content && event.status === 'COMPLETED') {
                updatedReport.content = event.content;
                updatedReport.completed_at = new Date().toISOString();
              }
              
              return updatedReport;
            });
          }
          
          // Also check for section building info
          if (event.build_section_with_web_research) {
            setReport(prev => prev ? {
              ...prev,
              status: 'RESEARCHING'
            } : null);
          }
        },
        onComplete: (reportContent) => {
          // Transform the backend response to match our frontend interface
          const transformedContent = typeof reportContent === 'string' 
            ? reportContent 
            : typeof reportContent === 'object' && reportContent !== null
              ? JSON.stringify(reportContent)
              : String(reportContent);

          // Update the report with the completed content
          setReport(prev => {
            if (!prev) return null;
            
            const updatedReport: ResearchReport = {
              ...prev,
              content: transformedContent,
              status: 'COMPLETED',
              completed_at: new Date().toISOString()
            };
            return updatedReport;
          });
          setApprovingPlan(false);
        },
        onError: (err) => {
          console.error('Error in approval stream:', err);
          toast.error('Error during research: ' + err);
          setApprovingPlan(false);
        }
      });
    } catch (error) {
      console.error('Error approving plan:', error);
      toast.error('Failed to approve research plan. Please try again.');
      setApprovingPlan(false);
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  const renderProgressIndicator = () => {
    const steps = [
      { status: 'PLANNING', label: 'Planning', description: 'Creating research plan' },
      { status: 'RESEARCHING', label: 'Researching', description: 'Gathering information' },
      { status: 'COMPLETED', label: 'Completed', description: 'Report finished' },
    ];
    
    const currentStatusIndex = steps.findIndex(step => step.status === report?.status);
    
    return (
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {steps.map((step, index) => (
            <React.Fragment key={step.status}>
              {/* Step circle */}
              <div className="flex flex-col items-center">
                <div 
                  className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    index <= currentStatusIndex 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-400 dark:bg-gray-700'
                  }`}
                >
                  {index < currentStatusIndex ? (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <span>{index + 1}</span>
                  )}
                </div>
                <span className="mt-2 text-sm text-gray-600 dark:text-gray-400">{step.label}</span>
                <span className="text-xs text-gray-500 dark:text-gray-500">{step.description}</span>
              </div>
              
              {/* Connector line between steps */}
              {index < steps.length - 1 && (
                <div 
                  className={`flex-1 h-1 mx-2 ${
                    index < currentStatusIndex 
                      ? 'bg-blue-600' 
                      : 'bg-gray-200 dark:bg-gray-700'
                  }`}
                />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    );
  };

  const renderPlanningView = () => {
    if (!report?.plan) return null;
    
    // Extract plan sections
    const sections = Array.isArray(report.plan) ? report.plan : [];
    
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-xl font-bold mb-4 dark:text-white">Research Plan</h2>
        
        {sections.length > 0 ? (
          <div>
            <ul className="mb-6">
              {sections.map((section, index) => (
                <li key={index} className="mb-4 pb-4 border-b border-gray-200 dark:border-gray-700 last:border-b-0">
                  <h3 className="font-semibold text-lg mb-1 dark:text-white">{section.name}</h3>
                  <p className="text-gray-700 dark:text-gray-300 mb-2">{section.description}</p>
                  <div className="flex items-center">
                    <span 
                      className={`px-2 py-1 rounded-full text-xs font-medium ${
                        section.research
                          ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300'
                          : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'
                      }`}
                    >
                      {section.research ? 'Research needed' : 'No research needed'}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
            
            <form onSubmit={handleProvideFeedback} className="mb-4">
              <div className="mb-4">
                <label 
                  htmlFor="feedback" 
                  className="block text-gray-700 dark:text-gray-300 text-sm font-medium mb-2"
                >
                  Provide Feedback (Optional)
                </label>
                <textarea
                  id="feedback"
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                  placeholder="Suggest changes to the research plan..."
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  disabled={submittingFeedback || approvingPlan}
                />
              </div>
              
              <div className="flex space-x-3">
                {feedback.trim() && (
                  <button
                    type="submit"
                    disabled={submittingFeedback || approvingPlan}
                    className="px-4 py-2 bg-blue-600 rounded-md shadow-sm text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {submittingFeedback ? 'Submitting...' : 'Submit Feedback'}
                  </button>
                )}
                
                <button
                  type="button"
                  onClick={handleApprovePlan}
                  disabled={submittingFeedback || approvingPlan}
                  className="px-4 py-2 bg-green-600 rounded-md shadow-sm text-sm font-medium text-white hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {approvingPlan ? 'Approving...' : 'Approve Plan'}
                </button>
              </div>
            </form>
          </div>
        ) : (
          <p className="text-gray-600 dark:text-gray-400">
            Waiting for the research plan...
          </p>
        )}
      </div>
    );
  };

  const renderResearchingView = () => {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-xl font-bold mb-4 dark:text-white">Research in Progress</h2>
        
        <div className="mb-4">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mr-3"></div>
            <span className="text-gray-700 dark:text-gray-300">Researching your topic...</span>
          </div>
        </div>
        
        {progressEvents.length > 0 && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-2 dark:text-white">Progress Updates</h3>
            <div className="overflow-y-auto max-h-60 p-3 bg-gray-50 dark:bg-gray-700 rounded-md text-sm">
              {progressEvents.map((event, index) => {
                // Log the event for debugging
                if (index === progressEvents.length - 1) {
                  console.log('[DEBUG:COMPONENT] Latest progress event:', event);
                }
                
                // Handle different event types
                if (event.type === 'researching_section' || 
                   (event.build_section_with_web_research && event.build_section_with_web_research.section)) {
                  // Extract section name from either format
                  const sectionName = event.section_name || 
                                     (event.build_section_with_web_research?.section?.name) || 
                                     (typeof event.build_section_with_web_research?.section === 'string' 
                                      ? event.build_section_with_web_research.section 
                                      : null) ||
                                     'Unknown section';
                  
                  return (
                    <div key={index} className="mb-2 pb-2 border-b dark:border-gray-600 last:border-b-0">
                      <p className="text-gray-700 dark:text-gray-300">
                        <span className="text-blue-600 dark:text-blue-400">Researching:</span> {sectionName}
                      </p>
                    </div>
                  );
                }
                
                if (event.type === 'section_completed' || event.completed_sections) {
                  // Get the section data from either format
                  const sections = event.sections || event.completed_sections || [];
                  
                  if (sections.length === 0) return null;
                  
                  return (
                    <div key={index} className="mb-2 pb-2 border-b dark:border-gray-600 last:border-b-0">
                      <p className="text-green-600 dark:text-green-400">
                        Completed section{sections.length > 1 ? 's' : ''}: {sections.map((s: any) => {
                          // Extract name from different possible formats
                          if (typeof s === 'string') return s;
                          if (s.name) return s.name;
                          if (typeof s === 'object') {
                            const name = s.name || s.title || s.heading || JSON.stringify(s);
                            return name;
                          }
                          return 'Unknown section';
                        }).join(', ')}
                      </p>
                    </div>
                  );
                }
                
                if (event.type === 'report_completed' || 
                   (event.compile_final_report && event.compile_final_report.final_report)) {
                  return (
                    <div key={index} className="mb-2 pb-2 border-b dark:border-gray-600 last:border-b-0">
                      <p className="text-green-600 dark:text-green-400 font-semibold">
                        Report completed!
                      </p>
                    </div>
                  );
                }
                
                // Display message if available for any other event
                if (event.message) {
                  return (
                    <div key={index} className="mb-2 pb-2 border-b dark:border-gray-600 last:border-b-0">
                      <p className="text-gray-700 dark:text-gray-300">
                        {event.message}
                      </p>
                    </div>
                  );
                }
                
                // Default display for unhandled event types - show the event type
                return (
                  <div key={index} className="mb-2 pb-2 border-b dark:border-gray-600 last:border-b-0">
                    <p className="text-gray-700 dark:text-gray-300">
                      {event.type || Object.keys(event)[0] || 'Event update'}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderCompletedView = () => {
    if (!report?.content) return null;
    
    const handleExportMarkdown = () => {
      if (!report?.content) return;
      
      try {
        // Try the modern Blob API approach
        const blob = new Blob([report.content], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        
        // Create a link element and trigger download
        const link = document.createElement('a');
        link.href = url;
        link.download = `${report.topic.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_report.md`;
        document.body.appendChild(link);
        link.click();
        
        // Clean up
        setTimeout(() => {
          URL.revokeObjectURL(url);
          document.body.removeChild(link);
        }, 100);
        
        toast.success('Markdown file downloaded successfully');
      } catch (err) {
        console.error('Export markdown error:', err);
        
        // Fallback for older browsers
        try {
          // Create a data URI
          const encodedContent = encodeURIComponent(report.content);
          const dataUri = `data:text/markdown;charset=utf-8,${encodedContent}`;
          
          // Create link with data URI
          const link = document.createElement('a');
          link.href = dataUri;
          link.download = `${report.topic.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_report.md`;
          link.style.display = 'none';
          document.body.appendChild(link);
          link.click();
          
          // Clean up
          setTimeout(() => {
            document.body.removeChild(link);
          }, 100);
          
          toast.success('Markdown file downloaded successfully');
        } catch (fallbackErr) {
          console.error('Export markdown fallback error:', fallbackErr);
          toast.error('Failed to export markdown file. Try copying the text instead.');
        }
      }
    };
    
    const handleCopyText = () => {
      if (!report?.content) return;
      
      try {
        // Try the modern Clipboard API first
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(report.content)
            .then(() => {
              toast.success('Report text copied to clipboard');
            })
            .catch(err => {
              console.error('Clipboard API failed:', err);
              // Fall back to textarea method
              copyTextFallback();
            });
        } else {
          // Use fallback for browsers without Clipboard API
          copyTextFallback();
        }
      } catch (error) {
        console.error('Copy text error:', error);
        // Final fallback
        copyTextFallback();
      }
    };
    
    const copyTextFallback = () => {
      if (!report?.content) return;
      
      // Create a temporary textarea element
      const textarea = document.createElement('textarea');
      textarea.value = report.content;
      
      // Make the textarea out of viewport
      textarea.style.position = 'fixed';
      textarea.style.left = '-999999px';
      textarea.style.top = '-999999px';
      
      // Add, select, copy, and remove
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      
      let success = false;
      try {
        success = document.execCommand('copy');
      } catch (err) {
        console.error('execCommand error:', err);
      }
      
      document.body.removeChild(textarea);
      
      if (success) {
        toast.success('Report text copied to clipboard');
      } else {
        toast.error('Failed to copy text to clipboard');
      }
    };
    
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 overflow-hidden">
        {/* Export/Copy buttons */}
        <div className="flex justify-end mb-4 space-x-2">
          <button
            onClick={handleCopyText}
            className="inline-flex items-center px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            title="Copy report text to clipboard"
          >
            <ClipboardIcon className="h-4 w-4 mr-2" />
            Copy Text
          </button>
          <button
            onClick={handleExportMarkdown}
            className="inline-flex items-center px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            title="Export report as markdown file"
          >
            <DocumentArrowDownIcon className="h-4 w-4 mr-2" />
            Export as Markdown
          </button>
        </div>
        
        <ResearchMarkdownViewer content={report.content} />
      </div>
    );
  };

  const handleFeedbackChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setFeedback(e.target.value);
  };

  const handleSubmitFeedback = async () => {
    try {
      // Validate feedback
      if (!feedback.trim()) {
        toast.error('Please enter feedback before submitting.');
        return;
      }
      
      // Always make sure we have a plan
      if (!report?.plan || (Array.isArray(report.plan) && report.plan.length === 0)) {
        console.error('Cannot provide feedback - no plan available');
        toast.error('Cannot provide feedback - plan is not available.');
        return;
      }
      
      // Set loading state
      setIsSendingFeedback(true);
      
      // Submit feedback
      console.log('[DEBUG:COMPONENT] Submitting feedback:', feedback);
      await researchService.provideFeedback(report.id, { feedback }, {
        onProgress: (event) => {
          // Add the event to progress updates
          setProgressEvents(prev => [...prev, event]);
        },
        onPlan: (updatedPlan) => {
          // Update the report with the new plan
          setReport(prev => prev ? {
            ...prev,
            plan: updatedPlan,
            status: 'PLANNING'
          } : null);
        },
        onError: (err) => {
          console.error('Error in feedback stream:', err);
          toast.error('Error processing feedback: ' + err);
          setIsSendingFeedback(false);
        }
      });
      
      // Clear feedback field
      setFeedback('');
      setIsSendingFeedback(false);
      toast.success('Feedback submitted successfully.');
      
    } catch (error) {
      console.error('Error submitting feedback:', error);
      toast.error('Failed to submit feedback. Please try again.');
      setIsSendingFeedback(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      ) : error ? (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
          <p>{error}</p>
        </div>
      ) : !report ? (
        <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-6" role="alert">
          <p>Research report not found</p>
        </div>
      ) : (
        <>
          <div className="mb-6 flex flex-col md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2 dark:text-white">{report.topic}</h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Created: {formatDate(report.created_at)}
                {report.completed_at && ` • Completed: ${formatDate(report.completed_at)}`}
              </p>
            </div>
            
            <div className="mt-4 md:mt-0 flex items-center">
              <span 
                className={`px-3 py-1 rounded-full text-sm font-medium ${
                  report.status === 'COMPLETED' 
                    ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'
                    : report.status === 'FAILED'
                    ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
                    : 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300'
                }`}
              >
                {report.status_display || report.status}
              </span>
              
              <button
                onClick={() => router.push('/research')}
                className="ml-4 px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-md text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Back to Reports
              </button>
            </div>
          </div>
          
          {/* Progress indicator */}
          {renderProgressIndicator()}
          
          {/* View selector for completed reports */}
          {report.status === 'COMPLETED' && (
            <div className="mb-6 flex items-center justify-end">
              <div className="inline-flex rounded-md shadow-sm">
                <button
                  onClick={() => setPlanView(false)}
                  className={`px-4 py-2 text-sm font-medium rounded-l-md ${
                    !planView 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600'
                  }`}
                >
                  Report
                </button>
                <button
                  onClick={() => setPlanView(true)}
                  className={`px-4 py-2 text-sm font-medium rounded-r-md ${
                    planView 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600'
                  }`}
                >
                  Research Plan
                </button>
              </div>
            </div>
          )}
          
          {/* Content based on status */}
          {report.status === 'PLANNING' ? (
            <div key={`planning-${report.status}-${Date.now()}`}>
              {renderPlanningView()}
            </div>
          ) : report.status === 'RESEARCHING' || report.status === 'WRITING' ? (
            <div key={`researching-${report.status}-${Date.now()}`}>
              {renderResearchingView()}
            </div>
          ) : report.status === 'COMPLETED' ? (
            <div key={`completed-${report.status}-${Date.now()}`}>
              {planView ? renderPlanningView() : renderCompletedView()}
            </div>
          ) : (
            <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
              <p>Research failed to complete. Please try again with a different topic or configuration.</p>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ResearchDetailPage; 