import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Interface for starting a research session
 */
export interface StartResearchRequest {
  topic: string;
  config?: {
    search_api?: string;
    planner_provider?: string;
    planner_model?: string;
    writer_provider?: string;
    writer_model?: string;
    max_search_depth?: number;
    number_of_queries?: number;
    report_structure?: string;
  };
}

/**
 * Interface for research feedback
 */
export interface ResearchFeedbackRequest {
  feedback: string;
}

/**
 * Interface for a research report
 */
export interface ResearchReport {
  id: number;
  user: number;
  username: string;
  topic: string;
  thread_id: string;
  status: string;
  status_display: string;
  content: string;
  plan: any;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  sections: ResearchSection[];
  isStreaming?: boolean;
}

/**
 * Interface for a research section
 */
export interface ResearchSection {
  id: number;
  name: string;
  content: string;
  order: number;
  created_at: string;
  updated_at: string;
}

/**
 * Interface for SSE event handling
 */
export interface ResearchEventHandlers {
  onPlan?: (plan: any) => void;
  onProgress?: (event: any) => void;
  onComplete?: (report: ResearchReport) => void;
  onError?: (error: any) => void;
}

/**
 * Start a new research session
 * @param data The research request data
 * @returns The created research report details
 */
export const startResearch = async (data: StartResearchRequest): Promise<ResearchReport> => {
  const token = localStorage.getItem('accessToken');
  
  const response = await axios.post(`${API_URL}/api/research/reports/`, data, {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    }
  });
  
  return response.data as ResearchReport;
};

/**
 * Get all research reports for the current user
 * @returns List of research reports
 */
export const getResearchReports = async (): Promise<ResearchReport[]> => {
  const token = localStorage.getItem('accessToken');
  
  const response = await axios.get(`${API_URL}/api/research/reports/`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  return response.data as ResearchReport[];
};

/**
 * Get a specific research report by ID
 * @param id The research report ID
 * @returns The research report details
 */
export const getResearchReport = async (id: number): Promise<ResearchReport> => {
  const token = localStorage.getItem('accessToken');
  
  const response = await axios.get(`${API_URL}/api/research/reports/${id}/`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  return response.data as ResearchReport;
};

/**
 * Stream a research report via SSE
 * @param reportId The research report ID
 * @param onEvent Event handler for all types of events
 * @param onComplete Event handler for completion
 * @param onError Event handler for errors
 * @returns A function to close the SSE connection
 */
export const streamResearchReport = (
  reportId: number,
  onEvent: (event: any) => void,
  onComplete: (report: any) => void,
  onError: (error: any) => void
): (() => void) => {
  console.log(`[DEBUG:SERVICE] Setting up stream for report ${reportId}`);
  
  // Get the JWT token from localStorage
  const token = localStorage.getItem('accessToken');
  if (!token) {
    onError('Authentication required');
    return () => {}; // Return empty cleanup function
  }
  
  // Set up EventSource with JWT token
  const eventSource = new EventSource(`${API_URL}/api/research/reports/${reportId}/stream/?token=${token}&format=txt`);
  let reconnectAttempts = 0;
  const maxReconnectAttempts = 5;
  const reconnectDelay = 2000; // 2 seconds
  
  // Set up event handlers
  eventSource.onopen = () => {
    console.log(`[DEBUG:SERVICE] Stream connection opened for report ${reportId}`);
    reconnectAttempts = 0; // Reset reconnect attempts on successful connection
  };
  
  eventSource.onmessage = (event) => {
    try {
      if (!event.data) {
        console.warn('Received empty event data');
        return;
      }
      
      const data = JSON.parse(event.data);
      console.log('[DEBUG:SERVICE] Received SSE event:', data);
      
      // Handle interrupt events (they contain the plan)
      if (data.__interrupt__) {
        console.log('[DEBUG:SERVICE] Detected interrupt event with data', data.__interrupt__);
        
        // Extract plan data if available
        try {
          const interruptData = data.__interrupt__[0]?.value;
          if (interruptData) {
            let planData;
            
            // Handle string or object format
            if (typeof interruptData === 'string') {
              // Try to parse string as JSON
              try {
                planData = JSON.parse(interruptData);
              } catch {
                // Check if it's a string containing section info
                if (interruptData.includes('Section:')) {
                  // This is a plan in text format, create a simpler event for handling
                  const processedEvent = {
                    type: 'plan_ready',
                    auto_approved: true,
                    force_update: true,
                    timestamp: Date.now() / 1000,
                    message: 'Plan extracted from interrupt data',
                    plan_text: interruptData
                  };
                  onEvent(processedEvent);
                  return;
                }
              }
            } else if (typeof interruptData === 'object' && interruptData !== null) {
              planData = interruptData;
            }
            
            // If we have plan data and it contains sections
            if (planData && planData.sections) {
              // Create a processed event
              const processedEvent = {
                type: 'plan_ready',
                sections: planData.sections,
                auto_approved: true,
                force_update: true,
                timestamp: Date.now() / 1000
              };
              onEvent(processedEvent);
              return;
            }
          }
        } catch (err) {
          console.error('[DEBUG:SERVICE] Error processing interrupt data:', err);
        }
      }
      
      // Handle researching_section events
      if (data.type === 'researching_section' || data.build_section_with_web_research) {
        // Extract section data if available
        let sectionName = 'Unknown section';
        console.log('[DEBUG:SERVICE] Original section event data:', JSON.stringify(data).substring(0, 500));
        
        // Try multiple approaches to extract section name
        if (data.build_section_with_web_research?.section?.name) {
          sectionName = data.build_section_with_web_research.section.name;
          console.log('[DEBUG:SERVICE] Found section name in build_section_with_web_research.section.name:', sectionName);
        } else if (data.section_name) {
          sectionName = data.section_name;
          console.log('[DEBUG:SERVICE] Found section name in section_name:', sectionName);
        } else if (data.message && data.message.includes('Researching section:')) {
          sectionName = data.message.split('Researching section:')[1].trim();
          console.log('[DEBUG:SERVICE] Extracted section name from message:', sectionName);
        } else if (data.build_section_with_web_research) {
          // Sometimes the section data structure might be different
          const sectionData = data.build_section_with_web_research;
          if (typeof sectionData === 'object') {
            console.log('[DEBUG:SERVICE] Section data keys:', Object.keys(sectionData));
            
            // Try to extract section from different possible structures
            if (typeof sectionData.section === 'string') {
              sectionName = sectionData.section;
            } else if (typeof sectionData.section === 'object' && sectionData.section !== null) {
              if (sectionData.section.name) {
                sectionName = sectionData.section.name;
              } else {
                // Last resort - look for any property that might be a name
                const potentialNameProps = ['name', 'title', 'heading', 'id'];
                for (const prop of potentialNameProps) {
                  if (sectionData.section[prop]) {
                    sectionName = sectionData.section[prop];
                    break;
                  }
                }
              }
            }
            console.log('[DEBUG:SERVICE] Extracted section name from complex structure:', sectionName);
          }
        }
        
        // Create a normalized event
        const processedEvent = {
          type: 'researching_section',
          section_name: sectionName,
          force_update: true,
          timestamp: data.timestamp || (Date.now() / 1000),
          message: `Researching: ${sectionName}`
        };
        onEvent(processedEvent);
        return;
      }
      
      // Handle section_completed events
      if (data.type === 'section_completed' || data.completed_sections) {
        const processedEvent = {
          type: 'section_completed',
          sections: data.completed_sections || [],
          force_update: true,
          timestamp: data.timestamp || (Date.now() / 1000)
        };
        onEvent(processedEvent);
        return;
      }
      
      // Handle report_completed events
      if (data.type === 'report_completed' || data.compile_final_report) {
        let reportContent = '';
        
        if (data.compile_final_report?.final_report) {
          reportContent = data.compile_final_report.final_report;
        }
        
        const processedEvent = {
          type: 'report_completed',
          content: reportContent,
          status: 'COMPLETED',
          force_update: true,
          timestamp: data.timestamp || (Date.now() / 1000)
        };
        
        onEvent(processedEvent);
        
        // Also call onComplete with the final report
        if (reportContent) {
          onComplete({
            content: reportContent,
            status: 'COMPLETED',
            completed_at: new Date().toISOString()
          });
          
          // Close the EventSource when report is completed to prevent duplicates
          console.log('[DEBUG:SERVICE] Report completed, closing EventSource');
          eventSource.close();
        }
        
        return;
      }
      
      // Pass through all other events
      onEvent(data);
      
      // Check for report completion in other event formats
      if ((data.status === 'COMPLETED' || data.type === 'report_status' && data.status === 'COMPLETED') && data.content) {
        onComplete({
          content: data.content,
          status: 'COMPLETED',
          completed_at: new Date().toISOString()
        });
        
        // Also close EventSource when completion is detected in other formats
        console.log('[DEBUG:SERVICE] Report completion detected, closing EventSource');
        eventSource.close();
      }
    } catch (error) {
      console.error('[DEBUG:SERVICE] Error handling SSE message:', error, 'raw data:', event.data);
      onError(error);
    }
  };
  
  eventSource.onerror = (error) => {
    console.error(`[DEBUG:SERVICE] SSE error for report ${reportId}:`, error);
    
    // Implement reconnect logic
    if (reconnectAttempts < maxReconnectAttempts) {
      reconnectAttempts++;
      console.log(`[DEBUG:SERVICE] Reconnect attempt ${reconnectAttempts} of ${maxReconnectAttempts} in ${reconnectDelay}ms`);
      
      // Only try to reconnect if the connection is closed
      if (eventSource.readyState === EventSource.CLOSED) {
        setTimeout(() => {
          // Create a new connection
          const newEventSource = new EventSource(`${API_URL}/api/research/reports/${reportId}/stream/?token=${token}&format=txt`);
          // Replace the old eventSource with the new one
          Object.assign(eventSource, newEventSource);
        }, reconnectDelay);
      }
    } else {
      console.error(`[DEBUG:SERVICE] Max reconnect attempts (${maxReconnectAttempts}) reached, giving up`);
      onError('Connection to research stream lost and could not be reestablished.');
      eventSource.close();
    }
  };
  
  // Return a cleanup function
  return () => {
    console.log(`[DEBUG:SERVICE] Closing stream for report ${reportId}`);
    eventSource.close();
  };
};

/**
 * Provide feedback on a research plan
 * @param id The research report ID
 * @param data The feedback data
 * @param handlers Event handlers for different SSE events
 * @returns A function to close the SSE connection
 */
export const provideFeedback = (
  id: number, 
  data: ResearchFeedbackRequest, 
  handlers: ResearchEventHandlers
): () => void => {
  const token = localStorage.getItem('accessToken');
  
  // First, submit the feedback
  const feedbackUrl = new URL(`${API_URL}/api/research/reports/${id}/feedback/`);
  feedbackUrl.searchParams.append('token', token || ''); // Add token to query params
  
  axios.post(feedbackUrl.toString(), data, {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    }
  }).catch(error => {
    console.error('Error submitting feedback:', error);
    if (handlers.onError) {
      handlers.onError(error);
    }
  });
  
  // Then, set up the SSE connection
  // Create URL with token in query parameter for authentication
  const url = new URL(`${API_URL}/api/research/reports/${id}/stream/`);
  url.searchParams.append('token', token || '');
  // Add format parameter to ensure the server knows we want SSE format
  url.searchParams.append('format', 'txt');
  
  // Create event source
  let eventSource: EventSource | null = null;
  
  try {
    eventSource = new EventSource(url.toString());
  } catch (error) {
    console.error('Failed to create EventSource:', error);
    if (handlers.onError) {
      handlers.onError(new Error(`Failed to establish SSE connection: ${error instanceof Error ? error.message : String(error)}`));
    }
    return () => {}; // Return empty cleanup function
  }
  
  eventSource.onmessage = (event) => {
    try {
      console.log('SSE feedback event received (raw):', event.data.slice(0, 100) + (event.data.length > 100 ? '...' : ''));
      
      // Extract the JSON data - handle both formats (with or without data: prefix)
      let jsonStr = '';
      const trimmedData = event.data.trim();
      
      if (trimmedData.startsWith('data: ')) {
        // Standard SSE format with 'data:' prefix
        jsonStr = trimmedData.slice(6); // Remove 'data: ' prefix
      } else {
        // Directly JSON data without prefix (possible with some proxy configurations)
        jsonStr = trimmedData;
        console.warn('Received feedback data without data: prefix, attempting to parse directly');
      }
      
      if (!jsonStr) {
        console.warn('Received empty feedback data after processing');
        return;
      }
      
      // Parse the data
      const data = JSON.parse(jsonStr);
      console.log('SSE feedback event processed:', data);
      
      // Handle different event types
      if (data.__interrupt__ && handlers.onPlan) {
        handlers.onPlan(data.__interrupt__[0].value);
      } else if (data.compile_final_report && handlers.onComplete) {
        handlers.onComplete(data.compile_final_report.final_report);
        
        // Close the EventSource when report is completed to prevent duplicates
        console.log('[DEBUG:SERVICE] Report completed through feedback, closing EventSource');
        eventSource?.close();
      } else if (data.type === 'report_completed' || 
                (data.type === 'report_status' && data.status === 'COMPLETED')) {
        if (handlers.onComplete) {
          // Extract content if available
          const content = data.content || (data.type === 'report_completed' ? data : null);
          handlers.onComplete(content);
        }
        
        // Close the EventSource when report is completed to prevent duplicates
        console.log('[DEBUG:SERVICE] Report completed through feedback status update, closing EventSource');
        eventSource?.close();
      } else if (handlers.onProgress) {
        handlers.onProgress(data);
      }
    } catch (error) {
      console.error('Error parsing SSE feedback event:', error, 'Raw data:', event.data);
      if (handlers.onError) {
        handlers.onError(error instanceof Error ? error : new Error(String(error)));
      }
    }
  };
  
  eventSource.onerror = (error) => {
    console.error('SSE feedback connection error:', error);
    if (handlers.onError) {
      handlers.onError(error instanceof Error ? error : new Error('SSE connection error'));
    }
    if (eventSource) {
      eventSource.close();
    }
  };
  
  // Return function to close the connection
  return () => {
    console.log('Closing SSE feedback connection');
    if (eventSource) {
      eventSource.close();
    }
  };
};

/**
 * Approve a research plan
 * @param id The research report ID
 * @param handlers Event handlers for different SSE events
 * @returns A function to close the SSE connection
 */
export const approveResearchPlan = (id: number, handlers: ResearchEventHandlers): () => void => {
  const token = localStorage.getItem('accessToken');
  
  // First, submit the approval
  const approveUrl = new URL(`${API_URL}/api/research/reports/${id}/approve/`);
  approveUrl.searchParams.append('token', token || ''); // Add token to query params
  
  axios.post(approveUrl.toString(), {}, {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    }
  }).catch(error => {
    console.error('Error approving plan:', error);
    if (handlers.onError) {
      handlers.onError(error);
    }
  });
  
  // Then, set up the SSE connection
  // Create URL with token in query parameter for authentication
  const url = new URL(`${API_URL}/api/research/reports/${id}/stream/`);
  url.searchParams.append('token', token || '');
  // Add format parameter to ensure the server knows we want SSE format
  url.searchParams.append('format', 'txt');
  
  // Create event source
  let eventSource: EventSource | null = null;
  
  try {
    eventSource = new EventSource(url.toString());
  } catch (error) {
    console.error('Failed to create EventSource:', error);
    if (handlers.onError) {
      handlers.onError(new Error(`Failed to establish SSE connection: ${error instanceof Error ? error.message : String(error)}`));
    }
    return () => {}; // Return empty cleanup function
  }
  
  eventSource.onmessage = (event) => {
    try {
      console.log('SSE approval event received (raw):', event.data.slice(0, 100) + (event.data.length > 100 ? '...' : ''));
      
      // Extract the JSON data - handle both formats (with or without data: prefix)
      let jsonStr = '';
      const trimmedData = event.data.trim();
      
      if (trimmedData.startsWith('data: ')) {
        // Standard SSE format with 'data:' prefix
        jsonStr = trimmedData.slice(6); // Remove 'data: ' prefix
      } else {
        // Directly JSON data without prefix (possible with some proxy configurations)
        jsonStr = trimmedData;
        console.warn('Received approval data without data: prefix, attempting to parse directly');
      }
      
      if (!jsonStr) {
        console.warn('Received empty approval data after processing');
        return;
      }
      
      // Parse the data
      const data = JSON.parse(jsonStr);
      console.log('SSE approval event processed:', data);
      
      // Handle different event types
      if (data.compile_final_report && handlers.onComplete) {
        handlers.onComplete(data.compile_final_report.final_report);
        
        // Close the EventSource when report is completed to prevent duplicates
        console.log('[DEBUG:SERVICE] Report completed through approval, closing EventSource');
        eventSource?.close();
      } else if (data.type === 'report_completed' || 
                (data.type === 'report_status' && data.status === 'COMPLETED')) {
        if (handlers.onComplete) {
          // Extract content if available
          const content = data.content || (data.type === 'report_completed' ? data : null);
          handlers.onComplete(content);
        }
        
        // Close the EventSource when report is completed to prevent duplicates
        console.log('[DEBUG:SERVICE] Report completed through approval status update, closing EventSource');
        eventSource?.close();
      } else if (handlers.onProgress) {
        handlers.onProgress(data);
      }
    } catch (error) {
      console.error('Error parsing SSE approval event:', error, 'Raw data:', event.data);
      if (handlers.onError) {
        handlers.onError(error instanceof Error ? error : new Error(String(error)));
      }
    }
  };
  
  eventSource.onerror = (error) => {
    console.error('SSE approval connection error:', error);
    if (handlers.onError) {
      handlers.onError(error instanceof Error ? error : new Error('SSE connection error'));
    }
    if (eventSource) {
      eventSource.close();
    }
  };
  
  // Return function to close the connection
  return () => {
    console.log('Closing SSE approval connection');
    if (eventSource) {
      eventSource.close();
    }
  };
};

/**
 * Delete a research report
 * @param id The research report ID
 * @returns True if deletion was successful
 */
export const deleteResearchReport = async (id: number): Promise<boolean> => {
  const token = localStorage.getItem('accessToken');
  
  try {
    await axios.delete(`${API_URL}/api/research/reports/${id}/`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    return true;
  } catch (error) {
    console.error('Error deleting research report:', error);
    return false;
  }
};

export default {
  startResearch,
  getResearchReports,
  getResearchReport,
  streamResearchReport,
  provideFeedback,
  approveResearchPlan,
  deleteResearchReport
}; 