'use client';

import React, { useState, useRef } from 'react';
import { ResearchEventHandlers } from '@/services/researchService';

// Simple component for directly testing the EventSource API
const SSEDebugger: React.FC = () => {
  const [events, setEvents] = useState<any[]>([]);
  const [status, setStatus] = useState('Not connected');
  const [url, setUrl] = useState('/api/research/sse-debug/');
  const [isConnected, setIsConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const connect = () => {
    try {
      // Close existing connection
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }

      setStatus('Connecting...');
      setEvents([]);

      // Determine full URL
      let fullUrl = url;
      if (!url.startsWith('http')) {
        const baseUrl = window.location.protocol + '//' + window.location.host;
        fullUrl = baseUrl + (url.startsWith('/') ? url : ('/' + url));
      }

      console.log('Connecting to SSE endpoint:', fullUrl);
      
      // Create new EventSource
      const eventSource = new EventSource(fullUrl);
      eventSourceRef.current = eventSource;

      // Set up event handlers
      eventSource.onopen = () => {
        console.log('SSE connection opened');
        setStatus('Connected');
        setIsConnected(true);
        addEvent('system', { message: 'Connection opened' });
      };

      eventSource.onmessage = (event) => {
        console.log('SSE message received:', event.data);
        try {
          const data = JSON.parse(event.data);
          addEvent(data.type || 'message', data);
        } catch (error) {
          console.error('Error parsing SSE data:', error);
          addEvent('error', { 
            message: 'Failed to parse event data', 
            error: error instanceof Error ? error.message : String(error),
            raw: event.data
          });
        }
      };

      eventSource.onerror = (error) => {
        console.error('SSE connection error:', error);
        setStatus('Error - See console');
        addEvent('error', { 
          message: 'Connection error',
          error: error instanceof ErrorEvent ? error.message : 'Unknown error' 
        });
      };
    } catch (error) {
      console.error('Failed to create EventSource:', error);
      setStatus('Connection Failed');
      addEvent('error', { 
        message: 'Failed to create EventSource',
        error: error instanceof Error ? error.message : String(error)
      });
    }
  };

  const disconnect = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setStatus('Disconnected');
      setIsConnected(false);
      addEvent('system', { message: 'Connection closed' });
    }
  };

  const addEvent = (type: string, data: any) => {
    setEvents(prev => [
      ...prev,
      {
        id: Date.now(),
        type,
        data,
        timestamp: new Date().toISOString()
      }
    ]);
  };

  const clearEvents = () => {
    setEvents([]);
  };

  // Get appropriate class for event type
  const getEventClass = (type: string) => {
    switch(type) {
      case 'error': return 'bg-red-50 border-red-500';
      case 'system': return 'bg-gray-50 border-gray-500';
      case 'connection_established': return 'bg-green-50 border-green-500';
      case 'test_event': return 'bg-blue-50 border-blue-500';
      case 'test_complete': return 'bg-purple-50 border-purple-500';
      default: return 'bg-white border-gray-300';
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">SSE Debug Page</h1>
      
      {/* Status and Controls */}
      <div className="bg-white shadow rounded-lg p-4 mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <span className="font-medium mr-2">Status:</span>
            <span className={`px-2 py-1 rounded ${
              status === 'Connected' ? 'bg-green-100 text-green-800' : 
              status === 'Connecting...' ? 'bg-yellow-100 text-yellow-800' :
              status.includes('Error') ? 'bg-red-100 text-red-800' :
              'bg-gray-100 text-gray-800'
            }`}>
              {status}
            </span>
          </div>
          <div className="space-x-2">
            <button 
              onClick={connect} 
              disabled={isConnected}
              className={`px-4 py-2 rounded font-medium ${
                isConnected ? 'bg-gray-100 text-gray-500 cursor-not-allowed' : 'bg-blue-500 text-white hover:bg-blue-600'
              }`}
            >
              Connect
            </button>
            <button 
              onClick={disconnect} 
              disabled={!isConnected}
              className={`px-4 py-2 rounded font-medium ${
                !isConnected ? 'bg-gray-100 text-gray-500 cursor-not-allowed' : 'bg-red-500 text-white hover:bg-red-600'
              }`}
            >
              Disconnect
            </button>
            <button 
              onClick={clearEvents}
              className="px-4 py-2 rounded font-medium bg-gray-200 text-gray-700 hover:bg-gray-300"
            >
              Clear Events
            </button>
          </div>
        </div>
        
        <div className="mb-2">
          <label htmlFor="sseUrl" className="block font-medium text-gray-700">Event Source URL</label>
          <input 
            id="sseUrl"
            type="text" 
            value={url} 
            onChange={(e) => setUrl(e.target.value)}
            className="mt-1 p-2 w-full border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter SSE URL"
          />
          <p className="text-sm text-gray-500 mt-1">
            Enter a relative URL (e.g., /api/research/sse-debug/) or an absolute URL
          </p>
        </div>
      </div>
      
      {/* Events Display */}
      <div className="bg-white shadow rounded-lg p-4">
        <h2 className="text-xl font-semibold mb-4">Events ({events.length})</h2>
        
        {events.length === 0 ? (
          <div className="text-gray-500 text-center py-6">
            No events received yet. Click "Connect" to start listening.
          </div>
        ) : (
          <div className="space-y-3">
            {events.map((event) => (
              <div 
                key={event.id}
                className={`p-3 border-l-4 rounded ${getEventClass(event.type)}`}
              >
                <div className="flex justify-between text-sm text-gray-500 mb-1">
                  <span className="font-medium">{event.type}</span>
                  <span>{new Date(event.timestamp).toLocaleTimeString()}</span>
                </div>
                <pre className="text-sm overflow-x-auto whitespace-pre-wrap">
                  {JSON.stringify(event.data, null, 2)}
                </pre>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default SSEDebugger; 