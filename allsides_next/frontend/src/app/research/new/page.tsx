'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import researchService, { StartResearchRequest } from '@/services/researchService';

const NewResearchPage: React.FC = () => {
  const [topic, setTopic] = useState<string>('');
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  
  // Advanced configuration options
  const [searchApi, setSearchApi] = useState<string>('tavily');
  const [plannerProvider, setPlannerProvider] = useState<string>('openai');
  const [plannerModel, setPlannerModel] = useState<string>('gpt-4o');
  const [writerProvider, setWriterProvider] = useState<string>('openai');
  const [writerModel, setWriterModel] = useState<string>('gpt-4o');
  const [searchDepth, setSearchDepth] = useState<number>(2);
  
  // Model options by provider
  const modelOptions = {
    openai: ['gpt-4o', 'gpt-4', 'gpt-3.5-turbo'],
    anthropic: ['claude-3-7-sonnet-latest', 'claude-3-5-sonnet-latest', 'claude-3-haiku-latest'],
    ollama: ['llama3.2', 'gemma3:1b', 'gemma3:4b']
  };
  
  // Update model when provider changes
  const handlePlannerProviderChange = (provider: string) => {
    setPlannerProvider(provider);
    // Set first model of the selected provider
    if (modelOptions[provider as keyof typeof modelOptions]) {
      setPlannerModel(modelOptions[provider as keyof typeof modelOptions][0]);
    }
  };
  
  // Update model when provider changes
  const handleWriterProviderChange = (provider: string) => {
    setWriterProvider(provider);
    // Set first model of the selected provider
    if (modelOptions[provider as keyof typeof modelOptions]) {
      setWriterModel(modelOptions[provider as keyof typeof modelOptions][0]);
    }
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!topic.trim()) {
      setError('Please enter a research topic');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const requestData: StartResearchRequest = {
        topic: topic.trim(),
      };
      
      // Add advanced configuration if enabled
      if (showAdvanced) {
        requestData.config = {
          search_api: searchApi,
          planner_provider: plannerProvider,
          planner_model: plannerModel,
          writer_provider: writerProvider,
          writer_model: writerModel,
          max_search_depth: searchDepth,
        };
      }
      
      const result = await researchService.startResearch(requestData);
      router.push(`/research/${result.id}`);
    } catch (err: any) {
      console.error('Error starting research:', err);
      
      // Extract error message from API response if available
      const errorMsg = err.response?.data?.error || 'Failed to start research. Please try again later.';
      setError(errorMsg);
      setLoading(false);
    }
  };
  
  return (
    <div className="container mx-auto px-4 py-8 max-w-3xl">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-bold mb-6 dark:text-white">Start New Research</h1>
        
        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
            <p>{error}</p>
          </div>
        )}
        
        <form onSubmit={handleSubmit}>
          <div className="mb-6">
            <label 
              htmlFor="topic" 
              className="block text-gray-700 dark:text-gray-300 text-sm font-medium mb-2"
            >
              Research Topic
            </label>
            <textarea
              id="topic"
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              placeholder="Enter a specific research question or topic..."
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              required
            />
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Be specific about what you want to learn. For example: "Compare the environmental impact of electric vs. combustion engine cars"
            </p>
          </div>
          
          <div className="mb-6">
            <button
              type="button"
              className="text-blue-600 dark:text-blue-400 text-sm font-medium hover:underline focus:outline-none"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? '- Hide Advanced Options' : '+ Show Advanced Options'}
            </button>
          </div>
          
          {showAdvanced && (
            <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-md">
              <h2 className="text-lg font-medium mb-4 dark:text-white">Advanced Configuration</h2>
              
              <div className="mt-4 space-y-4">
                <div>
                  <label 
                    htmlFor="searchApi" 
                    className="block text-gray-700 dark:text-gray-300 text-sm font-medium mb-2"
                  >
                    Search API
                  </label>
                  <select
                    id="searchApi"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                    value={searchApi}
                    onChange={(e) => setSearchApi(e.target.value)}
                  >
                    <option value="tavily">Tavily</option>
                    <option value="perplexity">Perplexity</option>
                    <option value="googlesearch">Google</option>
                  </select>
                </div>
                
                <div>
                  <label 
                    htmlFor="searchDepth" 
                    className="block text-gray-700 dark:text-gray-300 text-sm font-medium mb-2"
                  >
                    Search Depth
                  </label>
                  <select
                    id="searchDepth"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                    value={searchDepth}
                    onChange={(e) => setSearchDepth(parseInt(e.target.value))}
                  >
                    <option value="1">1 - Fast</option>
                    <option value="2">2 - Standard</option>
                    <option value="3">3 - Deep</option>
                  </select>
                </div>
                
                <div>
                  <label 
                    htmlFor="plannerProvider" 
                    className="block text-gray-700 dark:text-gray-300 text-sm font-medium mb-2"
                  >
                    Planner Provider
                  </label>
                  <select
                    id="plannerProvider"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                    value={plannerProvider}
                    onChange={(e) => handlePlannerProviderChange(e.target.value)}
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="ollama">Ollama (Local)</option>
                  </select>
                </div>
                
                <div>
                  <label 
                    htmlFor="plannerModel" 
                    className="block text-gray-700 dark:text-gray-300 text-sm font-medium mb-2"
                  >
                    Planner Model
                  </label>
                  <select
                    id="plannerModel"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                    value={plannerModel}
                    onChange={(e) => setPlannerModel(e.target.value)}
                  >
                    {modelOptions[plannerProvider as keyof typeof modelOptions]?.map(model => (
                      <option key={model} value={model}>{model}</option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <label 
                    htmlFor="writerProvider" 
                    className="block text-gray-700 dark:text-gray-300 text-sm font-medium mb-2"
                  >
                    Writer Provider
                  </label>
                  <select
                    id="writerProvider"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                    value={writerProvider}
                    onChange={(e) => handleWriterProviderChange(e.target.value)}
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="ollama">Ollama (Local)</option>
                  </select>
                </div>
                
                <div>
                  <label 
                    htmlFor="writerModel" 
                    className="block text-gray-700 dark:text-gray-300 text-sm font-medium mb-2"
                  >
                    Writer Model
                  </label>
                  <select
                    id="writerModel"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                    value={writerModel}
                    onChange={(e) => setWriterModel(e.target.value)}
                  >
                    {modelOptions[writerProvider as keyof typeof modelOptions]?.map(model => (
                      <option key={model} value={model}>{model}</option>
                    ))}
                  </select>
                </div>
              </div>
              
              <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">
                Advanced options allow you to customize the research process. The default configuration provides a good balance of speed and quality.
              </p>
            </div>
          )}
          
          <div className="flex items-center justify-end space-x-3">
            <button
              type="button"
              onClick={() => router.push('/research')}
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 bg-blue-600 rounded-md shadow-sm text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Starting...' : 'Start Research'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default NewResearchPage; 