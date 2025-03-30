import React, { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Button } from './ui/button';
import { Loader2, Send } from 'lucide-react';
import { cn } from '../lib/utils';

const QueryInput: React.FC = () => {
  const { userStats } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [queryLimitAnimation, setQueryLimitAnimation] = useState(false);

  useEffect(() => {
    if (userStats?.remaining_queries === 0) {
      setQueryLimitAnimation(true);
    } else {
      setQueryLimitAnimation(false);
    }
  }, [userStats?.remaining_queries]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsLoading(true);
    // Handle form submission
    setIsLoading(false);
  };

  return (
    <form onSubmit={handleSubmit}>
      <Button 
        type="submit"
        className={cn(
          "w-full md:w-auto",
          queryLimitAnimation && "animate-pulse"
        )}
        disabled={isLoading || userStats?.remaining_queries === 0}
      >
        {isLoading ? (
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        ) : (
          <Send className="mr-2 h-4 w-4" />
        )}
        {isLoading ? "Processing..." : "Submit"}
      </Button>
    </form>
  );
};

export default QueryInput; 