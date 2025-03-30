export const formatTimeRemaining = (resetTime: string | null): string => {
  // Get current time and midnight tonight in user's timezone
  const now = new Date();
  const midnight = new Date();
  midnight.setHours(24, 0, 0, 0);

  // Calculate difference in hours and minutes
  const diffMs = midnight.getTime() - now.getTime();
  const hours = Math.floor(diffMs / (1000 * 60 * 60));
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));

  // Return formatted string
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}; 