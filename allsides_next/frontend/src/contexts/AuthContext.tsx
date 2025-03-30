'use client';

import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import api, { auth, stats, UserStats as ApiUserStats } from '../services/api';
import type { User } from '../services/api';

// Extend the API UserStats type with our additional properties
interface UserStats extends ApiUserStats {
  daily_query_limit?: number;
}

interface AuthContextData {
  isAuthenticated: boolean;
  user: User | null;
  userStats: UserStats | null;
  loading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  refreshStats: () => Promise<UserStats | null>;
  checkAuth: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextData>({} as AuthContextData);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [userStats, setUserStats] = useState<UserStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const lastRefreshTime = useRef<number>(0);
  const isRefreshing = useRef<boolean>(false);
  const router = useRouter();

  const refreshStats = useCallback(async (forceRefresh = false): Promise<UserStats | null> => {
    try {
      const now = Date.now();
      // Prevent concurrent refreshes and rate limit unless force refresh is requested
      if (!forceRefresh && (isRefreshing.current || (now - lastRefreshTime.current < 5000))) {
        console.log('Skipping refresh: too soon or already refreshing');
        return userStats;
      }

      isRefreshing.current = true;
      setError(null);
      
      console.log('Refreshing user stats...');
      const userStatsData = await stats.get();
      
      // Compare with current stats to detect changes
      if (userStats) {
        // Use a more type-safe approach for comparison
        const oldLimit = 'daily_query_limit' in userStats ? userStats.daily_query_limit : undefined;
        const newLimit = 'daily_query_limit' in userStatsData ? userStatsData.daily_query_limit : undefined;
        
        const hasChanged = 
          userStatsData.daily_query_count !== userStats.daily_query_count ||
          userStatsData.remaining_queries !== userStats.remaining_queries ||
          newLimit !== oldLimit;
          
        if (hasChanged) {
          console.log('User stats changed:', {
            old: userStats,
            new: userStatsData
          });
        }
      }
      
      setUserStats(userStatsData);
      lastRefreshTime.current = now;

      // Check if queries are exhausted and set timer if needed
      if (userStatsData.remaining_queries === 0 && userStatsData.reset_time) {
        const resetTime = new Date(userStatsData.reset_time);
        const now = new Date();
        if (resetTime > now) {
          const timeUntilReset = resetTime.getTime() - now.getTime();
          console.log(`Setting refresh timer for ${timeUntilReset}ms (${new Date(now.getTime() + timeUntilReset).toLocaleTimeString()})`);
          setTimeout(() => refreshStats(true), timeUntilReset + 1000); // Add 1 second buffer
        }
      }

      return userStatsData;
    } catch (error: any) {
      if (error.message.includes('Network error')) {
        setError('Network error. Please check your connection.');
        throw error;
      }
      if (error.response?.status === 401) {
        auth.logout();
        setUser(null);
        setUserStats(null);
        router.replace('/login');
      }
      setError(error.message || 'Failed to load user data. Please try logging in again.');
      throw error;
    } finally {
      isRefreshing.current = false;
    }
  }, [userStats, router]);

  const checkAuth = useCallback(async (): Promise<boolean> => {
    const token = localStorage.getItem('accessToken');
    if (!token) {
      setUser(null);
      setUserStats(null);
      setLoading(false);
      return false;
    }

    try {
      // Only fetch user data if we don't have it
      if (!user) {
        const userData = await auth.getUser();
        setUser(userData);
      }
      // Only fetch stats if we don't have them
      if (!userStats) {
        await refreshStats();
      }
      setLoading(false);
      return true;
    } catch (error) {
      console.error('Auth check failed:', error);
      if (error instanceof Error && error.message.includes('Network error')) {
        setError('Network error. Please check your connection.');
        setLoading(false);
        return !!user;
      }
      auth.logout();
      setUser(null);
      setUserStats(null);
      setLoading(false);
      return false;
    }
  }, [user, userStats, refreshStats]);

  const login = useCallback(async (username: string, password: string) => {
    try {
      setLoading(true);
      setError(null);
      
      const { user: userData } = await auth.login(username, password);
      setUser(userData);
      const isAuthenticated = await checkAuth();
      
      if (isAuthenticated) {
        router.replace('/');
      } else {
        throw new Error('Authentication failed');
      }
    } catch (error: any) {
      console.error('Login error:', error);
      if (error.message.includes('Network error')) {
        setError('Network error. Please check your connection and try again.');
      } else {
        setError(error.message || 'Login failed. Please check your credentials.');
      }
      throw error;
    } finally {
      setLoading(false);
    }
  }, [checkAuth, router]);

  const logout = useCallback(() => {
    auth.logout();
    setUser(null);
    setUserStats(null);
    setError(null);
    router.replace('/login');
  }, [router]);

  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const isAuthenticated = await checkAuth();
        if (!isAuthenticated && window.location.pathname !== '/login') {
          router.replace('/login');
        }
      } catch (error) {
        if (error instanceof Error && !error.message.includes('Network error')) {
          router.replace('/login');
        }
      }
    };

    initializeAuth();
  }, [checkAuth, router]);

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated: !!user,
        user,
        userStats,
        loading,
        error,
        login,
        logout,
        refreshStats,
        checkAuth
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}; 