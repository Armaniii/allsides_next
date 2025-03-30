import axios from 'axios';

// Type definitions
interface CustomAxiosRequestConfig {
  baseURL?: string;
  headers?: Record<string, string>;
  timeout?: number;
  url?: string;
  _retry?: boolean;
  method?: string;
  data?: any;
}

interface CustomAxiosResponse<T = any> {
  data: T;
  status: number;
  statusText: string;
  headers: Record<string, string>;
  config: CustomAxiosRequestConfig;
}

interface ErrorWithResponse {
  response?: {
    status: number;
    data?: {
      detail?: string;
    };
  };
  message?: string;
  config?: CustomAxiosRequestConfig;
}

interface TokenResponse {
  access: string;
  refresh: string;
}

export interface User {
  id: number;
  username: string;
  first_name: string;
  last_name: string;
  email: string;
  daily_query_limit: number;
  daily_query_count: number;
  reset_time: string | null;
}

export interface UserStats {
  remaining_queries: number;
  reset_time: string | null;
  daily_query_count: number;
  total_queries: number;
  allstars: number;
}

interface AuthError {
  detail: string;
  code?: string;
}

// Define the base URL for the API
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9000';

// Ensure the URL has the correct format - add /api only if it's not already there
const normalizedBaseUrl = API_BASE_URL.endsWith('/api') 
  ? API_BASE_URL 
  : `${API_BASE_URL}/api`;

console.log('🌐 API Base URL:', normalizedBaseUrl);

// Create axios instance with default config
const api = axios.create({
  baseURL: normalizedBaseUrl,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  timeout: 60000, // 60 seconds
  withCredentials: true // Enable credentials for cross-origin requests
});

// Add detailed request logging
api.interceptors.request.use(
  (config: any) => {
    const token = localStorage.getItem('accessToken');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    console.log('🚀 Outgoing Request:', {
      method: config.method?.toUpperCase(),
      url: config.url,
      baseURL: config.baseURL,
      headers: {
        ...config.headers,
        Authorization: config.headers?.Authorization ? 
          `Bearer ${config.headers.Authorization.split(' ')[1].substring(0, 10)}...` : 
          'No token'
      },
      data: config.data,
      withCredentials: config.withCredentials
    });
    return config;
  },
  (error: unknown) => {
    console.error('❌ Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response: any) => {
    console.log('✅ Response Success:', {
      status: response.status,
      statusText: response.statusText,
      url: response.config.url,
      headers: response.headers,
      data: response.data ? JSON.stringify(response.data, null, 2) : undefined
    });
    return response;
  },
  async (error: any) => {
    console.error('❌ Response Error:', {
      status: error.response?.status,
      statusText: error.response?.statusText,
      url: error.config?.url,
      message: error.message,
      headers: error.response?.headers,
      data: error.response?.data ? JSON.stringify(error.response.data, null, 2) : undefined,
      config: {
        method: error.config?.method,
        baseURL: error.config?.baseURL,
        headers: error.config?.headers,
        withCredentials: error.config?.withCredentials
      }
    });

    const originalRequest = error.config;

    // Handle 401 Unauthorized error
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem('refreshToken');
        if (!refreshToken) {
          throw new Error('No refresh token available');
        }

        const response = await api.post<TokenResponse>('/token/refresh/', { refresh: refreshToken });
        const { access } = response.data;
        
        localStorage.setItem('accessToken', access);
        
        if (originalRequest.headers) {
          originalRequest.headers.Authorization = `Bearer ${access}`;
        }
        
        return api(originalRequest);
      } catch (refreshError) {
        auth.clearTokens();
        auth.redirectToLogin();
        return Promise.reject(new Error('Session expired'));
      }
    }

    return Promise.reject(error);
  }
);

export const auth = {
  setTokens: (tokens: TokenResponse) => {
    console.log('💾 Setting tokens');
    localStorage.setItem('accessToken', tokens.access);
    localStorage.setItem('refreshToken', tokens.refresh);
  },

  clearTokens: () => {
    console.log('🗑️ Clearing tokens');
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
  },

  redirectToLogin: () => {
    if (typeof window !== 'undefined' && !window.location.pathname.includes('/login')) {
      console.log('🔄 Redirecting to login page');
      window.location.href = '/login';
    }
  },

  login: async (username: string, password: string): Promise<{ user: User; tokens: TokenResponse }> => {
    console.log('🔑 Attempting login...');
    try {
      // Get tokens
      console.log('🎯 Requesting tokens...');
      const tokenResponse = await api.post<TokenResponse>('/token/', { username, password });
      console.log('✅ Tokens received');
      auth.setTokens(tokenResponse.data);

      // Get user data
      console.log('🎯 Fetching user data...');
      const userResponse = await api.get<User>('/users/me/');
      console.log('✅ User data received');
      
      return {
        user: userResponse.data,
        tokens: tokenResponse.data
      };
    } catch (error: any) {
      console.error('❌ Login failed:', {
        status: error.response?.status,
        data: error.response?.data,
        message: error.message
      });
      
      if (error.response?.status === 401) {
        throw new Error('Invalid username or password');
      }
      throw new Error(error.response?.data?.detail || error.message || 'Login failed. Please try again.');
    }
  },
  
  logout: async () => {
    try {
      auth.clearTokens();
      auth.redirectToLogin();
    } catch (error) {
      console.error('Logout error:', error);
      // Still clear tokens and redirect even if the API call fails
      auth.clearTokens();
      auth.redirectToLogin();
    }
  },

  isAuthenticated: (): boolean => {
    return !!localStorage.getItem('accessToken');
  },

  getUser: async (): Promise<User> => {
    try {
      const response = await api.get<User>('/users/me/');
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 401) {
        auth.clearTokens();
        auth.redirectToLogin();
      }
      throw error;
    }
  }
};

export const stats = {
  get: async (): Promise<UserStats> => {
    const response = await api.get<UserStats>('/users/stats/');
    return response.data;
  }
};

export default api; 