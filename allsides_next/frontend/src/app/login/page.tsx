'use client';

import React, { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { motion, AnimatePresence } from 'framer-motion';
import { useRouter } from 'next/navigation';
import api from '@/services/api';
import Image from 'next/image';

type TabType = 'login' | 'signup';

interface SignupData {
  username: string;
  email: string;
  password: string;
  password2: string;
  first_name: string;
  last_name: string;
  bias_rating: string;
  [key: string]: string;  // Index signature for string access
}

interface ApiErrorResponse {
  error?: string;
  details?: string;
  help?: string;
  detail?: string;
  [key: string]: any;
}

type ErrorType = {
  [key: string]: string[];
} & ApiErrorResponse;

const LoginPage = () => {
  const [activeTab, setActiveTab] = useState<TabType>('login');
  const [errors, setErrors] = useState<ErrorType>({});
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const router = useRouter();

  // Login form state
  const [loginData, setLoginData] = useState({
    username: '',
    password: '',
  });

  // Signup form state
  const [signupData, setSignupData] = useState({
    username: '',
    email: '',
    password: '',
    password2: '',
    first_name: '',
    last_name: '',
    bias_rating: 'C',
  });

  // Update the label classes to use darker text
  const labelClass = "block text-sm font-medium text-gray-900";
  const inputClass = "mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-900 focus:border-purple-500 focus:outline-none focus:ring-1 focus:ring-purple-500";

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrors({});
    setLoading(true);

    try {
      await login(loginData.username, loginData.password);
      router.push('/');
    } catch (error: any) {
      if (error.response?.data) {
        setErrors(error.response.data);
      } else {
        setErrors({ non_field_errors: [error.message || 'Failed to login'] });
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrors({});
    setLoading(true);

    try {
      // Client-side validation
      if (signupData.password !== signupData.password2) {
        setErrors({ password2: ["Passwords don't match"] });
        setLoading(false);
        return;
      }

      // Ensure all required fields are present and properly formatted
      const requiredFields = ['username', 'email', 'password', 'password2', 'first_name', 'last_name', 'bias_rating'] as const;
      const missingFields = requiredFields.filter(field => !signupData[field]);
      if (missingFields.length > 0) {
        setErrors({ non_field_errors: [`Missing required fields: ${missingFields.join(', ')}`] });
        setLoading(false);
        return;
      }

      // Clean and format the data
      const formattedData: SignupData = {
        ...signupData,
        email: signupData.email.toLowerCase().trim(),
        username: signupData.username.trim(),
        first_name: signupData.first_name.trim(),
        last_name: signupData.last_name.trim(),
        bias_rating: signupData.bias_rating.toUpperCase().trim()
      };

      // Register the user with proper headers
      await api.post('/register/', formattedData, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });
      
      // If registration successful, login with the new credentials
      await login(formattedData.username, formattedData.password);
      router.push('/');
    } catch (error: any) {
      console.error('Registration error:', error.response?.data || error.message);
      
      const responseErrors = error.response?.data as ApiErrorResponse || {};
      
      // Handle different types of errors
      if (typeof responseErrors === 'string') {
        setErrors({ non_field_errors: [responseErrors] });
      } else if (Array.isArray(responseErrors)) {
        setErrors({ non_field_errors: responseErrors });
      } else if (responseErrors.error && responseErrors.details) {
        // Handle structured error responses
        setErrors({
          non_field_errors: [
            `${responseErrors.error}: ${responseErrors.details}`,
            responseErrors.help || ''
          ].filter(Boolean)
        });
      } else {
        setErrors(responseErrors as ErrorType);
      }
    } finally {
      setLoading(false);
    }
  };

  // Helper function to render field errors
  const renderFieldError = (fieldName: keyof ErrorType) => {
    const fieldErrors = errors[fieldName];
    if (!fieldErrors || fieldErrors.length === 0) return null;
    
    return (
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-sm text-red-600 mt-1"
      >
        {Array.isArray(fieldErrors) ? fieldErrors[0] : fieldErrors}
      </motion.div>
    );
  };

  // Helper function to get input class based on error state
  const getInputClass = (fieldName: keyof ErrorType) => {
    return `${inputClass} ${errors[fieldName] ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-indigo-50 flex items-center justify-center p-4">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative flex flex-col items-center"
      >
        {/* Logo */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8 relative"
        >
          <div className="relative w-24 h-24">
            <Image
              src="/logo.webp"
              alt="AllStances Logo"
              fill
              className="object-contain"
            />
          </div>
          {/* Decorative ring */}
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.5 }}
            className="absolute inset-0 rounded-full bg-gradient-to-r from-purple-500/10 to-indigo-500/10 -z-10 blur-md"
          />
        </motion.div>

        {/* Form Container */}
        <div className="bg-white rounded-2xl shadow-xl w-full max-w-md overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b border-gray-200">
            <button
              onClick={() => setActiveTab('login')}
              className={`flex-1 py-4 text-sm font-medium transition-colors ${
                activeTab === 'login'
                  ? 'text-purple-600 border-b-2 border-purple-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Login
            </button>
            <button
              onClick={() => setActiveTab('signup')}
              className={`flex-1 py-4 text-sm font-medium transition-colors ${
                activeTab === 'signup'
                  ? 'text-purple-600 border-b-2 border-purple-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Sign Up
            </button>
          </div>

          {/* Form Container */}
          <div className="p-6">
            <AnimatePresence mode="wait">
              {activeTab === 'login' ? (
                <motion.form
                  key="login"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  onSubmit={handleLogin}
                  className="space-y-4"
                >
                  <div>
                    <label className={labelClass}>Username</label>
                    <input
                      type="text"
                      value={loginData.username}
                      onChange={(e) => setLoginData({ ...loginData, username: e.target.value })}
                      className={getInputClass('username')}
                      required
                    />
                    {renderFieldError('username')}
                  </div>
                  <div>
                    <label className={labelClass}>Password</label>
                    <input
                      type="password"
                      value={loginData.password}
                      onChange={(e) => setLoginData({ ...loginData, password: e.target.value })}
                      className={getInputClass('password')}
                      required
                    />
                    {renderFieldError('password')}
                  </div>
                  {renderFieldError('non_field_errors')}
                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-md py-2 text-sm font-medium hover:from-purple-700 hover:to-indigo-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50"
                  >
                    {loading ? 'Logging in...' : 'Login'}
                  </button>
                </motion.form>
              ) : (
                <motion.form
                  key="signup"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  onSubmit={handleSignup}
                  className="space-y-4"
                >
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className={labelClass}>First Name</label>
                      <input
                        type="text"
                        value={signupData.first_name}
                        onChange={(e) => setSignupData({ ...signupData, first_name: e.target.value })}
                        className={getInputClass('first_name')}
                        required
                      />
                      {renderFieldError('first_name')}
                    </div>
                    <div>
                      <label className={labelClass}>Last Name</label>
                      <input
                        type="text"
                        value={signupData.last_name}
                        onChange={(e) => setSignupData({ ...signupData, last_name: e.target.value })}
                        className={getInputClass('last_name')}
                        required
                      />
                      {renderFieldError('last_name')}
                    </div>
                  </div>
                  <div>
                    <label className={labelClass}>Username</label>
                    <input
                      type="text"
                      value={signupData.username}
                      onChange={(e) => setSignupData({ ...signupData, username: e.target.value })}
                      className={getInputClass('username')}
                      required
                    />
                    {renderFieldError('username')}
                    <p className="text-xs text-gray-500 mt-1">
                      Only letters, numbers, and @/./+/-/_ characters are allowed
                    </p>
                  </div>
                  <div>
                    <label className={labelClass}>Email</label>
                    <input
                      type="email"
                      value={signupData.email}
                      onChange={(e) => setSignupData({ ...signupData, email: e.target.value })}
                      className={getInputClass('email')}
                      required
                    />
                    {renderFieldError('email')}
                    <p className="text-xs text-gray-500 mt-1">
                      Must be an @allsides.com email address
                    </p>
                  </div>
                  <div>
                    <label className={labelClass}>Password</label>
                    <input
                      type="password"
                      value={signupData.password}
                      onChange={(e) => setSignupData({ ...signupData, password: e.target.value })}
                      className={getInputClass('password')}
                      required
                    />
                    {renderFieldError('password')}
                    <p className="text-xs text-gray-500 mt-1">
                      Must be at least 8 characters long and cannot be entirely numeric
                    </p>
                  </div>
                  <div>
                    <label className={labelClass}>Confirm Password</label>
                    <input
                      type="password"
                      value={signupData.password2}
                      onChange={(e) => setSignupData({ ...signupData, password2: e.target.value })}
                      className={getInputClass('password2')}
                      required
                    />
                    {renderFieldError('password2')}
                  </div>
                  <div>
                    <label className={labelClass}>Bias Rating</label>
                    <select
                      value={signupData.bias_rating}
                      onChange={(e) => setSignupData({ ...signupData, bias_rating: e.target.value })}
                      className={getInputClass('bias_rating')}
                      required
                    >
                      <option value="L">Left</option>
                      <option value="LL">Lean Left</option>
                      <option value="C">Center</option>
                      <option value="LR">Lean Right</option>
                      <option value="R">Right</option>
                    </select>
                    {renderFieldError('bias_rating')}
                  </div>
                  {renderFieldError('non_field_errors')}
                  {errors.detail && (
                    <div className="text-sm text-red-600 mt-1">{errors.detail}</div>
                  )}
                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-md py-2 text-sm font-medium hover:from-purple-700 hover:to-indigo-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50"
                  >
                    {loading ? 'Creating Account...' : 'Sign Up'}
                  </button>
                </motion.form>
              )}
            </AnimatePresence>
          </div>
        </div>
      </motion.div>

      {/* Error Message */}
      <AnimatePresence>
        {errors.non_field_errors && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-4 p-3 rounded-md bg-red-50 text-red-500 text-sm"
          >
            {errors.non_field_errors[0]}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default LoginPage; 