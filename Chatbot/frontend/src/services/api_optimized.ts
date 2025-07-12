// src/services/api_optimized.ts
import { OnboardingData } from '../types';

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  processing?: boolean;
}

// Timeout wrapper for fetch requests
const fetchWithTimeout = async (url: string, options: RequestInit & { timeout?: number } = {}) => {
  const { timeout = 60000, ...fetchOptions } = options;
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error && typeof error === 'object' && (error as any).name === 'AbortError') {
      throw new Error('Request timed out');
    }
    throw error;
  }
};

// Base API URL - can be configured for different environments
const BASE_URL = '';

export const api = {
  // Check user status with timeout
  checkUserStatus: async (username: string): Promise<ApiResponse<{ isOnboarded: boolean }>> => {
    try {
      const response = await fetchWithTimeout(`${BASE_URL}/api/check_user_status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username }),
        timeout: 10000, // 10 second timeout
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Error checking user status:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  },

  // Submit onboarding with immediate response
  submitOnboarding: async (
    username: string,
    data: OnboardingData,
  ): Promise<ApiResponse<{ model: string; processing?: boolean }>> => {
    try {
      const response = await fetchWithTimeout(`${BASE_URL}/api/onboard`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, ...data }),
        timeout: 15000, // 15 second timeout for quick validation
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Error submitting onboarding:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to submit onboarding',
      };
    }
  },

  // Check onboarding status for polling
  checkOnboardStatus: async (username: string): Promise<ApiResponse<{ 
    isReady: boolean; 
    processing: boolean; 
    status: string 
  }>> => {
    try {
      const response = await fetchWithTimeout(`${BASE_URL}/api/onboard_status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username }),
        timeout: 5000, // 5 second timeout
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Error checking onboard status:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to check status',
      };
    }
  },

  // Send message with optimized timeout handling
  sendMessage: async (
    username: string,
    message: string,
    file?: File,
  ): Promise<ApiResponse<{ response: string; source?: string; mediaStatus?: string }>> => {
    try {
      let response;
      
      if (file) {
        // File upload with longer timeout
        const formData = new FormData();
        formData.append('username', username);
        formData.append('query', message);
        formData.append('media', file);
        
        response = await fetchWithTimeout(`${BASE_URL}/api/query`, {
          method: 'POST',
          body: formData,
          timeout: 120000, // 2 minute timeout for file processing
        });
      } else {
        // Text query with shorter timeout
        response = await fetchWithTimeout(`${BASE_URL}/api/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, query: message }),
          timeout: 60000, // 1 minute timeout
        });
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Error sending message:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to send message',
      };
    }
  },

  // Utility function to poll onboarding status
  pollOnboardingStatus: async (
    username: string,
    onUpdate: (status: { isReady: boolean; processing: boolean; status: string }) => void,
    maxAttempts: number = 60, // 5 minutes at 5-second intervals
    interval: number = 5000, // 5 seconds
  ): Promise<boolean> => {
    let attempts = 0;
    
    const poll = async (): Promise<boolean> => {
      try {
        const response = await api.checkOnboardStatus(username);
        
        if (response.success && response.data) {
          onUpdate(response.data);
          
          if (response.data.isReady) {
            return true; // Processing complete
          }
          
          if (response.data.status === 'error') {
            throw new Error('Processing failed');
          }
        }
        
        attempts++;
        if (attempts >= maxAttempts) {
          throw new Error('Polling timeout - processing took too long');
        }
        
        // Continue polling
        await new Promise(resolve => setTimeout(resolve, interval));
        return poll();
      } catch (error) {
        console.error('Error polling onboarding status:', error);
        onUpdate({ isReady: false, processing: false, status: 'error' });
        return false;
      }
    };
    
    return poll();
  },

  // Health check endpoint
  healthCheck: async (): Promise<boolean> => {
    try {
      const response = await fetchWithTimeout(`${BASE_URL}/api/health`, {
        method: 'GET',
        timeout: 5000,
      });
      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  },

  // Get available models
  getAvailableModels: async (): Promise<ApiResponse<{ models: string[] }>> => {
    try {
      const response = await fetchWithTimeout(`${BASE_URL}/api/models`, {
        method: 'GET',
        timeout: 10000,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Error getting available models:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get models',
      };
    }
  },
};

// Export utility functions for error handling
export const ApiError = {
  isTimeout: (error: any): boolean => {
    return error?.message?.includes('timeout') || error?.name === 'AbortError';
  },
  
  isNetworkError: (error: any): boolean => {
    return error?.message?.includes('Failed to fetch') || error?.code === 'NETWORK_ERROR';
  },
  
  isServerError: (error: any): boolean => {
    return error?.message?.includes('HTTP 5') || error?.status >= 500;
  },
};

// Export types for better type safety
export type { ApiResponse };