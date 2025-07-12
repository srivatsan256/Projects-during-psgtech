// src/services/api.ts
// import { OnboardingData } from '../types'; // Removed because OnboardingData is not exported from '../types'
type OnboardingData = {
  // Define the expected structure here, update as needed
  // example:
  // name: string;
  // email: string;
  // age?: number;
};

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export const api = {
  checkUserStatus: async (username: string): Promise<ApiResponse<{ isOnboarded: boolean }>> => {
    const response = await fetch('/api/check_user_status', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username }),
    });
    return response.json();
  },

  submitOnboarding: async (
    username: string,
    data: OnboardingData,
  ): Promise<ApiResponse<{ model: string }>> => {
    const response = await fetch('/api/onboard', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, ...data }),
    });
    return response.json();
  },

  sendMessage: async (
    username: string,
    message: string,
    file?: File,
  ): Promise<ApiResponse<{ response: string; mediaStatus?: string; weather?: string }>> => {
    if (file) {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('query', message);
      formData.append('media', file);
      const response = await fetch('/api/query', {
        method: 'POST',
        body: formData,
      });
      return response.json();
    }
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, query: message }),
    });
    return response.json();
  },
};