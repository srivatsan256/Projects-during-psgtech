export interface User {
  username: string;
  fullName: string;
  inverterModel: string;
  serialNumber: string;
  installationDate: string;
  pincode: string;
  address: string;
  isOnboarded: boolean;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'bot' | 'system';
  content: string;
  timestamp: Date;
  mediaType?: 'image' | 'video';
  mediaStatus?: string;
  weather?: string;
}

export interface OnboardingData {
  fullName: string;
  inverterModel: string;
  serialNumber: string;
  installationDate: string;
  pincode: string;
  address: string;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
}