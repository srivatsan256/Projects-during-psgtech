// src/types.ts
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
    installedDate: string;
    country: string;
    pinCode: string;
    address: string;
}