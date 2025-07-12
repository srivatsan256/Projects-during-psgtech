// src/App.tsx
import { useState, useEffect, useRef } from 'react';
import { Zap, User, MessageSquare, ArrowLeft } from 'lucide-react';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import OnboardingForm from './components/OnboardingForm';
import { ChatMessage as ChatMessageType, OnboardingData } from './types';
import { api } from './services/api';
import { Component, ReactNode } from 'react';

// Error Boundary
class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean }> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center">
          <div className="bg-red-100 text-red-700 p-4 rounded-lg">
            Something went wrong. Please refresh the page.
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

const App: React.FC = () => {
  const [username, setUsername] = useState('');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isOnboarded, setIsOnboarded] = useState(false);
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
    return () => clearTimeout(timer);
  }, [messages.length]);

  const isValidUsername = (username: string) => /^[a-zA-Z0-9_-]{3,20}$/.test(username);

  const handleLogin = async () => {
    if (!username.trim()) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          type: 'system',
          content: 'Please enter a username.',
          timestamp: new Date(),
        },
      ]);
      return;
    }

    if (!isValidUsername(username)) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          type: 'system',
          content: 'Username must be 3-20 characters and contain only letters, numbers, underscores, or hyphens.',
          timestamp: new Date(),
        },
      ]);
      return;
    }

    setIsLoading(true);
    try {
      const response = (await api.checkUserStatus(username)) as {
        success: boolean;
        data?: { isOnboarded: boolean };
        error?: string;
      };

      if (!response.success || !response.data) {
        throw new Error(response.error || 'Invalid response from server');
      }

      setIsLoggedIn(true);
      setIsOnboarded(response.data.isOnboarded);

      if (response.data.isOnboarded) {
        setMessages([
          {
            id: Date.now().toString(),
            type: 'bot',
            content: `Welcome back, ${username}! I'm your Inverter Expert assistant. How can I help you today?`,
            timestamp: new Date(),
          },
        ]);
      }
    } catch (error: any) {
      console.error('Login error:', error);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          type: 'system',
          content: error.message || 'Failed to log in. Please try again.',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleOnboardingComplete = async (data: OnboardingData) => {
    setIsLoading(true);
    try {
      const response = (await api.submitOnboarding(username, data)) as {
        success: boolean;
        error?: string;
      };

      if (!response.success) {
        throw new Error(response.error || 'Onboarding failed');
      }

      setIsOnboarded(true);
      setMessages([
        {
          id: Date.now().toString(),
          type: 'system',
          content: '🎉 Setup completed successfully!',
          timestamp: new Date(),
        },
        {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: `Welcome ${data.fullName}! Your ${data.inverterModel} inverter is now registered. I'm here to help with any questions or issues.`,
          timestamp: new Date(),
        },
      ]);
    } catch (error: any) {
      console.error('Onboarding error:', error);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          type: 'system',
          content: error.message || 'Failed to complete onboarding.',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (
    message: string,
    file?: File,
    fileType?: 'image' | 'video'
  ) => {
    if (!message.trim() && !file) return;

    if (file) {
      const maxSize = 10 * 1024 * 1024;
      const allowedTypes = ['image/jpeg', 'image/png', 'video/mp4', 'video/avi', 'video/mov'];

      if (!allowedTypes.includes(file.type)) {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now().toString(),
            type: 'system',
            content: 'Unsupported file type.',
            timestamp: new Date(),
          },
        ]);
        return;
      }

      if (file.size > maxSize) {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now().toString(),
            type: 'system',
            content: 'File size exceeds 10MB.',
            timestamp: new Date(),
          },
        ]);
        return;
      }
    }

    const userMessage: ChatMessageType = {
      id: Date.now().toString(),
      type: 'user',
      content: message || (file ? `Uploaded ${fileType}` : ''),
      timestamp: new Date(),
      mediaType: fileType,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      let fileFn: (() => Promise<string>) | undefined = undefined;
      if (file) {
        fileFn = () =>
          new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = reject;
            reader.readAsDataURL(file);
          });
      }
      const response = (await api.sendMessage(username, message)) as {
        success: boolean;
        data?: { response: string; mediaStatus?: string; weather?: any };
        error?: string;
      };

      if (!response.success || !response.data) {
        throw new Error(response.error || 'Invalid server response');
      }

      const botMessage: ChatMessageType = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: response.data.response,
        timestamp: new Date(),
        mediaStatus: response.data.mediaStatus,
        weather: response.data.weather,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error: any) {
      console.error('Send error:', error);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          type: 'system',
          content: error.message || 'Error sending message.',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setIsOnboarded(false);
    setUsername('');
    setMessages([]);
  };

  // UI Rendering
  if (!isLoggedIn) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-500 via-blue-600 to-teal-600 flex items-center justify-center p-6">
        <div className="w-full max-w-md bg-white rounded-3xl shadow-2xl p-8">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-teal-500 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg">
              <Zap className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-gray-800 mb-2">Inverter Expert</h1>
            <p className="text-gray-600 text-sm">Your AI-powered inverter assistant</p>
          </div>
          <div className="space-y-4">
            <label htmlFor="username" className="block text-sm font-medium text-gray-700">
              Username
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              onKeyPress={(e) => e.key === 'Enter' && handleLogin()}
            />
            <button
              onClick={handleLogin}
              disabled={isLoading || !username.trim()}
              className="w-full bg-blue-500 text-white py-2 rounded-lg hover:bg-blue-600 disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <User size={20} />
              )}
              {isLoading ? 'Checking...' : 'Login'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!isOnboarded) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-500 via-blue-600 to-teal-600">
        <OnboardingForm onComplete={handleOnboardingComplete} onStepChange={() => { }} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <header className="bg-white shadow-sm p-4 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-teal-500 rounded-full flex items-center justify-center">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-800">Inverter Expert</h1>
              <p className="text-xs text-gray-500">AI Assistant</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs flex items-center gap-1">
              <div className="w-2 h-2 bg-green-500 rounded-full" />
              Online
            </span>
            <button
              onClick={handleLogout}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-full"
            >
              <ArrowLeft size={18} />
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto p-4 max-w-4xl mx-auto w-full">
        <div className="space-y-4">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {isLoading && (
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
                <MessageSquare size={16} className="text-gray-600" />
              </div>
              <div className="bg-gray-100 p-3 rounded-lg flex items-center gap-2">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                </div>
                <span className="text-sm text-gray-600">Analyzing...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
    </div>
  );
};

const AppWithErrorBoundary: React.FC = () => (
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);

export default AppWithErrorBoundary;
