// src/components/ChatMessage.tsx
import { FC } from 'react';
import { ChatMessage as ChatMessageType } from '../types';
import { MessageSquare } from 'lucide-react';

const ChatMessage: FC<{ message: ChatMessageType }> = ({ message }) => {
  const { type, content, timestamp, mediaType, mediaStatus, weather } = message;

  const formattedTime = timestamp.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });

  const getMessageStyles = () => {
    switch (type) {
      case 'user':
        return 'bg-blue-500 text-white ml-auto rounded-br-none';
      case 'bot':
        return 'bg-gray-200 text-gray-800 mr-auto rounded-bl-none';
      case 'system':
        return 'bg-yellow-100 text-yellow-800 mx-auto rounded-lg text-center';
      default:
        return 'bg-gray-200 text-gray-800';
    }
  };

  return (
    <div
      className={`flex items-start gap-3 max-w-[80%] ${type === 'user' ? 'ml-auto' : 'mr-auto'}`}
      role="log"
      aria-label={`${type} message at ${formattedTime}`}
    >
      {type !== 'system' && (
        <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center flex-shrink-0">
          <MessageSquare size={16} className="text-gray-600" />
        </div>
      )}
      <div className="flex flex-col gap-1">
        <div className={`p-3 rounded-lg shadow-sm ${getMessageStyles()}`}>
          <p className="text-sm">{content}</p>
          {mediaType && (
            <p className="text-xs mt-1 opacity-75">
              Media: {mediaType.charAt(0).toUpperCase() + mediaType.slice(1)}
            </p>
          )}
          {mediaStatus && (
            <p className="text-xs mt-1 opacity-75">Status: {mediaStatus}</p>
          )}
          {weather && (
            <p className="text-xs mt-1 opacity-75">Weather: {weather}</p>
          )}
        </div>
        <span className="text-xs text-gray-500 ml-2">
          {formattedTime}
        </span>
      </div>
    </div>
  );
};

export default ChatMessage;