import React, { useRef, useState } from 'react';
import { Upload, Image, Video, X, CheckCircle, Camera } from 'lucide-react';

interface FileUploadProps {
  onFileSelect: (file: File, type: 'image' | 'video') => void;
  isUploading: boolean;
}

export default function FileUpload({ onFileSelect, isUploading }: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');
    
    if (isImage || isVideo) {
      setSelectedFile(file);
      onFileSelect(file, isImage ? 'image' : 'video');
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  const openCamera = () => {
    if (inputRef.current) {
      inputRef.current.setAttribute('capture', 'environment');
      inputRef.current.click();
    }
  };

  return (
    <div className="space-y-4">
      <div
        className={`relative border-2 border-dashed rounded-3xl p-6 text-center transition-all ${
          dragActive
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*,video/*"
          onChange={handleChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isUploading}
        />
        
        <div className="space-y-4">
          <div className="mx-auto w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center">
            <Upload className="w-8 h-8 text-gray-400" />
          </div>
          
          <div>
            <p className="text-base font-semibold text-gray-700">
              Upload LED Status Media
            </p>
            <p className="text-sm text-gray-500 mt-1">
              Drag and drop or tap to select
            </p>
          </div>
          
          <div className="flex justify-center gap-6 text-sm text-gray-400">
            <div className="flex items-center gap-2">
              <Image size={18} />
              <span>Images</span>
            </div>
            <div className="flex items-center gap-2">
              <Video size={18} />
              <span>Videos</span>
            </div>
          </div>
        </div>
      </div>

      {/* Camera Button */}
      <button
        onClick={openCamera}
        className="w-full flex items-center justify-center gap-3 p-4 bg-blue-50 text-blue-600 rounded-2xl border-2 border-blue-200 hover:bg-blue-100 transition-colors"
        disabled={isUploading}
      >
        <Camera size={20} />
        <span className="font-medium">Take Photo</span>
      </button>

      {selectedFile && (
        <div className="flex items-center justify-between p-4 bg-gray-50 rounded-2xl border-2 border-gray-200">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
              {selectedFile.type.startsWith('image/') ? (
                <Image size={18} className="text-blue-600" />
              ) : (
                <Video size={18} className="text-blue-600" />
              )}
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-700">{selectedFile.name}</p>
              <p className="text-xs text-gray-500">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            {isUploading && (
              <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            )}
            {!isUploading && (
              <CheckCircle size={18} className="text-green-500" />
            )}
            <button
              onClick={clearFile}
              className="p-2 hover:bg-gray-200 rounded-full transition-colors"
              disabled={isUploading}
            >
              <X size={16} className="text-gray-400" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}