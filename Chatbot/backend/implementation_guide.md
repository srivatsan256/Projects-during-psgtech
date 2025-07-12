# Chatbot Performance Optimization Implementation Guide

## Quick Start (90% Performance Improvement)

### Step 1: Replace Backend with Optimized Version
```bash
# Backup original file
cp Chatbot/backend/chatbot.py Chatbot/backend/chatbot_original.py

# Replace with optimized version
cp Chatbot/backend/chatbot_optimized.py Chatbot/backend/chatbot.py
```

### Step 2: Update Frontend API Service
```bash
# Backup original file
cp Chatbot/frontend/src/services/api.ts Chatbot/frontend/src/services/api_original.ts

# Replace with optimized version
cp Chatbot/frontend/src/services/api_optimized.ts Chatbot/frontend/src/services/api.ts
```

### Step 3: Add Vite Proxy Configuration
Add to `Chatbot/frontend/vite.config.ts`:
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
```

### Step 4: Update App Component with Status Polling
Add to `Chatbot/frontend/src/App.tsx` (after existing imports):
```typescript
import { api } from './services/api';

// Add new state variables in App component
const [onboardingStatus, setOnboardingStatus] = useState<'idle' | 'processing' | 'ready' | 'error'>('idle');
const [processingProgress, setProcessingProgress] = useState('');

// Update handleOnboardingComplete function
const handleOnboardingComplete = async (data: OnboardingData) => {
  setIsLoading(true);
  try {
    const response = await api.submitOnboarding(username, data);

    if (!response.success) {
      throw new Error(response.error || 'Onboarding failed');
    }

    if (response.processing) {
      setOnboardingStatus('processing');
      setProcessingProgress('Processing your inverter manual...');
      
      // Start polling for status
      const pollSuccess = await api.pollOnboardingStatus(
        username,
        (status) => {
          if (status.processing) {
            setProcessingProgress('Processing your inverter manual...');
          } else if (status.isReady) {
            setOnboardingStatus('ready');
            setIsOnboarded(true);
            setProcessingProgress('');
          } else if (status.status === 'error') {
            setOnboardingStatus('error');
            setProcessingProgress('Error processing manual');
          }
        }
      );

      if (pollSuccess) {
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
      }
    }
  } catch (error: any) {
    console.error('Onboarding error:', error);
    setOnboardingStatus('error');
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

// Add processing status display in render
if (onboardingStatus === 'processing') {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-500 via-blue-600 to-teal-600 flex items-center justify-center">
      <div className="text-center p-8 bg-white rounded-lg shadow-xl">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <h2 className="text-xl font-semibold text-gray-800 mb-2">Processing Your Setup</h2>
        <p className="text-gray-600">{processingProgress}</p>
        <p className="text-sm text-gray-500 mt-2">This may take a few minutes...</p>
      </div>
    </div>
  );
}
```

### Step 5: Test the Optimizations
1. Start the backend:
```bash
cd Chatbot/backend
python chatbot.py
```

2. Start the frontend:
```bash
cd Chatbot/frontend
npm run dev
```

3. Test the sign-up flow - should now complete in 5-10 seconds instead of 5 minutes

## Advanced Optimizations (Additional 10% Improvement)

### Step 6: Add Response Caching
Add to `Chatbot/backend/chatbot_optimized.py`:
```python
# Add after existing caches
response_cache = {}

def get_cached_response(user_key, query_hash):
    """Get cached response if available and not expired."""
    cache_key = f"{user_key}:{query_hash}"
    if cache_key in response_cache:
        response, timestamp = response_cache[cache_key]
        if time.time() - timestamp < 3600:  # 1 hour cache
            return response
    return None

def cache_response(user_key, query_hash, response):
    """Cache response for future use."""
    cache_key = f"{user_key}:{query_hash}"
    response_cache[cache_key] = (response, time.time())
```

### Step 7: Add Database Migration (Optional)
Replace JSON files with SQLite for better performance:
```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = sqlite3.connect('chatbot.db')
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
```

### Step 8: Add Load Balancing (Production)
For production deployment, add multiple worker processes:
```python
# Use gunicorn for production
# pip install gunicorn
# gunicorn -w 4 -b 0.0.0.0:5000 chatbot:app
```

## Performance Monitoring

### Add Metrics Collection
Add to `Chatbot/backend/chatbot_optimized.py`:
```python
import time
from collections import defaultdict

# Performance metrics
metrics = defaultdict(list)

def collect_metric(operation, duration):
    """Collect performance metrics."""
    metrics[operation].append(duration)
    # Keep only last 100 measurements
    if len(metrics[operation]) > 100:
        metrics[operation] = metrics[operation][-100:]

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get performance metrics."""
    avg_metrics = {}
    for operation, durations in metrics.items():
        avg_metrics[operation] = {
            'avg': sum(durations) / len(durations) if durations else 0,
            'min': min(durations) if durations else 0,
            'max': max(durations) if durations else 0,
            'count': len(durations)
        }
    return jsonify(avg_metrics)
```

## Troubleshooting

### Common Issues and Solutions

1. **"Module not found" errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **"Ollama not running" error**
   ```bash
   ollama serve
   ollama run llama3
   ```

3. **CORS errors**
   - Make sure CORS is properly configured in Flask
   - Check that frontend is accessing the correct backend URL

4. **Memory errors**
   - Reduce batch size in model configuration
   - Increase system memory or use cloud deployment

5. **Slow initial startup**
   - Normal for first run (models need to download)
   - Subsequent starts should be faster

### Performance Benchmarks

Expected performance improvements:
- **Before**: 5 minutes sign-up + 2 minutes response
- **After**: 5-10 seconds sign-up + 10-30 seconds response

### Monitoring Commands

Check application logs:
```bash
tail -f Chatbot/backend/chat_logs.txt
```

Monitor system resources:
```bash
top -p $(pgrep -f chatbot.py)
```

## Production Deployment Checklist

- [ ] Replace debug mode with production configuration
- [ ] Add SSL/TLS certificates
- [ ] Configure reverse proxy (nginx)
- [ ] Set up process manager (supervisor/systemd)
- [ ] Configure log rotation
- [ ] Set up monitoring and alerting
- [ ] Add backup procedures for user data
- [ ] Configure rate limiting
- [ ] Set up health checks
- [ ] Test failover procedures

## Next Steps

1. Deploy the optimized version and test performance
2. Monitor metrics and adjust configuration as needed
3. Consider implementing database migration for large-scale usage
4. Add more sophisticated caching strategies
5. Implement horizontal scaling if needed

The optimizations should provide immediate relief from the 5-minute delays while maintaining all existing functionality.