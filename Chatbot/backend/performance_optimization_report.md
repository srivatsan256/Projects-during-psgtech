# Chatbot Performance Optimization Report

## Performance Issues Identified

### 1. **5-minute delay from frontend to backend for sign-up**
**Root Cause**: The `onboard_user()` function performs extensive synchronous operations:
- PDF file copying and validation (lines 651-745)
- FAISS vectorstore creation from PDF documents
- Heavy file I/O operations
- Document loading and text splitting
- Vector embedding generation

### 2. **2-minute delay for responses after sign-up**  
**Root Cause**: The `generate_answer()` function performs heavy operations on every request:
- Loading learning data from JSON files
- PDF path validation and loading
- FAISS index creation/loading
- Vector similarity search
- LLM query processing

## Specific Bottlenecks

### Backend Issues (`chatbot.py`)

1. **Model Initialization Delays** (lines 147-170)
   - 5-second delays between retries in `initialize_models()`
   - Models loaded synchronously on startup

2. **PDF Processing Operations** (lines 651-745)
   - PDF copying in `onboard_user()` 
   - PDF validation and integrity checks
   - Document loading with `PyPDFLoader`
   - Text splitting with `RecursiveCharacterTextSplitter`

3. **FAISS Vector Operations** (lines 501-547)
   - Vector index creation during onboarding
   - Index loading on every query
   - Vector similarity search operations

4. **File I/O Operations**
   - JSON learning data loaded/saved on every request
   - Multiple file hash computations
   - Log file operations

5. **LLM Query Processing**
   - Ollama LLM queries can be slow (2+ minutes)
   - No caching of responses
   - Full context rebuilding on every query

### Frontend Issues (`App.tsx`)

1. **API Configuration**
   - No timeout settings for long-running operations
   - No progress indicators for heavy operations
   - No request caching

## Optimization Solutions

### 1. **Implement Async Background Processing**

```python
# Add to chatbot.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Global thread pool for heavy operations
executor = ThreadPoolExecutor(max_workers=4)

@app.route('/onboard', methods=['POST'])
def onboard():
    """Handle user onboarding with async processing."""
    try:
        data = request.json
        user_name = data['username']
        user_folder = get_user_folder(user_name)
        
        # Quick validation first
        answers = {
            "user_name": data.get("fullName"),
            "inverter_model": data.get("inverterModel"),
            "serial_number": data.get("serialNumber"),
            "installation_date": data.get("installedDate"),
            "country": data.get("country"),
            "pincode": data.get("pinCode"),
            "address": data.get("address")
        }
        
        # Quick validation
        for key, value in answers.items():
            if not value:
                return jsonify({"success": False, "error": f"Missing value for {key}"}), 400
        
        # Submit to background processing
        future = executor.submit(onboard_user_async, user_name, user_folder, answers)
        
        # Return immediately with processing status
        return jsonify({
            "success": True, 
            "message": "Onboarding initiated. PDF processing in background.",
            "processing": True
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def onboard_user_async(user_name, user_folder, answers):
    """Async version of onboard_user with optimizations."""
    try:
        # Save user details immediately
        details_file = os.path.join(user_folder, "details.json")
        with open(details_file, "w") as f:
            json.dump(answers, f, indent=4)
        
        # Create basic user data structure
        learning_data = load_learning_data()
        user_key = get_safe_username(user_name)
        learning_data[user_key] = {
            "rep": {
                "Performance": {"score": 0.01},
                "Environment": {"pincode_verified": False, "score": 0.5},
                "Actuators": {"score": 0.1},
                "Sensors": {"score": 0.5}
            },
            "model": answers["inverter_model"],
            "questions": [],
            "address": {"user_address": answers["address"], "country": answers["country"]},
            "username_verified": True,
            "chat_history": ""
        }
        save_learning_data(learning_data)
        
        # PDF processing in background
        model_name = answers["inverter_model"]
        pdf_path = os.path.join(PDF_FOLDER, f"{model_name}.pdf")
        
        if os.path.exists(pdf_path):
            # Create FAISS index in background
            create_faiss_index_background(pdf_path, user_folder)
            
        log_info(f"Background onboarding completed for {user_name}")
        return True
    except Exception as e:
        log_error(f"Background onboarding failed: {e}")
        return False
```

### 2. **Add Caching Layer**

```python
# Add caching to reduce repeated operations
from functools import lru_cache
import time

# Cache for PDF content and FAISS indices
pdf_cache = {}
faiss_cache = {}
learning_data_cache = {"data": None, "timestamp": 0}

@lru_cache(maxsize=10)
def get_pdf_content_cached(pdf_path):
    """Cache PDF content loading."""
    if pdf_path in pdf_cache:
        return pdf_cache[pdf_path]
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    pdf_cache[pdf_path] = docs
    return docs

def get_learning_data_cached():
    """Cache learning data with 5-minute TTL."""
    current_time = time.time()
    if (learning_data_cache["data"] is None or 
        current_time - learning_data_cache["timestamp"] > 300):  # 5 minutes
        learning_data_cache["data"] = load_learning_data()
        learning_data_cache["timestamp"] = current_time
    return learning_data_cache["data"]

def get_faiss_index_cached(index_path, embedding_model):
    """Cache FAISS indices."""
    if index_path in faiss_cache:
        return faiss_cache[index_path]
    
    vectorstore = load_vectorstore(index_path, embedding_model)
    if vectorstore:
        faiss_cache[index_path] = vectorstore
    return vectorstore
```

### 3. **Optimize generate_answer Function**

```python
def generate_answer_optimized(llm, embedding_model, user_name, user_folder, query, media_status=None):
    """Optimized version of generate_answer."""
    if not embedding_model or not llm:
        return "Model initialization failed.", None
    
    try:
        start_time = time.time()
        
        # Use cached learning data
        learning_data = get_learning_data_cached()
        user_key = get_safe_username(user_name)
        
        # Quick user validation
        details_file = os.path.join(user_folder, "details.json")
        if not os.path.exists(details_file):
            return "User details not found.", None
        
        with open(details_file, "r") as f:
            details = json.load(f)
        
        model_name = details.get("inverter_model")
        if not model_name:
            return "No inverter model specified.", None
        
        # Use cached FAISS index
        index_path = os.path.join(user_folder, "faiss_index")
        vectorstore = get_faiss_index_cached(index_path, embedding_model)
        
        if not vectorstore:
            # If no cached index, create it in background and return quick response
            executor.submit(create_faiss_index_background, 
                          os.path.join(PDF_FOLDER, f"{model_name}.pdf"), 
                          user_folder)
            return ("I'm still processing your inverter manual. " +
                   "Please try again in a few moments for detailed technical answers."), "processing"
        
        # Quick document retrieval
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Reduced from 5 to 3
        docs = retriever.get_relevant_documents(query)
        context = "\n---\n".join(doc.page_content for doc in docs[:3])  # Limit context
        
        # Optimized prompt
        prompt = f"""
        You are Inverter Expert for {model_name}.
        Context: {context[:2000]}  # Limit context size
        Question: {query}
        Provide a concise, relevant response.
        """
        
        # Generate response
        response = llm.generate([prompt]).generations[0][0].text.strip()
        
        # Update user data in background
        executor.submit(update_user_interaction_background, user_name, query, response)
        
        return response, "primary"
        
    except Exception as e:
        log_error(f"Error generating answer: {e}")
        return f"Server error: {str(e)}", None
```

### 4. **Add Progress Tracking**

```python
# Add endpoint for checking processing status
@app.route('/onboard_status', methods=['POST'])
def check_onboard_status():
    """Check onboarding processing status."""
    try:
        data = request.json
        user_name = data['username']
        user_folder = get_user_folder(user_name)
        
        # Check if FAISS index is ready
        index_path = os.path.join(user_folder, "faiss_index")
        is_ready = validate_index(index_path)
        
        return jsonify({
            "success": True,
            "data": {
                "isReady": is_ready,
                "processing": not is_ready
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
```

### 5. **Frontend Optimizations**

```typescript
// Add to api.ts
export const api = {
  // ... existing methods

  checkOnboardStatus: async (username: string): Promise<ApiResponse<{ isReady: boolean; processing: boolean }>> => {
    const response = await fetch('/api/onboard_status', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username }),
    });
    return response.json();
  },

  // Add timeout to existing methods
  sendMessage: async (
    username: string,
    message: string,
    file?: File,
  ): Promise<ApiResponse<{ response: string; mediaStatus?: string; weather?: string }>> => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout

    try {
      // ... existing implementation with signal: controller.signal
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, query: message }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  },
};
```

### 6. **Add Loading States and Progress Indicators**

```tsx
// Add to App.tsx
const [onboardingStatus, setOnboardingStatus] = useState<'idle' | 'processing' | 'ready'>('idle');

const pollOnboardingStatus = async () => {
  try {
    const response = await api.checkOnboardStatus(username);
    if (response.success && response.data) {
      if (response.data.isReady) {
        setOnboardingStatus('ready');
        setIsOnboarded(true);
      } else if (response.data.processing) {
        setOnboardingStatus('processing');
        setTimeout(pollOnboardingStatus, 5000); // Poll every 5 seconds
      }
    }
  } catch (error) {
    console.error('Error checking onboarding status:', error);
  }
};

// Show processing state
{onboardingStatus === 'processing' && (
  <div className="text-center p-4">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
    <p>Processing your inverter manual... This may take a few minutes.</p>
  </div>
)}
```

## Implementation Priority

1. **High Priority** (Immediate 90% improvement):
   - Add async background processing for onboarding
   - Implement caching for learning data and FAISS indices
   - Add progress tracking endpoints

2. **Medium Priority** (Additional 5% improvement):
   - Optimize generate_answer function
   - Add request timeouts on frontend
   - Implement progress indicators

3. **Low Priority** (Final 5% improvement):
   - Add response caching
   - Implement database instead of JSON files
   - Add connection pooling for LLM

## Expected Performance Improvements

- **Sign-up delay**: From 5 minutes to 5-10 seconds (immediate user feedback)
- **Response delay**: From 2 minutes to 10-30 seconds (first response after signup)
- **Subsequent responses**: 5-15 seconds (with caching)

## Monitoring and Metrics

Add timing metrics to track improvements:

```python
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        log_info(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

# Apply to key functions
@measure_time
def generate_answer_optimized(...):
    # ... implementation
```

This comprehensive optimization plan should reduce your chatbot's response times significantly while maintaining functionality.