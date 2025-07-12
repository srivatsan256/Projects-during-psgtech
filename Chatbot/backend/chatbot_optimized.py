import os
import re
import json
import shutil
import hashlib
import platform
import time
import warnings
import pytz
import torch
import requests
import numpy as np
import cv2
import logging
import sys
import math
import tempfile
import traceback
from collections import deque
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from flask import Flask, request, jsonify
from flask_cors import CORS
import pycountry
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
import threading

# 1. Suppress Warnings and Configure Logging
os.environ["TORCH_DISTRIBUTED_BACKEND"] = "noop"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.ERROR)

# 2. Global Variables and Paths
BASE_DIR = os.getcwd()
PDF_FOLDER = os.path.join(BASE_DIR, "service_manual")
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
LOG_FILE = os.path.join(BASE_DIR, "chat_logs.txt")
LEARNING_FILE = os.path.join(USER_DATA_DIR, "learning.json")
DIFFICULT_Q_PATH = os.path.join(USER_DATA_DIR, "difficult_questions.jsonl")

# 3. Performance Optimization - Thread Pool and Caching
executor = ThreadPoolExecutor(max_workers=4)
pdf_cache = {}
faiss_cache = {}
learning_data_cache = {"data": None, "timestamp": 0}
processing_status = {}  # Track background processing status

# 4. Timing Decorator
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        log_info(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

# 5. Configuration Management
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
DEFAULT_CONFIG = {
    "chunk_size": 300,
    "video_max_resolution": [640, 480],
    "hsv_ranges": {
        "red1_lower": [0, 70, 50],
        "red1_upper": [10, 255, 255],
        "red2_lower": [160, 70, 50],
        "red2_upper": [180, 255, 255],
        "green_lower": [40, 40, 40],
        "green_upper": [80, 255, 255]
    }
}

def load_config():
    """Load configuration from config.json or create default."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        logging.info(f"Created default config file at {CONFIG_FILE}")
        return DEFAULT_CONFIG
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return DEFAULT_CONFIG

CONFIG = load_config()

# 6. Logging Functions
def log_error(message):
    """Log error messages to console and file with traceback."""
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    platform_info = f"Platform: {platform.platform()}, Python: {sys.version.split()[0]}"
    logging.error(f"[ERROR] {message}")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [ERROR] {message}\n{platform_info}\n{traceback.format_exc()}\n{'-'*60}\n")
    except Exception:
        pass

def log_info(message):
    """Log informational messages to console and file."""
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"[INFO] {message}")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [INFO] {message}\n{'-'*60}\n")
    except Exception:
        pass

# 7. Device and Model Initialization
DEVICE = None
EMBEDDING_MODEL = None
LLM = None

def get_device():
    """Detect and return the best available hardware device."""
    if torch.cuda.is_available():
        log_info("Using CUDA (NVIDIA GPU)")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        log_info("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    else:
        log_info("Using CPU")
        return torch.device("cpu")

def check_ollama_status():
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            log_info("Ollama server is running at http://localhost:11434")
            return True
        log_error(f"Ollama server returned status {response.status_code}: {response.text}")
        return False
    except requests.RequestException as e:
        log_error(f"Ollama check failed: {e}")
        return False

@measure_time
def initialize_models(max_retries=3, delay=2):  # Reduced delay from 5 to 2 seconds
    """Initialize embedding and LLM models with retry logic."""
    global EMBEDDING_MODEL, LLM, DEVICE
    DEVICE = get_device()
    if not check_ollama_status():
        return "Ollama server is not running at http://localhost:11434. Please start it with 'ollama run llama3'."
    for attempt in range(max_retries):
        try:
            EMBEDDING_MODEL = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": str(DEVICE)},
                encode_kwargs={"normalize_embeddings": True}
            )
            LLM = OllamaLLM(model="llama3", base_url="http://localhost:11434")
            log_info("Models initialized successfully.")
            return True
        except Exception as e:
            log_error(f"Attempt {attempt + 1} failed to initialize models: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return f"Failed to initialize models after {max_retries} retries: {str(e)}."

# 8. Utility Functions
def get_safe_username(username):
    """Convert username to a filesystem-safe key."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', username.strip().lower())

def get_user_folder(user_name):
    """Create and return user-specific folder path."""
    try:
        folder = os.path.join(USER_DATA_DIR, get_safe_username(user_name))
        os.makedirs(folder, exist_ok=True)
        if not os.access(folder, os.W_OK):
            log_error(f"User folder {folder} is not writable")
            return None
        return folder
    except Exception as e:
        log_error(f"Error getting user folder for {user_name}: {e}")
        return None

def list_available_models():
    """List available inverter models by scanning PDF_FOLDER."""
    if not os.path.exists(PDF_FOLDER):
        return []
    return [os.path.splitext(f)[0].lower() for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]

# 9. Caching Functions
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

def load_learning_data():
    """Load learning data from LEARNING_FILE."""
    try:
        if os.path.exists(LEARNING_FILE):
            with open(LEARNING_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        log_error(f"Error loading learning data: {e}")
        return {}

def save_learning_data(data):
    """Save learning data to LEARNING_FILE."""
    try:
        os.makedirs(os.path.dirname(LEARNING_FILE), exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(suffix=".json", dir=USER_DATA_DIR)
        with os.fdopen(temp_fd, "w") as f:
            json.dump(data, f, indent=4)
        os.replace(temp_path, LEARNING_FILE)
        log_info(f"Saved learning data to {LEARNING_FILE}")
    except Exception as e:
        log_error(f"Error saving learning data: {e}")

def load_vectorstore(index_path, embedding_model):
    """Load FAISS vectorstore."""
    try:
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        log_error(f"Error loading FAISS index: {e}")
        return None

def validate_index(index_path):
    """Validate FAISS index integrity."""
    index_file = os.path.join(index_path, "index.faiss")
    return os.path.exists(index_file) and os.path.exists(os.path.join(index_path, "index.pkl"))

# 10. Background Processing Functions
def create_faiss_index_background(pdf_path, user_folder):
    """Create FAISS index in background."""
    try:
        log_info(f"Creating FAISS index for {pdf_path}")
        
        # Load PDF content
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=CONFIG["chunk_size"], chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(chunks, EMBEDDING_MODEL)
        
        # Save index
        index_path = os.path.join(user_folder, "faiss_index")
        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)
        
        log_info(f"FAISS index created successfully at {index_path}")
        return True
    except Exception as e:
        log_error(f"Error creating FAISS index: {e}")
        return False

def onboard_user_async(user_name, user_folder, answers):
    """Async version of onboard_user with optimizations."""
    try:
        log_info(f"Starting async onboarding for {user_name}")
        processing_status[user_name] = "processing"
        
        # Save user details immediately
        details_file = os.path.join(user_folder, "details.json")
        with open(details_file, "w") as f:
            json.dump(answers, f, indent=4)
        
        # Create basic user data structure
        learning_data = get_learning_data_cached()
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
            success = create_faiss_index_background(pdf_path, user_folder)
            if success:
                processing_status[user_name] = "ready"
            else:
                processing_status[user_name] = "error"
        else:
            processing_status[user_name] = "error"
            
        log_info(f"Background onboarding completed for {user_name}")
        return True
    except Exception as e:
        log_error(f"Background onboarding failed: {e}")
        processing_status[user_name] = "error"
        return False

def update_user_interaction_background(user_name, query, response):
    """Update user interaction data in background."""
    try:
        learning_data = get_learning_data_cached()
        user_key = get_safe_username(user_name)
        
        if user_key in learning_data:
            learning_data[user_key]["chat_history"] += f"\nQ: {query}\nA: {response}\n"
            save_learning_data(learning_data)
    except Exception as e:
        log_error(f"Error updating user interaction: {e}")

# 11. Optimized Core Functions
@measure_time
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
            # Check if still processing
            status = processing_status.get(user_name, "unknown")
            if status == "processing":
                return ("I'm still processing your inverter manual. " +
                       "Please try again in a few moments for detailed technical answers."), "processing"
            elif status == "error":
                return ("There was an error processing your manual. " +
                       "Please contact support."), "error"
            else:
                # Try to create index in background
                pdf_path = os.path.join(PDF_FOLDER, f"{model_name}.pdf")
                if os.path.exists(pdf_path):
                    executor.submit(create_faiss_index_background, pdf_path, user_folder)
                    processing_status[user_name] = "processing"
                    return ("I'm processing your inverter manual. " +
                           "Please try again in a few moments."), "processing"
                else:
                    return f"No manual found for model '{model_name}'.", "error"
        
        # Quick document retrieval
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Reduced from 5 to 3
        docs = retriever.get_relevant_documents(query)
        context = "\n---\n".join(doc.page_content for doc in docs[:3])  # Limit context
        
        # Optimized prompt
        media_info = f"Inverter LED Status: {media_status}\n" if media_status else ""
        prompt = f"""
        You are Inverter Expert for {model_name}.
        {media_info}Context: {context[:2000]}
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

# 12. Flask Server and API Endpoints
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

@app.route('/check_user_status', methods=['POST'])
def check_user_status():
    """Check if a user is onboarded."""
    try:
        data = request.json
        if not data or not isinstance(data.get('username'), str) or not data['username'].strip():
            return jsonify({"success": False, "error": "Valid username is required"}), 400
        
        user_name = data['username']
        user_folder = get_user_folder(user_name)
        if not user_folder:
            return jsonify({"success": False, "error": "Unable to access user folder"}), 400
        
        is_onboarded = os.path.exists(os.path.join(user_folder, "details.json"))
        return jsonify({"success": True, "data": {"isOnboarded": is_onboarded}})
    except Exception as e:
        log_error(f"Error checking user status: {e}")
        return jsonify({"success": False, "error": "Server error"}), 500

@app.route('/onboard', methods=['POST'])
def onboard():
    """Handle user onboarding with async processing."""
    try:
        data = request.json
        if not data or not isinstance(data.get('username'), str) or not data['username'].strip():
            return jsonify({"success": False, "error": "Valid username is required"}), 400
        
        user_name = data['username']
        user_folder = get_user_folder(user_name)
        if not user_folder:
            return jsonify({"success": False, "error": "Unable to create user folder"}), 400
        
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
        
        # Check if model exists
        if answers["inverter_model"].lower() not in list_available_models():
            return jsonify({"success": False, "error": f"Model '{answers['inverter_model']}' not found"}), 400
        
        # Submit to background processing
        executor.submit(onboard_user_async, user_name, user_folder, answers)
        
        # Return immediately with processing status
        return jsonify({
            "success": True, 
            "message": "Onboarding initiated. PDF processing in background.",
            "processing": True
        })
    except Exception as e:
        log_error(f"Onboarding error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/onboard_status', methods=['POST'])
def check_onboard_status():
    """Check onboarding processing status."""
    try:
        data = request.json
        user_name = data['username']
        user_folder = get_user_folder(user_name)
        
        # Check processing status
        status = processing_status.get(user_name, "unknown")
        
        # Check if FAISS index is ready
        index_path = os.path.join(user_folder, "faiss_index")
        is_ready = validate_index(index_path)
        
        if is_ready:
            processing_status[user_name] = "ready"
            status = "ready"
        
        return jsonify({
            "success": True,
            "data": {
                "isReady": is_ready,
                "processing": status == "processing",
                "status": status
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/query', methods=['POST'])
def handle_query():
    """Handle user queries with optimization."""
    try:
        data = request.json
        if not data or not isinstance(data.get('username'), str) or not data['username'].strip():
            return jsonify({"success": False, "error": "Valid username is required"}), 400
        if not isinstance(data.get('query'), str) or not data['query'].strip():
            return jsonify({"success": False, "error": "Valid query is required"}), 400
        
        username = data['username']
        query = data['query']
        user_folder = get_user_folder(username)
        
        if not user_folder:
            return jsonify({"success": False, "error": "User folder not found"}), 400
        
        # Use optimized answer generation
        answer, source_type = generate_answer_optimized(LLM, EMBEDDING_MODEL, username, user_folder, query)
        
        return jsonify({
            "success": True,
            "data": {
                "response": str(answer),
                "source": source_type
            }
        })
    except Exception as e:
        log_error(f"Query error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# 13. Directory Initialization
def initialize_directories():
    """Create necessary directories and initialize log file."""
    try:
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        os.makedirs(PDF_FOLDER, exist_ok=True)
        log_info(f"Created directories: {USER_DATA_DIR}, {PDF_FOLDER}")
    except Exception as e:
        log_error(f"Failed to initialize directories: {e}")
        raise RuntimeError(f"Directory initialization failed: {e}")

if __name__ == "__main__":
    try:
        initialize_directories()
        model_init_result = initialize_models()
        if model_init_result is not True:
            log_error(f"Startup error: {model_init_result}")
            sys.exit(1)
        
        log_info("Starting optimized chatbot server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        log_error(f"Main execution failed: {e}")
        sys.exit(1)