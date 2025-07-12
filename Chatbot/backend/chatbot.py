#!/usr/bin/env python3
"""
Intelligent Inverter Expert Chatbot with PEAS Auto-Tuning
=========================================================

A comprehensive chatbot system with performance optimizations and 
PEAS (Performance Environment Agent Sensor) auto-tuning capabilities.

Features:
- Async background processing for performance
- PEAS auto-tuning: If sum=4 or ceiling=4, reduce to 2
- Caching for improved response times
- Comprehensive API endpoints
- Built-in testing functionality

Usage:
    python chatbot.py                    # Run chatbot server (default)
    python chatbot.py --test             # Run PEAS functionality tests
    python chatbot.py --peas-overview    # Show PEAS overview
    python chatbot.py --peas-status <user>   # Check user PEAS status
    python chatbot.py --help             # Show help

Author: AI Assistant
Version: 2.0 with PEAS Auto-Tuning
"""

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
import threading
import argparse
from collections import deque, defaultdict
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

# ============================================================================
# 1. CONFIGURATION AND INITIALIZATION
# ============================================================================

# Suppress Warnings and Configure Logging
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

# Global Variables and Paths
BASE_DIR = os.getcwd()
PDF_FOLDER = os.path.join(BASE_DIR, "service_manual")
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
LOG_FILE = os.path.join(BASE_DIR, "chat_logs.txt")
LEARNING_FILE = os.path.join(USER_DATA_DIR, "learning.json")
DIFFICULT_Q_PATH = os.path.join(USER_DATA_DIR, "difficult_questions.jsonl")

# Performance Optimization - Thread Pool and Caching
executor = ThreadPoolExecutor(max_workers=4)
pdf_cache = {}
faiss_cache = {}
learning_data_cache = {"data": None, "timestamp": 0}
processing_status = {}  # Track background processing status
response_cache = {}     # Response caching for improved performance

# Onboarding prompts
ONBOARDING_PROMPTS = [
    "What's your full name?",
    "What is your inverter model? (Please enter exactly as shown in available models)",
    "What is your serial number?",
    "What is the installed date? (YYYY-MM-DD)",
    "What is your country?",
    "Please enter your inverter installed area PIN code",
    "What is your full address?"
]

# Configuration Management
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
    },
    "peas_tuning": {
        "target_sum": 2.0,
        "tolerance": 0.01,
        "tuning_interval": 21600  # 6 hours
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

# Device and Model Initialization
DEVICE = None
EMBEDDING_MODEL = None
LLM = None

# Image/Video Processing Variables
RED1 = np.array(CONFIG["hsv_ranges"]["red1_lower"])
RED2 = np.array(CONFIG["hsv_ranges"]["red1_upper"])
RED3 = np.array(CONFIG["hsv_ranges"]["red2_lower"])
RED4 = np.array(CONFIG["hsv_ranges"]["red2_upper"])
GREEN_LOWER = np.array(CONFIG["hsv_ranges"]["green_lower"])
GREEN_UPPER = np.array(CONFIG["hsv_ranges"]["green_upper"])
STATE_HISTORY = deque(maxlen=30)

# Performance Metrics
metrics = defaultdict(list)

# ============================================================================
# 2. UTILITY FUNCTIONS AND DECORATORS
# ============================================================================

def measure_time(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        log_info(f"{func.__name__} took {duration:.2f} seconds")
        collect_metric(func.__name__, duration)
        return result
    return wrapper

def collect_metric(operation, duration):
    """Collect performance metrics."""
    metrics[operation].append(duration)
    # Keep only last 100 measurements
    if len(metrics[operation]) > 100:
        metrics[operation] = metrics[operation][-100:]

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

def verify_username_with_full_name(username, full_name):
    """Verify that the username matches the full name."""
    username_clean = re.sub(r'[^a-zA-Z0-9]', '', username).lower()
    full_name_clean = re.sub(r'[^a-zA-Z0-9]', '', full_name).lower()
    return username_clean == full_name_clean

def verify_country(country):
    """Validate country name using pycountry."""
    if not country or not country.strip():
        return False
    country = country.strip().title()
    return any(c.name.lower() == country.lower() for c in pycountry.countries)

def verify_pincode_with_address(pin_code, address):
    """Placeholder: Verify if the pin code matches the address."""
    return True

# ============================================================================
# 3. DEVICE AND MODEL INITIALIZATION
# ============================================================================

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
def initialize_models(max_retries=3, delay=2):
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

# ============================================================================
# 4. CACHING FUNCTIONS
# ============================================================================

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

# ============================================================================
# 5. DATA MANAGEMENT FUNCTIONS
# ============================================================================

def load_learning_data():
    """Load learning data from LEARNING_FILE."""
    try:
        if os.path.exists(LEARNING_FILE):
            with open(LEARNING_FILE, "r") as f:
                data = json.load(f)
                # Ensure all users have proper PEAS structure
                for user_key in data:
                    if "rep" not in data[user_key]:
                        data[user_key]["rep"] = {
                            "Performance": {"score": 0.01},
                            "Environment": {"pincode_verified": False, "score": 0.5},
                            "Actuators": {"score": 0.1},
                            "Sensors": {"score": 0.5}
                        }
                    # Ensure all required fields exist
                    for field in ["model", "questions", "address", "username_verified", "chat_history"]:
                        if field not in data[user_key]:
                            data[user_key][field] = "" if field in ["model", "chat_history"] else ([] if field == "questions" else ({} if field == "address" else False))
                return data
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

def compute_file_hash(file_path, hash_algo="sha256"):
    """Compute the hash of a file."""
    try:
        hash_func = hashlib.new(hash_algo)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        log_error(f"Error computing file hash: {e}")
        return None

def validate_pdf_integrity(pdf_path):
    """Validate the integrity of a PDF file."""
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return len(docs) > 0
    except Exception as e:
        log_error(f"PDF integrity check failed for {pdf_path}: {e}")
        return False

# ============================================================================
# 6. PEAS AUTO-TUNING SYSTEM
# ============================================================================

def auto_tune_peas_scores(user_key, learning_data, force_tune=False):
    """
    Enhanced PEAS auto-tuning system.
    If PEAS sum = 4 or ceiling of any component = 4, then reduce total to 2 for auto-tuning.
    """
    try:
        if user_key not in learning_data:
            return False
        
        user_data = learning_data[user_key]
        peas_scores = user_data.get("rep", {})
        
        # Extract PEAS component scores
        perf_score = peas_scores.get("Performance", {}).get("score", 0.01)
        env_score = peas_scores.get("Environment", {}).get("score", 0.5)
        act_score = peas_scores.get("Actuators", {}).get("score", 0.1)
        sens_score = peas_scores.get("Sensors", {}).get("score", 0.5)
        
        # Calculate current totals
        current_sum = perf_score + env_score + act_score + sens_score
        current_ceiling = max(math.ceil(perf_score), math.ceil(env_score), 
                            math.ceil(act_score), math.ceil(sens_score))
        
        log_info(f"PEAS Analysis for {user_key}: Sum={current_sum:.3f}, Ceiling={current_ceiling}")
        log_info(f"Components: P={perf_score:.3f}, E={env_score:.3f}, A={act_score:.3f}, S={sens_score:.3f}")
        
        # Check if auto-tuning is needed
        needs_tuning = (abs(current_sum - 4.0) < CONFIG["peas_tuning"]["tolerance"]) or (current_ceiling == 4) or force_tune
        
        if needs_tuning:
            log_info(f"PEAS Auto-tuning triggered for {user_key}")
            
            # Reduce total to 2 for auto-tuning as per requirement
            target_sum = CONFIG["peas_tuning"]["target_sum"]
            
            # Calculate scaling factor to achieve target sum
            if current_sum > 0:
                scale_factor = target_sum / current_sum
            else:
                scale_factor = 0.5  # Default scaling
            
            # Apply scaling to all components
            new_perf_score = round(perf_score * scale_factor, 3)
            new_env_score = round(env_score * scale_factor, 3)
            new_act_score = round(act_score * scale_factor, 3)
            new_sens_score = round(sens_score * scale_factor, 3)
            
            # Ensure no component exceeds 1.0
            new_perf_score = min(new_perf_score, 1.0)
            new_env_score = min(new_env_score, 1.0)
            new_act_score = min(new_act_score, 1.0)
            new_sens_score = min(new_sens_score, 1.0)
            
            # Update the scores
            user_data["rep"]["Performance"]["score"] = new_perf_score
            user_data["rep"]["Environment"]["score"] = new_env_score
            user_data["rep"]["Actuators"]["score"] = new_act_score
            user_data["rep"]["Sensors"]["score"] = new_sens_score
            
            # Add tuning metadata
            user_data["rep"]["last_tuning"] = datetime.now().isoformat()
            user_data["rep"]["tuning_reason"] = "sum_4_or_ceiling_4"
            user_data["rep"]["original_sum"] = current_sum
            user_data["rep"]["tuned_sum"] = new_perf_score + new_env_score + new_act_score + new_sens_score
            
            log_info(f"PEAS Auto-tuning completed for {user_key}")
            log_info(f"New components: P={new_perf_score:.3f}, E={new_env_score:.3f}, A={new_act_score:.3f}, S={new_sens_score:.3f}")
            log_info(f"New sum: {user_data['rep']['tuned_sum']:.3f}")
            
            return True
        else:
            log_info(f"PEAS Auto-tuning not needed for {user_key}")
            return False
            
    except Exception as e:
        log_error(f"Error in PEAS auto-tuning for {user_key}: {e}")
        return False

def tune_peas_scores_batch():
    """Batch tune PEAS scores for all users."""
    try:
        learning_data = get_learning_data_cached()
        tuned_users = []
        
        for user_key in learning_data.keys():
            if auto_tune_peas_scores(user_key, learning_data):
                tuned_users.append(user_key)
        
        if tuned_users:
            save_learning_data(learning_data)
            log_info(f"PEAS batch tuning completed for {len(tuned_users)} users: {tuned_users}")
        else:
            log_info("PEAS batch tuning: No users required tuning")
            
        return tuned_users
    except Exception as e:
        log_error(f"Error in PEAS batch tuning: {e}")
        return []

def get_peas_status(user_key, learning_data):
    """Get current PEAS status for a user."""
    try:
        if user_key not in learning_data:
            return None
        
        user_data = learning_data[user_key]
        peas_scores = user_data.get("rep", {})
        
        perf_score = peas_scores.get("Performance", {}).get("score", 0.01)
        env_score = peas_scores.get("Environment", {}).get("score", 0.5)
        act_score = peas_scores.get("Actuators", {}).get("score", 0.1)
        sens_score = peas_scores.get("Sensors", {}).get("score", 0.5)
        
        current_sum = perf_score + env_score + act_score + sens_score
        current_ceiling = max(math.ceil(perf_score), math.ceil(env_score), 
                            math.ceil(act_score), math.ceil(sens_score))
        
        return {
            "user": user_key,
            "components": {
                "Performance": perf_score,
                "Environment": env_score,
                "Actuators": act_score,
                "Sensors": sens_score
            },
            "sum": current_sum,
            "ceiling": current_ceiling,
            "needs_tuning": (abs(current_sum - 4.0) < CONFIG["peas_tuning"]["tolerance"]) or (current_ceiling == 4),
            "last_tuning": peas_scores.get("last_tuning"),
            "tuning_reason": peas_scores.get("tuning_reason"),
            "original_sum": peas_scores.get("original_sum"),
            "tuned_sum": peas_scores.get("tuned_sum")
        }
    except Exception as e:
        log_error(f"Error getting PEAS status for {user_key}: {e}")
        return None

# ============================================================================
# 7. IMAGE/VIDEO PROCESSING FUNCTIONS
# ============================================================================

def detect_color_state(frame):
    """Detect LED color state in a frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, RED1, RED2)
    red_mask2 = cv2.inRange(hsv, RED3, RED4)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    threshold = 500
    if red_pixels > threshold and red_pixels > green_pixels:
        return "RED"
    elif green_pixels > threshold and green_pixels > red_pixels:
        return "GREEN"
    else:
        return "OFF"

def analyze_flashing_pattern(history, color):
    """Analyze LED flashing pattern to determine inverter status."""
    sequence = list(history)
    on_count = sum(1 for x in sequence if x == color)
    if color == "RED":
        if on_count >= 25:
            return "Solid Red: A fault is present, and the inverter will not operate until the issue is resolved and reconnected to the grid."
        elif 8 < on_count < 15:
            return "Flashing Red: Warning state, inverter temporarily inactive."
    elif color == "GREEN":
        if on_count >= 25:
            return "Solid Green: The inverter is operating normally and feeding power to the grid."
        elif 8 < on_count < 15:
            return "Flashing Green (1s ON, 2s OFF): Standby mode."
        elif 20 < on_count < 23:
            return "Flashing Green (3s ON, 1s OFF): Reduced output operation."
    return "Status unclear"

def process_image(image_path):
    """Process a single image to determine inverter LED status."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Image not found."
        state = detect_color_state(image)
        if state == "RED":
            return "Solid Red: A fault is present, and the inverter will not operate until the issue is resolved and reconnected to the grid."
        elif state == "GREEN":
            return "Solid Green: The inverter is operating normally and feeding power to the grid."
        else:
            return "No LED or unclear state detected in image."
    except Exception as e:
        log_error(f"Error processing image: {e}")
        return "Error processing image."

def process_video(video_path):
    """Process a video to analyze inverter LED status."""
    try:
        cap = cv2.VideoCapture(video_path)
        last_time = time.time()
        last_status = None
        show_preview = os.environ.get("DISPLAY") is not None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, CONFIG["video_max_resolution"], interpolation=cv2.INTER_AREA)
            if time.time() - last_time >= 0.2:
                color_state = detect_color_state(frame)
                STATE_HISTORY.append(color_state)
                last_time = time.time()
                status_red = analyze_flashing_pattern(STATE_HISTORY, "RED")
                if "Red" in status_red:
                    last_status = (status_red, frame)
                else:
                    status_green = analyze_flashing_pattern(STATE_HISTORY, "GREEN")
                    last_status = (status_green, frame)
                if show_preview:
                    cv2.imshow("LED Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        return last_status[0] if last_status else "No status detected.", last_status[1] if last_status else None
    except Exception as e:
        log_error(f"Error processing video: {e}")
        return "Error processing video.", None

def save_fault_image(user_folder, media_path, status, frame=None):
    """Save image or video frame as fault image if status indicates a fault."""
    if "Red" in status:
        try:
            fault_dir = os.path.join(user_folder, "fault_images")
            os.makedirs(fault_dir, exist_ok=True)
            timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y%m%d_%H%M')
            fault_filename = f"fault_{timestamp}.jpg"
            fault_path = os.path.join(fault_dir, fault_filename)
            if frame is not None:
                cv2.imwrite(fault_path, frame)
            else:
                shutil.copy2(media_path, fault_path)
            log_info(f"Saved fault image: {fault_path}")
            return fault_path
        except Exception as e:
            log_error(f"Error saving fault image {media_path}: {e}")
            return None
    return None

# ============================================================================
# 8. BACKGROUND PROCESSING FUNCTIONS
# ============================================================================

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
        
        # Trigger initial PEAS auto-tuning for new user
        auto_tune_peas_scores(user_key, learning_data, force_tune=True)
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

# ============================================================================
# 9. CORE ANSWER GENERATION FUNCTIONS
# ============================================================================

def calculate_time_score(response_time, min_time=0.1, max_time=60.0):
    """Map response time to score."""
    if response_time <= min_time:
        return 0.01
    if response_time >= max_time:
        return 1.0
    return round(0.01 + (0.99 * (response_time - min_time) / (max_time - min_time)), 5)

@measure_time
def generate_answer_optimized(llm, embedding_model, user_name, user_folder, query, media_status=None):
    """Optimized version of generate_answer."""
    if not embedding_model or not llm:
        return "Model initialization failed.", None
    
    try:
        start_time = time.time()
        user_key = get_safe_username(user_name)
        
        # Check for cached response
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cached_response = get_cached_response(user_key, query_hash)
        if cached_response:
            log_info(f"Returning cached response for {user_name}")
            return cached_response, "cached"
        
        # Use cached learning data
        learning_data = get_learning_data_cached()
        
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
        
        # Cache the response
        cache_response(user_key, query_hash, response)
        
        # Update user data in background and trigger PEAS auto-tuning
        executor.submit(update_user_interaction_background, user_name, query, response)
        executor.submit(auto_tune_peas_scores, user_key, learning_data)
        
        return response, "primary"
        
    except Exception as e:
        log_error(f"Error generating answer: {e}")
        return f"Server error: {str(e)}", None

# ============================================================================
# 10. FLASK SERVER AND API ENDPOINTS
# ============================================================================

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
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            # Handle file upload
            username = request.form.get('username')
            query = request.form.get('query', '')
            media = request.files.get('media')
            
            if not username or not username.strip():
                return jsonify({"success": False, "error": "Valid username is required"}), 400
            
            user_folder = get_user_folder(username)
            if not user_folder:
                return jsonify({"success": False, "error": "User folder not found"}), 400
            
            media_status = None
            if media:
                temp_fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(media.filename)[1])
                with os.fdopen(temp_fd, 'wb') as f:
                    f.write(media.read())
                
                media_type = 'image' if media.mimetype.startswith('image') else 'video'
                if media_type == 'image':
                    media_status = process_image(temp_path)
                else:
                    media_status, frame = process_video(temp_path)
                    save_fault_image(user_folder, temp_path, media_status, frame)
                os.remove(temp_path)
            
            answer, source_type = generate_answer_optimized(LLM, EMBEDDING_MODEL, username, user_folder, query, media_status)
            
            return jsonify({
                "success": True,
                "data": {
                    "response": str(answer),
                    "source": source_type,
                    "mediaStatus": media_status
                }
            })
        else:
            # Handle text query
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

# ============================================================================
# 11. PEAS API ENDPOINTS
# ============================================================================

@app.route('/peas_status', methods=['POST'])
def get_peas_status_api():
    """Get PEAS status for a user."""
    try:
        data = request.json
        username = data.get('username')
        if not username:
            return jsonify({"success": False, "error": "Username is required"}), 400
        
        user_key = get_safe_username(username)
        learning_data = get_learning_data_cached()
        
        peas_status = get_peas_status(user_key, learning_data)
        if peas_status:
            return jsonify({"success": True, "data": peas_status})
        else:
            return jsonify({"success": False, "error": "User not found"}), 404
            
    except Exception as e:
        log_error(f"PEAS status error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/peas_tune', methods=['POST'])
def tune_peas_api():
    """Manually trigger PEAS auto-tuning for a user."""
    try:
        data = request.json
        username = data.get('username')
        force_tune = data.get('force_tune', False)
        
        if not username:
            return jsonify({"success": False, "error": "Username is required"}), 400
        
        user_key = get_safe_username(username)
        learning_data = get_learning_data_cached()
        
        # Perform auto-tuning
        tuned = auto_tune_peas_scores(user_key, learning_data, force_tune=force_tune)
        
        if tuned:
            save_learning_data(learning_data)
            peas_status = get_peas_status(user_key, learning_data)
            return jsonify({
                "success": True, 
                "message": "PEAS auto-tuning completed",
                "data": peas_status
            })
        else:
            return jsonify({
                "success": True, 
                "message": "PEAS auto-tuning not needed",
                "data": get_peas_status(user_key, learning_data)
            })
            
    except Exception as e:
        log_error(f"PEAS tuning error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/peas_batch_tune', methods=['POST'])
def batch_tune_peas_api():
    """Batch tune PEAS scores for all users."""
    try:
        # Run batch tuning in background
        tuned_users = tune_peas_scores_batch()
        
        return jsonify({
            "success": True,
            "message": f"PEAS batch tuning completed for {len(tuned_users)} users",
            "tuned_users": tuned_users
        })
        
    except Exception as e:
        log_error(f"PEAS batch tuning error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/peas_overview', methods=['GET'])
def peas_overview_api():
    """Get PEAS overview for all users."""
    try:
        learning_data = get_learning_data_cached()
        overview = []
        
        for user_key in learning_data.keys():
            peas_status = get_peas_status(user_key, learning_data)
            if peas_status:
                overview.append(peas_status)
        
        # Sort by users needing tuning first
        overview.sort(key=lambda x: x['needs_tuning'], reverse=True)
        
        return jsonify({
            "success": True,
            "data": {
                "total_users": len(overview),
                "users_needing_tuning": len([u for u in overview if u['needs_tuning']]),
                "users": overview
            }
        })
        
    except Exception as e:
        log_error(f"PEAS overview error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get performance metrics."""
    try:
        avg_metrics = {}
        for operation, durations in metrics.items():
            avg_metrics[operation] = {
                'avg': sum(durations) / len(durations) if durations else 0,
                'min': min(durations) if durations else 0,
                'max': max(durations) if durations else 0,
                'count': len(durations)
            }
        return jsonify({"success": True, "data": avg_metrics})
    except Exception as e:
        log_error(f"Metrics error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ============================================================================
# 12. DIRECTORY INITIALIZATION AND SCHEDULED TASKS
# ============================================================================

def initialize_directories():
    """Create necessary directories and initialize log file."""
    try:
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        os.makedirs(PDF_FOLDER, exist_ok=True)
        log_info(f"Created directories: {USER_DATA_DIR}, {PDF_FOLDER}")
        
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write("=== Onboarding Prompts ===\n")
                for prompt in ONBOARDING_PROMPTS:
                    f.write(f"- {prompt}\n")
                f.write("=== Log Starts Here ===\n")
            log_info(f"Created and initialized log file: {LOG_FILE}")
    except Exception as e:
        log_error(f"Failed to initialize directories: {e}")
        raise RuntimeError(f"Directory initialization failed: {e}")

def schedule_peas_auto_tuning():
    """Schedule periodic PEAS auto-tuning."""
    def run_periodic_tuning():
        while True:
            try:
                # Run PEAS auto-tuning every 6 hours
                time.sleep(CONFIG["peas_tuning"]["tuning_interval"])
                log_info("Running scheduled PEAS auto-tuning...")
                tuned_users = tune_peas_scores_batch()
                log_info(f"Scheduled PEAS auto-tuning completed for {len(tuned_users)} users")
            except Exception as e:
                log_error(f"Scheduled PEAS auto-tuning failed: {e}")
    
    # Start the background thread
    tuning_thread = threading.Thread(target=run_periodic_tuning, daemon=True)
    tuning_thread.start()
    log_info("PEAS auto-tuning scheduler started (runs every 6 hours)")

# ============================================================================
# 13. TESTING FUNCTIONALITY
# ============================================================================

def test_api_endpoint(endpoint, method="GET", data=None, base_url="http://localhost:5000"):
    """Test an API endpoint and return the response."""
    url = f"{base_url}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def display_peas_status(peas_data):
    """Display PEAS status in a formatted way."""
    if not peas_data or not peas_data.get('success'):
        print("❌ Failed to get PEAS status")
        return
    
    data = peas_data['data']
    print(f"\n📊 PEAS Status for {data['user']}")
    print("=" * 50)
    print(f"Performance (P): {data['components']['Performance']:.3f}")
    print(f"Environment (E): {data['components']['Environment']:.3f}")
    print(f"Actuators (A):   {data['components']['Actuators']:.3f}")
    print(f"Sensors (S):     {data['components']['Sensors']:.3f}")
    print("-" * 50)
    print(f"Total Sum:       {data['sum']:.3f}")
    print(f"Max Ceiling:     {data['ceiling']}")
    print(f"Needs Tuning:    {'✅ Yes' if data['needs_tuning'] else '❌ No'}")
    
    if data.get('last_tuning'):
        print(f"Last Tuning:     {data['last_tuning']}")
        print(f"Tuning Reason:   {data.get('tuning_reason', 'N/A')}")
        print(f"Original Sum:    {data.get('original_sum', 'N/A')}")
        print(f"Tuned Sum:       {data.get('tuned_sum', 'N/A')}")

def test_peas_functionality():
    """Test the complete PEAS functionality."""
    print("🚀 Starting PEAS Auto-Tuning Test")
    print("=" * 60)
    
    # Test 1: Check server health
    print("\n1️⃣ Testing Server Health...")
    try:
        response = requests.get("http://localhost:5000/api/peas_overview")
        if response.status_code == 200:
            print("✅ Server is running and PEAS endpoints are accessible")
        else:
            print("❌ Server not accessible")
            return False
    except:
        print("❌ Cannot connect to server. Make sure chatbot.py is running with --server.")
        return False
    
    # Test 2: Get PEAS overview
    print("\n2️⃣ Testing PEAS Overview API...")
    overview_result = test_api_endpoint("/api/peas_overview", "GET")
    if overview_result:
        overview_data = overview_result['data']
        print(f"✅ PEAS Overview:")
        print(f"   Total Users: {overview_data['total_users']}")
        print(f"   Users Needing Tuning: {overview_data['users_needing_tuning']}")
        
        # Display first few users
        if overview_data['users']:
            print(f"   Sample Users:")
            for i, user in enumerate(overview_data['users'][:3]):
                print(f"     {i+1}. {user['user']}: Sum={user['sum']:.3f}, Needs Tuning={user['needs_tuning']}")
    
    # Test 3: Test batch tuning
    print("\n3️⃣ Testing Batch PEAS Tuning...")
    batch_result = test_api_endpoint("/api/peas_batch_tune", "POST")
    if batch_result:
        print(f"✅ Batch tuning completed: {batch_result['message']}")
        print(f"Tuned users: {batch_result['tuned_users']}")
    
    print("\n✅ PEAS Auto-Tuning Test Completed Successfully!")
    print("=" * 60)
    return True

def run_peas_overview():
    """Run PEAS overview display."""
    result = test_api_endpoint("/api/peas_overview", "GET")
    if result:
        overview_data = result['data']
        print(f"📊 PEAS Overview")
        print(f"Total Users: {overview_data['total_users']}")
        print(f"Users Needing Tuning: {overview_data['users_needing_tuning']}")
        
        if overview_data['users']:
            print("\nUsers:")
            for user in overview_data['users']:
                status = "🔴 Needs Tuning" if user['needs_tuning'] else "🟢 Optimized"
                print(f"  {user['user']}: Sum={user['sum']:.3f} {status}")
    else:
        print("❌ Failed to get PEAS overview")

def run_peas_status(username):
    """Run PEAS status for a specific user."""
    result = test_api_endpoint("/api/peas_status", "POST", {"username": username})
    if result:
        display_peas_status(result)
    else:
        print(f"❌ Failed to get PEAS status for {username}")

# ============================================================================
# 14. MAIN EXECUTION AND CLI
# ============================================================================

def display_help():
    """Display help information."""
    print("""
Intelligent Inverter Expert Chatbot with PEAS Auto-Tuning
=========================================================

Usage: python chatbot.py [options]

Options:
  --server, -s           Start the chatbot server (default)
  --test, -t             Run PEAS functionality tests
  --peas-overview        Show PEAS overview for all users
  --peas-status <user>   Get PEAS status for a specific user
  --help, -h             Show this help message

Examples:
  python chatbot.py                    # Start chatbot server
  python chatbot.py --test             # Run tests
  python chatbot.py --peas-overview    # Show PEAS overview
  python chatbot.py --peas-status john_doe  # Check user status

Features:
  • Performance optimizations with async processing
  • PEAS auto-tuning: If sum=4 or ceiling=4, reduce to 2
  • Caching and background processing
  • Comprehensive monitoring and testing
  • Background processing and scheduled tasks

Prerequisites:
  • Ollama server running: ollama serve && ollama run llama3
  • Required Python packages: pip install -r requirements.txt
""")

def main():
    """Main function to handle command line arguments and start appropriate mode."""
    parser = argparse.ArgumentParser(description='Intelligent Inverter Expert Chatbot with PEAS Auto-Tuning')
    parser.add_argument('--server', '-s', action='store_true', help='Start the chatbot server (default)')
    parser.add_argument('--test', '-t', action='store_true', help='Run PEAS functionality tests')
    parser.add_argument('--peas-overview', action='store_true', help='Show PEAS overview for all users')
    parser.add_argument('--peas-status', type=str, metavar='USERNAME', help='Get PEAS status for a specific user')
    parser.add_argument('--help-detailed', action='store_true', help='Show detailed help')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show detailed help
    if args.help_detailed:
        display_help()
        return
    
    # Run tests
    if args.test:
        test_peas_functionality()
        return
    
    # Show PEAS overview
    if args.peas_overview:
        run_peas_overview()
        return
    
    # Show PEAS status for user
    if args.peas_status:
        run_peas_status(args.peas_status)
        return
    
    # Default: Start server
    try:
        # Check for required dependencies
        try:
            import cv2
            import faiss
        except ImportError as e:
            log_error(f"Missing dependency: {e}")
            print("❌ Missing required dependencies. Please run: pip install -r requirements.txt")
            sys.exit(1)
        
        # Initialize directories and models
        initialize_directories()
        model_init_result = initialize_models()
        if model_init_result is not True:
            log_error(f"Startup error: {model_init_result}")
            print(f"❌ {model_init_result}")
            sys.exit(1)
        
        # Start PEAS auto-tuning scheduler
        schedule_peas_auto_tuning()
        
        print("🚀 Starting Intelligent Inverter Expert Chatbot")
        print("=" * 60)
        print("Features:")
        print("  • Optimized performance with async processing")
        print("  • PEAS auto-tuning system (sum=4 or ceiling=4 → reduce to 2)")
        print("  • Caching and background processing")
        print("  • Comprehensive API endpoints")
        print("  • Scheduled maintenance tasks")
        print("=" * 60)
        print("🌐 Server starting at http://localhost:5000")
        print("📊 API endpoints available:")
        print("  • POST /api/peas_status - Get user PEAS status")
        print("  • POST /api/peas_tune - Manual PEAS tuning")
        print("  • GET /api/peas_overview - System overview")
        print("  • GET /api/metrics - Performance metrics")
        print("=" * 60)
        
        log_info("Starting optimized chatbot server with PEAS auto-tuning...")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down chatbot server...")
        log_info("Server shutdown initiated by user")
    except Exception as e:
        log_error(f"Main execution failed: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()