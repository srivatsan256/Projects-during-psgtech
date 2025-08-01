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
ONBOARDING_PROMPTS = [
    "What's your full name?",
    "What is your inverter model? (Please enter exactly as shown in available models)",
    "What is your serial number?",
    "What is the installed date? (YYYY-MM-DD)",
    "What is your country?",
    "Please enter your inverter installed area PIN code",
    "What is your full address?"
]

# 3. Performance Optimization
executor = ThreadPoolExecutor(max_workers=4)
pdf_cache = {}
faiss_cache = {}
learning_data_cache = {"data": None, "timestamp": 0}
processing_status = {}

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

# 6. Image/Video Processing Variables
RED1 = np.array(CONFIG["hsv_ranges"]["red1_lower"])
RED2 = np.array(CONFIG["hsv_ranges"]["red1_upper"])
RED3 = np.array(CONFIG["hsv_ranges"]["red2_lower"])
RED4 = np.array(CONFIG["hsv_ranges"]["red2_upper"])
GREEN_LOWER = np.array(CONFIG["hsv_ranges"]["green_lower"])
GREEN_UPPER = np.array(CONFIG["hsv_ranges"]["green_upper"])
STATE_HISTORY = deque(maxlen=30)

# 7. Logging Functions
def log_error(message):
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    platform_info = f"Platform: {platform.platform()}, Python: {sys.version.split()[0]}"
    logging.error(f"[ERROR] {message}")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [ERROR] {message}\n{platform_info}\n{traceback.format_exc()}\n{'-'*60}\n")
    except Exception:
        pass

def log_info(message):
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"[INFO] {message}")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [INFO] {message}\n{'-'*60}\n")
    except Exception:
        pass

# 8. Device and Model Initialization
DEVICE = None
EMBEDDING_MODEL = None
LLM = None

def get_device():
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

# 9. Utility Functions
def get_safe_username(username):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', username.strip().lower())

def verify_username_with_full_name(username, full_name):
    username_clean = re.sub(r'[^a-zA-Z0-9]', '', username).lower()
    full_name_clean = re.sub(r'[^a-zA-Z0-9]', '', full_name).lower()
    return username_clean == full_name_clean

def verify_country(country):
    if not country or not country.strip():
        return False
    country = country.strip().title()
    return any(c.name.lower() == country.lower() for c in pycountry.countries)

def verify_pincode_with_address(pin_code, address):
    return True

def get_user_folder(user_name):
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
    if not os.path.exists(PDF_FOLDER):
        return []
    return [os.path.splitext(f)[0].lower() for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]

def sanitize_path(path, base_dir):
    try:
        abs_path = os.path.normpath(os.path.join(base_dir, path))
        if not abs_path.startswith(os.path.abspath(base_dir)):
            raise ValueError("Invalid path: Outside allowed directory")
        return abs_path
    except Exception as e:
        log_error(f"Path sanitization failed for {path}: {e}")
        return None

def validate_media_path(path, media_type):
    sanitized_path = sanitize_path(path, BASE_DIR)
    if not sanitized_path or not os.path.exists(sanitized_path):
        log_info(f"Invalid {media_type} path: {path} does not exist")
        return False
    valid_image_exts = {".jpg", ".jpeg", ".png"}
    valid_video_exts = {".mp4", ".avi", ".mkv"}
    ext = os.path.splitext(sanitized_path)[1].lower()
    if media_type == "image" and ext not in valid_image_exts:
        log_info(f"Unsupported image format for {path}")
        return False
    if media_type == "video" and ext not in valid_video_exts:
        log_info(f"Unsupported video format for {path}")
        return False
    return True

def compute_file_hash(file_path, hash_algo="sha256"):
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
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return len(docs) > 0
    except Exception as e:
        log_error(f"PDF integrity check failed for {pdf_path}: {e}")
        return False

# 10. Caching Functions
@lru_cache(maxsize=10)
def get_pdf_content_cached(pdf_path):
    if pdf_path in pdf_cache:
        return pdf_cache[pdf_path]
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    pdf_cache[pdf_path] = docs
    return docs

def get_learning_data_cached():
    current_time = time.time()
    if (learning_data_cache["data"] is None or
            current_time - learning_data_cache["timestamp"] > 300):
        learning_data_cache["data"] = load_learning_data()
        learning_data_cache["timestamp"] = current_time
    return learning_data_cache["data"]

def get_faiss_index_cached(index_path, embedding_model):
    if index_path in faiss_cache:
        return faiss_cache[index_path]
    vectorstore = load_vectorstore(index_path, embedding_model)
    if vectorstore:
        faiss_cache[index_path] = vectorstore
    return vectorstore

def load_learning_data():
    try:
        if os.path.exists(LEARNING_FILE):
            with open(LEARNING_FILE, "r") as f:
                data = json.load(f)
                for user_key in data:
                    if "rep" not in data[user_key]:
                        data[user_key]["rep"] = {
                            "Performance": {"score": 0.01},
                            "Environment": {"pincode_verified": False, "score": 0.5},
                            "Actuators": {"score": 0.1},
                            "Sensors": {"score": 0.5}
                        }
                    if "model" not in data[user_key]:
                        data[user_key]["model"] = ""
                    if "questions" not in data[user_key]:
                        data[user_key]["questions"] = []
                    if "address" not in data[user_key]:
                        data[user_key]["address"] = {}
                    if "username_verified" not in data[user_key]:
                        data[user_key]["username_verified"] = False
                    if "chat_history" not in data[user_key]:
                        data[user_key]["chat_history"] = ""
                return data
        return {}
    except Exception as e:
        log_error(f"Error loading learning data: {e}")
        return {}

def save_learning_data(data):
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
    try:
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        log_error(f"Error loading FAISS index: {e}")
        return None

def save_index_with_hash(vectorstore, index_path):
    try:
        os.makedirs(index_path, exist_ok=True)
        temp_dir = tempfile.mkdtemp(dir=os.path.dirname(index_path))
        vectorstore.save_local(temp_dir)
        index_file = os.path.join(temp_dir, "index.faiss")
        if os.path.exists(index_file):
            with open(os.path.join(temp_dir, "index_hash.txt"), "w") as f:
                f.write(compute_file_hash(index_file))
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            shutil.move(temp_dir, index_path)
            log_info(f"Saved FAISS index at {index_path}")
    except Exception as e:
        log_error(f"Error saving FAISS index: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)

def validate_index(index_path):
    index_file = os.path.join(index_path, "index.faiss")
    hash_file = os.path.join(index_path, "index_hash.txt")
    if not os.path.exists(index_file) or not os.path.exists(hash_file):
        log_error(f"Invalid FAISS index: Missing files at {index_path}")
        return False
    try:
        with open(hash_file, "r") as f:
            expected_hash = f.read().strip()
        computed_hash = compute_file_hash(index_file)
        if computed_hash != expected_hash:
            log_error(f"FAISS index corrupted at {index_path}: Hash mismatch")
            return False
        return True
    except Exception as e:
        log_error(f"Error validating FAISS index: {e}")
        return False

# 11. Image/Video Processing Functions
def detect_color_state(frame):
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
    sequence = list(history)
    on_count = sum(1 for x in sequence if x == color)
    off_count = sum(1 for x in sequence if x == "OFF")
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
    if not validate_media_path(image_path, "image"):
        return "Invalid image path or format."
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

def process_video(video_path):
    if not validate_media_path(video_path, "video"):
        return "Invalid video path or format.", None
    cap = cv2.VideoCapture(video_path)
    last_time = time.time()
    last_status = None
    show_preview = os.environ.get("DISPLAY") is not None
    if show_preview:
        log_info("Analyzing video for LED status with preview...")
    else:
        log_info("Analyzing video for LED status (no preview, no display detected)...")
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
                log_info(f"Video analysis: {status_red}")
                last_status = (status_red, frame)
            else:
                status_green = analyze_flashing_pattern(STATE_HISTORY, "GREEN")
                log_info(f"Video analysis: {status_green}")
                last_status = (status_green, frame)
            if show_preview:
                cv2.imshow("LED Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    return last_status[0] if last_status else "No status detected.", last_status[1] if last_status else None

def save_fault_image(user_folder, media_path, status, frame=None):
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

# 12. Enhanced PEAS Auto-Tuning System
def auto_tune_peas_scores(user_key, learning_data, force_tune=False):
    try:
        if user_key not in learning_data:
            return False
        user_data = learning_data[user_key]
        peas_scores = user_data.get("rep", {})
        perf_score = peas_scores.get("Performance", {}).get("score", 0.01)
        env_score = peas_scores.get("Environment", {}).get("score", 0.5)
        act_score = peas_scores.get("Actuators", {}).get("score", 0.1)
        sens_score = peas_scores.get("Sensors", {}).get("score", 0.5)
        current_sum = perf_score + env_score + act_score + sens_score
        current_ceiling = max(math.ceil(perf_score), math.ceil(env_score),
                              math.ceil(act_score), math.ceil(sens_score))
        log_info(f"PEAS Analysis for {user_key}: Sum={current_sum:.3f}, Ceiling={current_ceiling}")
        log_info(f"Components: P={perf_score:.3f}, E={env_score:.3f}, A={act_score:.3f}, S={sens_score:.3f}")
        needs_tuning = (abs(current_sum - 4.0) < 0.01) or (current_ceiling == 4) or force_tune
        if needs_tuning:
            log_info(f"PEAS Auto-tuning triggered for {user_key}")
            target_sum = 2.0
            scale_factor = target_sum / current_sum if current_sum > 0 else 0.5
            new_perf_score = min(round(perf_score * scale_factor, 3), 1.0)
            new_env_score = min(round(env_score * scale_factor, 3), 1.0)
            new_act_score = min(round(act_score * scale_factor, 3), 1.0)
            new_sens_score = min(round(sens_score * scale_factor, 3), 1.0)
            user_data["rep"]["Performance"]["score"] = new_perf_score
            user_data["rep"]["Environment"]["score"] = new_env_score
            user_data["rep"]["Actuators"]["score"] = new_act_score
            user_data["rep"]["Sensors"]["score"] = new_sens_score
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
            "needs_tuning": (abs(current_sum - 4.0) < 0.01) or (current_ceiling == 4),
            "last_tuning": peas_scores.get("last_tuning"),
            "tuning_reason": peas_scores.get("tuning_reason"),
            "original_sum": peas_scores.get("original_sum"),
            "tuned_sum": peas_scores.get("tuned_sum")
        }
    except Exception as e:
        log_error(f"Error getting PEAS status for {user_key}: {e}")
        return None

# 13. Background Processing Functions
def create_faiss_index_background(pdf_path, user_folder):
    try:
        log_info(f"Creating FAISS index for {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CONFIG["chunk_size"], chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, EMBEDDING_MODEL)
        index_path = os.path.join(user_folder, "faiss_index")
        save_index_with_hash(vectorstore, index_path)
        log_info(f"FAISS index created successfully at {index_path}")
        return True
    except Exception as e:
        log_error(f"Error creating FAISS index: {e}")
        return False

def onboard_user_async(user_name, user_folder, answers):
    try:
        log_info(f"Starting async onboarding for {user_name}")
        processing_status[user_name] = "processing"
        details_file = os.path.join(user_folder, "details.json")
        with open(details_file, "w") as f:
            json.dump(answers, f, indent=4)
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
        auto_tune_peas_scores(user_key, learning_data, force_tune=True)
        save_learning_data(learning_data)
        model_name = answers["inverter_model"]
        pdf_path = os.path.join(PDF_FOLDER, f"{model_name}.pdf")
        if os.path.exists(pdf_path) and validate_pdf_integrity(pdf_path):
            success = create_faiss_index_background(pdf_path, user_folder)
            processing_status[user_name] = "ready" if success else "error"
        else:
            processing_status[user_name] = "error"
        log_info(f"Background onboarding completed for {user_name}")
        return True
    except Exception as e:
        log_error(f"Background onboarding failed: {e}")
        processing_status[user_name] = "error"
        return False

def update_user_interaction_background(user_name, query, response):
    try:
        learning_data = get_learning_data_cached()
        user_key = get_safe_username(user_name)
        if user_key in learning_data:
            learning_data[user_key]["chat_history"] += f"\nQ: {query}\nA: {response}\n"
            save_learning_data(learning_data)
    except Exception as e:
        log_error(f"Error updating user interaction: {e}")

# 14. Environment and Question Handling
def extract_environment_from_question(question):
    if not question:
        return {}
    match = re.search(r"\b(\d{5,6})\b", question)
    if match:
        return {"pincode": match.group(1)}
    return {}

def load_difficult_questions():
    try:
        if not os.path.exists(DIFFICULT_Q_PATH):
            return []
        with open(DIFFICULT_Q_PATH, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        log_error(f"Error loading difficult questions: {e}")
        return []

def save_difficult_question(user, question, answer):
    entry = {"user": user, "question": question, "answer": answer, "timestamp": datetime.now().isoformat()}
    try:
        with open(DIFFICULT_Q_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        log_info(f"Saved difficult question for {user}: {question}")
    except Exception as e:
        log_error(f"Error saving difficult question: {e}")

def embed_text(text, embedding_model):
    try:
        embedding = embedding_model.embed_query(text)
        return np.array([embedding])
    except Exception as e:
        log_error(f"Error embedding text: {e}")
        return None

def check_similar_difficult_question(question, difficult_qs, embedding_model, threshold=0.9):
    if not question.strip():
        return None
    query_emb = embed_text(question, embedding_model)
    if query_emb is None:
        return None
    for item in difficult_qs:
        q_emb = embed_text(item["question"], embedding_model)
        if q_emb is None:
            continue
        if cosine_similarity(query_emb, q_emb)[0][0] >= threshold:
            return item
    return None

def calculate_time_score(response_time, min_time=0.1, max_time=60.0):
    if response_time <= min_time:
        return 0.01
    if response_time >= max_time:
        return 1.0
    return round(0.01 + (0.99 * (response_time - min_time) / (max_time - min_time)), 5)

# 15. Core Logic Functions
def get_pdf_path_with_fallback(model_name, user_folder):
    try:
        primary_path = os.path.join(PDF_FOLDER, f"{model_name}.pdf")
        secondary_path = os.path.join(user_folder, f"{model_name}.pdf")
        if os.path.exists(primary_path) and validate_pdf_integrity(primary_path):
            return primary_path, "primary"
        elif os.path.exists(secondary_path) and validate_pdf_integrity(secondary_path):
            hash_file = os.path.join(user_folder, f"{model_name}_hash.txt")
            if os.path.exists(hash_file):
                with open(hash_file, "r") as f:
                    expected_hash = f.read().strip()
                computed_hash = compute_file_hash(secondary_path)
                if computed_hash == expected_hash:
                    return secondary_path, "secondary"
            return secondary_path, "secondary"
        log_error(f"No valid PDF found for model '{model_name}'")
        return None, None
    except Exception as e:
        log_error(f"Error fetching PDF path for model '{model_name}': {e}")
        return None, None

@measure_time
def generate_answer_optimized(llm, embedding_model, user_name, user_folder, query, media_status=None):
    if not embedding_model or not llm:
        return "Model initialization failed.", None
    try:
        start_time = time.time()
        learning_data = get_learning_data_cached()
        user_key = get_safe_username(user_name)
        details_file = os.path.join(user_folder, "details.json")
        if not os.path.exists(details_file):
            return "User details not found.", None
        with open(details_file, "r") as f:
            details = json.load(f)
        model_name = details.get("inverter_model")
        if not model_name:
            return "No inverter model specified.", None
        index_path = os.path.join(user_folder, "faiss_index")
        vectorstore = get_faiss_index_cached(index_path, embedding_model)
        if not vectorstore:
            status = processing_status.get(user_name, "unknown")
            if status == "processing":
                return ("I'm still processing your inverter manual. "
                        "Please try again in a few moments for detailed technical answers."), "processing"
            elif status == "error":
                return ("There was an error processing your manual. "
                        "Please contact support."), "error"
            else:
                pdf_path = os.path.join(PDF_FOLDER, f"{model_name}.pdf")
                if os.path.exists(pdf_path) and validate_pdf_integrity(pdf_path):
                    executor.submit(create_faiss_index_background, pdf_path, user_folder)
                    processing_status[user_name] = "processing"
                    return ("I'm processing your inverter manual. "
                            "Please try again in a few moments."), "processing"
                else:
                    return f"No manual found for model '{model_name}'.", "error"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        context = "\n---\n".join(doc.page_content for doc in docs[:3])
        media_info = f"Inverter LED Status: {media_status}\n" if media_status else ""
        env_info = extract_environment_from_question(query)
        if env_info.get("pincode"):
            learning_data[user_key]["rep"]["Environment"]["pin_code"] = env_info["pincode"]
            learning_data[user_key]["rep"]["Environment"]["score"] = 1.0 if learning_data[user_key]["rep"]["Environment"].get("pincode_verified", False) else 0.5
            learning_data[user_key]["rep"]["Sensors"]["score"] = 1.0
            save_learning_data(learning_data)
        prompt = f"""
        You are Inverter Expert for {model_name}.
        {media_info}Context: {context[:2000]}
        User address: {learning_data[user_key]["address"].get("user_address", "N/A")}
        Conversation history: {learning_data[user_key].get("chat_history", "")}
        Question: {query}
        Provide a concise, relevant response, referencing {model_name}.pdf where applicable.
        """
        response = llm.generate([prompt]).generations[0][0].text.strip()
        if any(kw in response.lower() for kw in ["can't", "not working", "error", "problem"]) or (media_status and "Red" in media_status):
            difficult_qs = load_difficult_questions()
            similar = check_similar_difficult_question(query, difficult_qs, embedding_model)
            if similar:
                response += f"\n\nNote: Similar question previously asked:\nQ: {similar['question']}\nA: {similar['answer']}"
            else:
                save_difficult_question(user_name, query, response)
        response_time = time.time() - start_time
        score = calculate_time_score(response_time)
        learning_data[user_key]["questions"].append({
            "question": query,
            "stage": min(len(docs), 26),
            "score": score
        })
        question_scores = [q["score"] for q in learning_data[user_key]["questions"]]
        learning_data[user_key]["rep"]["Performance"]["score"] = round(sum(question_scores) / len(question_scores) if question_scores else 0.01, 5)
        executor.submit(update_user_interaction_background, user_name, query, response)
        executor.submit(auto_tune_peas_scores, user_key, learning_data)
        return response, "primary"
    except Exception as e:
        log_error(f"Error generating answer: {e}")
        return f"Server error: {str(e)}", None

# 16. Flask Server and API Endpoints
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        if not data or not isinstance(data.get('username'), str) or not data['username'].strip():
            return jsonify({"success": False, "error": "Valid username is required"}), 400
        if not isinstance(data.get('fullName'), str) or not data['fullName'].strip():
            return jsonify({"success": False, "error": "Valid full name is required"}), 400
        username = data['username']
        full_name = data['fullName']
        if not verify_username_with_full_name(username, full_name):
            return jsonify({"success": False, "error": "Username does not match full name"}), 401
        user_folder = get_user_folder(username)
        if not user_folder:
            return jsonify({"success": False, "error": "User not found"}), 404
        is_onboarded = os.path.exists(os.path.join(user_folder, "details.json"))
        if not is_onboarded:
            return jsonify({"success": False, "error": "User not onboarded"}), 403
        log_info(f"User {username} logged in successfully")
        return jsonify({"success": True, "message": "Login successful", "isOnboarded": is_onboarded})
    except Exception as e:
        log_error(f"Login error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/check_user_status', methods=['POST'])
def check_user_status():
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
    try:
        data = request.json
        if not data or not isinstance(data.get('username'), str) or not data['username'].strip():
            return jsonify({"success": False, "error": "Valid username is required"}), 400
        user_name = data['username']
        user_folder = get_user_folder(user_name)
        if not user_folder:
            return jsonify({"success": False, "error": "Unable to create user folder"}), 400
        answers = {
            "user_name": data.get("fullName"),
            "inverter_model": data.get("inverterModel"),
            "serial_number": data.get("serialNumber"),
            "installation_date": data.get("installedDate"),
            "country": data.get("country"),
            "pincode": data.get("pinCode"),
            "address": data.get("address")
        }
        for key, value in answers.items():
            if not value:
                return jsonify({"success": False, "error": f"Missing value for {key}"}), 400
            if key == "user_name" and not verify_username_with_full_name(user_name, value):
                return jsonify({"success": False, "error": f"Username '{user_name}' does not match full name '{value}'"}), 400
            if key == "inverter_model" and value.lower() not in list_available_models():
                return jsonify({"success": False, "error": f"Model '{value}' not found"}), 400
            if key == "installation_date":
                try:
                    datetime.strptime(value, "%Y-%m-%d")
                except ValueError:
                    return jsonify({"success": False, "error": "Invalid date format. Use YYYY-MM-DD"}), 400
            if key == "country" and not verify_country(value):
                return jsonify({"success": False, "error": "Invalid country name"}), 400
            if key == "address" and not verify_pincode_with_address(answers.get("pincode"), value):
                return jsonify({"success": False, "error": "Pincode mismatch with address"}), 400
        executor.submit(onboard_user_async, user_name, user_folder, answers)
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
    try:
        data = request.json
        user_name = data['username']
        user_folder = get_user_folder(user_name)
        status = processing_status.get(user_name, "unknown")
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
    try:
        if request.content_type.startswith('multipart/form-data'):
            username = request.form.get('username')
            query = request.form.get('query', '')
            media = request.files.get('media')
            if not username or not username.strip():
                return jsonify({"success": False, "error": "Valid username is required"}), 400
            user_folder = get_user_folder(username)
            if not user_folder:
                return jsonify({"success": False, "error": "User folder not found"}), 400
            with open(os.path.join(user_folder, "details.json"), "r") as f:
                details = json.load(f)
            model_name = details.get("inverter_model")
            if not model_name:
                return jsonify({"success": False, "error": "No inverter model found"}), 400
            media_status = None
            media_path = None
            if media:
                temp_fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(media.filename)[1])
                with os.fdopen(temp_fd, 'wb') as f:
                    f.write(media.read())
                media_type = 'image' if media.mimetype.startswith('image') else 'video'
                if media_type == 'image':
                    media_status = process_image(temp_path)
                    media_path = save_fault_image(user_folder, temp_path, media_status)
                else:
                    media_status, frame = process_video(temp_path)
                    media_path = save_fault_image(user_folder, temp_path, media_status, frame)
                os.remove(temp_path)
            answer, source_type = generate_answer_optimized(LLM, EMBEDDING_MODEL, username, user_folder, query, media_status)
            log_info(f"User: {username}\nQ: {query}\nA: {answer}\nMedia Path: {media_path}\nMedia Status: {media_status}\n{'-'*60}")
            return jsonify({
                "success": True,
                "data": {
                    "response": str(answer),
                    "source": source_type,
                    "mediaStatus": media_status
                }
            })
        else:
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
            answer, source_type = generate_answer_optimized(LLM, EMBEDDING_MODEL, username, user_folder, query)
            log_info(f"User: {username}\nQ: {query}\nA: {answer}\n{'-'*60}")
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

@app.route('/peas_status', methods=['POST'])
def get_peas_status_api():
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
    try:
        data = request.json
        username = data.get('username')
        force_tune = data.get('force_tune', False)
        if not username:
            return jsonify({"success": False, "error": "Username is required"}), 400
        user_key = get_safe_username(username)
        learning_data = get_learning_data_cached()
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
    try:
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
    try:
        learning_data = get_learning_data_cached()
        overview = []
        for user_key in learning_data.keys():
            peas_status = get_peas_status(user_key, learning_data)
            if peas_status:
                overview.append(peas_status)
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

# 17. Directory Initialization
def initialize_directories():
    try:
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        os.makedirs(PDF_FOLDER, exist_ok=True)
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write("=== Onboarding Prompts ===\n")
                for prompt in ONBOARDING_PROMPTS:
                    f.write(f"- {prompt}\n")
                f.write("=== Log Starts Here ===\n")
            log_info(f"Created and initialized log file: {LOG_FILE}")
        log_info(f"Created directories: {USER_DATA_DIR}, {PDF_FOLDER}")
    except Exception as e:
        log_error(f"Failed to initialize directories: {e}")
        raise RuntimeError(f"Directory initialization failed: {e}")

# 18. Scheduled PEAS Auto-Tuning
def schedule_peas_auto_tuning():
    def run_periodic_tuning():
        while True:
            try:
                time.sleep(6 * 3600)
                log_info("Running scheduled PEAS auto-tuning...")
                tuned_users = tune_peas_scores_batch()
                log_info(f"Scheduled PEAS auto-tuning completed for {len(tuned_users)} users")
            except Exception as e:
                log_error(f"Scheduled PEAS auto-tuning failed: {e}")
    tuning_thread = threading.Thread(target=run_periodic_tuning, daemon=True)
    tuning_thread.start()
    log_info("PEAS auto-tuning scheduler started (runs every 6 hours)")

if __name__ == "__main__":
    try:
        import cv2
        import faiss
        initialize_directories()
        model_init_result = initialize_models()
        if model_init_result is not True:
            log_error(f"Startup error: {model_init_result}")
            sys.exit(1)
        schedule_peas_auto_tuning()
        log_info("Starting optimized chatbot server with PEAS auto-tuning...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        log_error(f"Import error: {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"Main execution failed: {e}")
        sys.exit(1)