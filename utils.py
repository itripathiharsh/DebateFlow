# utils.py - ENHANCED LOGGING FOR ASSIGNMENT
import os
import re
from typing import List, Dict
from functools import lru_cache
from datetime import datetime

# --------------------------
# Enhanced Logging Setup
# --------------------------

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "debate_log.txt")

def ensure_log_directory():
    """Ensure logs directory exists"""
    os.makedirs(LOG_DIR, exist_ok=True)

def get_timestamp():
    """Get current timestamp in assignment format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_run_start(topic: str):
    """Log debate run start with timestamp"""
    ensure_log_directory()
    timestamp = get_timestamp()
    
    with open(LOG_FILE, "w", encoding="utf-8") as f:  # Overwrite for new run
        f.write(f"=== DEBATE RUN START ===\n")
        f.write(f"[{timestamp}] RUN_START\n")
        f.write(f"[{timestamp}] Topic: {topic}\n")
        f.write(f"[{timestamp}] Starting debate between Scientist and Philosopher...\n\n")

def log_run_end():
    """Log debate run end with timestamp"""
    timestamp = get_timestamp()
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] RUN_END\n")
        f.write(f"=== DEBATE RUN COMPLETE ===\n")

def log_node_event(node_name: str, message: str):
    """Log node events in required format: [timestamp] [NodeName] message"""
    timestamp = get_timestamp()
    formatted_message = f"[{timestamp}] [{node_name}] {message}"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted_message + "\n")
    
    print(f"🔧 [{node_name}] {message}")

def log_round_entry(round_num: int, agent: str, argument: str):
    """Log round entries in required format"""
    timestamp = get_timestamp()
    formatted_entry = f"[{timestamp}] [Round {round_num}] {agent}: {argument}"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted_entry + "\n")
    
    print(f"[Round {round_num}] {agent}: {argument}")

def log_validation(check_type: str, message: str, is_valid: bool = True):
    """Log validation checks"""
    timestamp = get_timestamp()
    status = "PASS" if is_valid else "FAIL"
    formatted_message = f"[{timestamp}] [Validation-{check_type}] {status}: {message}"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted_message + "\n")
    
    if not is_valid:
        print(f" Validation {check_type}: {message}")

def log_memory_update(summary: str, persona_views: Dict[str, str] = None):
    """Log memory updates with persona views"""
    timestamp = get_timestamp()
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [Memory] Updated Summary: {summary}\n")
        
        if persona_views:
            for persona, view in persona_views.items():
                f.write(f"[{timestamp}] [Memory] {persona} Perspective: {view}\n")
    
    print(f" Memory updated: {summary[:100]}...")

def log_judgment(summary: str, winner: str, reason: str):
    """Log final judgment"""
    timestamp = get_timestamp()
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] [Judge] Final Summary: {summary}\n")
        f.write(f"[{timestamp}] [Judge] Winner: {winner}\n")
        f.write(f"[{timestamp}] [Judge] Reason: {reason}\n")
    
    print(f" Judge: {winner} wins - {reason}")

def log_to_file(message: str, include_timestamp=True):
    """General logging function"""
    ensure_log_directory()
    
    timestamp = f"[{get_timestamp()}] " if include_timestamp else ""
    full_message = f"{timestamp}{message}"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")

# --------------------------
# Repetition Detection (Existing)
# --------------------------

# Try to use semantic similarity (more accurate)
# If not available, fall back to keyword overlap
USE_SEMANTIC_CHECK = True

try:
    from sentence_transformers import SentenceTransformer, util
    import torch

    @lru_cache(maxsize=1)
    def _get_embedder():
        # Use a small, fast model
        return SentenceTransformer('all-MiniLM-L6-v2')

    def _semantic_similarity(text1: str, text2: str) -> float:
        model = _get_embedder()
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()

except ImportError:
    # Fallback: disable semantic check
    USE_SEMANTIC_CHECK = False
    log_to_file("[INFO] sentence-transformers not installed. Using keyword-based repetition check.")

def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return ' '.join(text.split())

def _keyword_overlap_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity of word sets."""
    words1 = set(_normalize_text(text1).split())
    words2 = set(_normalize_text(text2).split())
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)

def is_argument_repeated(new_arg: str, history: List[Dict]) -> bool:
    """
    Check if `new_arg` is substantially similar to any past argument.
    
    Returns True if similarity > threshold.
    """
    if not new_arg.strip() or not history:
        return False

    SIMILARITY_THRESHOLD = 0.75  # 75% similarity considered repetition

    for entry in history:
        old_arg = entry["argument"]
        if not old_arg.strip():
            continue

        if USE_SEMANTIC_CHECK:
            try:
                sim = _semantic_similarity(new_arg, old_arg)
            except Exception as e:
                log_to_file(f"[WARNING] Semantic similarity failed: {e}. Falling back to keyword check.")
                sim = _keyword_overlap_similarity(new_arg, old_arg)
        else:
            sim = _keyword_overlap_similarity(new_arg, old_arg)

        if sim >= SIMILARITY_THRESHOLD:
            log_to_file(f"[DEBUG] Repetition detected (sim={sim:.2f}): '{new_arg[:50]}...'")
            return True

    return False