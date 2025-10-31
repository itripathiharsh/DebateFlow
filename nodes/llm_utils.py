# nodes/llm_utils.py - UPDATED WITH LATEST GROQ MODELS
import os
import time
import random
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Import Groq with fallback
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("  langchain-groq not installed. Install with: pip install langchain-groq")

# Import Hugging Face with fallback
try:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("  langchain-huggingface not installed. Install with: pip install langchain-huggingface")

# Import error handling
from google.api_core import exceptions as google_exceptions
from openai import RateLimitError, APIError, AuthenticationError

class MultiKeyLLMProvider:
    def __init__(self):
        self.google_keys = self._load_google_keys()
        self.openai_keys = self._load_openai_keys()
        self.groq_keys = self._load_groq_keys()
        self.hf_tokens = self._load_hf_tokens()
        
        self.current_google_key_index = 0
        self.current_openai_key_index = 0
        
        # Set priority: Groq > Google > OpenAI > Hugging Face
        if self.groq_keys and GROQ_AVAILABLE:
            self.current_provider = "groq"
        elif self.google_keys:
            self.current_provider = "google"
        elif self.openai_keys:
            self.current_provider = "openai"
        elif self.hf_tokens and HF_AVAILABLE:
            self.current_provider = "huggingface"
        else:
            self.current_provider = None
            
        self.last_switch_time = 0
        self.switch_cooldown = 30
        self.failed_keys = set()
        self.failed_providers = set()
        
    def _load_google_keys(self) -> List[str]:
        """Load multiple Google API keys from environment"""
        keys = []
        single_key = os.getenv("GOOGLE_API_KEY")
        if single_key and single_key.strip():
            keys.append(single_key.strip())
            
        i = 1
        while True:
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if not key or not key.strip():
                break
            keys.append(key.strip())
            i += 1
            
        print(f" Loaded {len(keys)} Google API keys")
        return keys
    
    def _load_openai_keys(self) -> List[str]:
        """Load multiple OpenAI API keys from environment"""
        keys = []
        single_key = os.getenv("OPENAI_API_KEY")
        if single_key and single_key.strip():
            keys.append(single_key.strip())
            
        i = 1
        while True:
            key = os.getenv(f"OPENAI_API_KEY_{i}")
            if not key or not key.strip():
                break
            keys.append(key.strip())
            i += 1
            
        print(f" Loaded {len(keys)} OpenAI API keys")
        return keys
    
    def _load_groq_keys(self) -> List[str]:
        """Load Groq API keys"""
        keys = []
        single_key = os.getenv("GROQ_API_KEY")
        if single_key and single_key.strip():
            keys.append(single_key.strip())
            
        i = 1
        while True:
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if not key or not key.strip():
                break
            keys.append(key.strip())
            i += 1
            
        print(f" Loaded {len(keys)} Groq API keys")
        return keys
    
    def _load_hf_tokens(self) -> List[str]:
        """Load Hugging Face tokens"""
        tokens = []
        single_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if single_token and single_token.strip():
            tokens.append(single_token.strip())
            
        print(f" Loaded {len(tokens)} Hugging Face tokens")
        return tokens
    
    def _get_next_google_key(self) -> Optional[str]:
        """Get next available Google API key"""
        if not self.google_keys:
            return None
            
        available_keys = [k for k in self.google_keys if k not in self.failed_keys]
        if not available_keys:
            print(" All Google API keys are expired or invalid!")
            return None
            
        self.current_google_key_index = (self.current_google_key_index + 1) % len(available_keys)
        key = available_keys[self.current_google_key_index]
        print(f" Using Google key: {key[:15]}...")
        return key
    
    def _get_next_openai_key(self) -> Optional[str]:
        """Get next available OpenAI API key"""
        if not self.openai_keys:
            return None
            
        available_keys = [k for k in self.openai_keys if k not in self.failed_keys]
        if not available_keys:
            print(" All OpenAI API keys are expired or invalid!")
            return None
            
        self.current_openai_key_index = (self.current_openai_key_index + 1) % len(available_keys)
        key = available_keys[self.current_openai_key_index]
        print(f" Using OpenAI key: {key[:15]}...")
        return key
    
    def _get_groq_llm(self, model_name="llama-3.1-8b-instant", temperature=0.7, max_tokens=None):
        """Get Groq LLM with UPDATED MODELS"""
        if not self.groq_keys or not GROQ_AVAILABLE:
            raise ValueError("Groq not available")
        
        current_key = self.groq_keys[0]
        
        # Available Groq models (updated)
        groq_models = {
            "llama-instant": "llama-3.1-8b-instant",
            "llama-70b": "llama-3.1-70b-versatile",
            "mixtral": "mixtral-8x7b-32768",
            "gemma": "gemma2-9b-it"
        }
        
        # Use specified model or default to llama-instant
        actual_model = groq_models.get(model_name, model_name)
        
        return ChatGroq(
            model_name=actual_model,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=current_key
        )
    
    def _get_hf_llm(self, model_name="HuggingFaceH4/zephyr-7b-beta", temperature=0.7, max_tokens=None):
        """Get Hugging Face LLM"""
        if not self.hf_tokens or not HF_AVAILABLE:
            raise ValueError("Hugging Face not available")
        
        hf_token = self.hf_tokens[0]
        return ChatHuggingFace.from_llm(
            llm=HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=hf_token,
                temperature=temperature,
                max_new_tokens=max_tokens or 150
            )
        )
    
    def _get_google_llm(self, model_name="gemini-2.0-flash", temperature=0.7, max_tokens=None):
        """Get Google Gemini LLM"""
        current_key = self._get_next_google_key()
        if not current_key:
            raise ValueError("No available Google API keys")
            
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=current_key,
            max_output_tokens=max_tokens
        )
    
    def _get_openai_llm(self, model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=None):
        """Get OpenAI LLM"""
        current_key = self._get_next_openai_key()
        if not current_key:
            raise ValueError("No available OpenAI API keys")
            
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=current_key
        )
    
    def _mark_key_failed(self, key: str, error_msg: str = ""):
        """Mark a key as failed"""
        self.failed_keys.add(key)
        print(f" Marked key as failed: {key[:15]}... - {error_msg}")
        
        def reset_failed_keys():
            time.sleep(3600)
            self.failed_keys.discard(key)
            print(f" Reset failed key: {key[:15]}...")
        
        import threading
        threading.Thread(target=reset_failed_keys, daemon=True).start()
    
    def _mark_provider_failed(self, provider: str, error_msg: str = ""):
        """Mark a provider as temporarily failed"""
        self.failed_providers.add(provider)
        print(f" Marked provider as failed: {provider} - {error_msg}")
        
        def reset_failed_provider():
            time.sleep(600)
            self.failed_providers.discard(provider)
            print(f" Reset failed provider: {provider}")
        
        import threading
        threading.Thread(target=reset_failed_provider, daemon=True).start()
    
    def _should_switch_provider(self, error):
        """Determine if we should switch providers"""
        current_time = time.time()
        
        if current_time - self.last_switch_time < self.switch_cooldown:
            return False
            
        if isinstance(error, (google_exceptions.Unauthenticated, 
                            google_exceptions.PermissionDenied,
                            AuthenticationError)):
            return True
            
        if isinstance(error, (RateLimitError, google_exceptions.ResourceExhausted)):
            return True
            
        return False
    
    def _get_fallback_provider(self, current_provider):
        """Get the next fallback provider in priority order"""
        providers_priority = ["groq", "google", "openai", "huggingface"]
        
        current_index = providers_priority.index(current_provider) if current_provider in providers_priority else -1
        
        for i in range(current_index + 1, len(providers_priority)):
            provider = providers_priority[i]
            if provider == "groq" and self.groq_keys and GROQ_AVAILABLE and provider not in self.failed_providers:
                return provider
            elif provider == "google" and self.google_keys and provider not in self.failed_providers:
                return provider
            elif provider == "openai" and self.openai_keys and provider not in self.failed_providers:
                return provider
            elif provider == "huggingface" and self.hf_tokens and HF_AVAILABLE and provider not in self.failed_providers:
                return provider
        
        return None
    
    def get_llm(self, model_name=None, temperature=0.7, max_tokens=None):
        """Get LLM with automatic fallback across all providers"""
        if not any([self.google_keys, self.openai_keys, self.groq_keys, self.hf_tokens]):
            raise ValueError("No API keys found. Please set at least one API key in .env file")
        
        # Set default models based on provider (UPDATED)
        provider_models = {
            "groq": "llama-3.1-8b-instant",  # UPDATED MODEL
            "google": "gemini-2.0-flash", 
            "openai": "gpt-3.5-turbo",
            "huggingface": "HuggingFaceH4/zephyr-7b-beta"
        }
        
        if model_name is None and self.current_provider in provider_models:
            model_name = provider_models[self.current_provider]
        
        # Try current provider first
        if self.current_provider == "groq" and self.groq_keys and GROQ_AVAILABLE:
            try:
                llm = self._get_groq_llm(model_name, temperature, max_tokens)
                print(" Using Groq API (Free & Fast)")
                return llm
            except Exception as e:
                print(f"  Groq failed: {e}")
                self._mark_provider_failed("groq", str(e))
                fallback = self._get_fallback_provider("groq")
                if fallback:
                    print(f" Switching to {fallback}...")
                    self.current_provider = fallback
                    self.last_switch_time = time.time()
                    return self.get_llm(None, temperature, max_tokens)
                
        elif self.current_provider == "google" and self.google_keys:
            current_key = self._get_next_google_key()
            try:
                llm = self._get_google_llm(model_name, temperature, max_tokens)
                print(" Using Google Gemini API")
                return llm
            except Exception as e:
                print(f"  Google Gemini failed: {e}")
                if current_key:
                    self._mark_key_failed(current_key, str(e))
                
                if self._should_switch_provider(e):
                    fallback = self._get_fallback_provider("google")
                    if fallback:
                        print(f" Switching to {fallback}...")
                        self.current_provider = fallback
                        self.last_switch_time = time.time()
                        return self.get_llm(None, temperature, max_tokens)
                
                try:
                    return self._get_google_llm(model_name, temperature, max_tokens)
                except Exception as retry_e:
                    self._mark_provider_failed("google", str(retry_e))
                    fallback = self._get_fallback_provider("google")
                    if fallback:
                        self.current_provider = fallback
                        return self.get_llm(None, temperature, max_tokens)
                    raise retry_e
                
        elif self.current_provider == "openai" and self.openai_keys:
            current_key = self._get_next_openai_key()
            try:
                llm = self._get_openai_llm(model_name, temperature, max_tokens)
                print(" Using OpenAI API")
                return llm
            except Exception as e:
                print(f"  OpenAI failed: {e}")
                if current_key:
                    self._mark_key_failed(current_key, str(e))
                
                if self._should_switch_provider(e):
                    fallback = self._get_fallback_provider("openai")
                    if fallback:
                        print(f" Switching to {fallback}...")
                        self.current_provider = fallback
                        self.last_switch_time = time.time()
                        return self.get_llm(None, temperature, max_tokens)
                
                try:
                    return self._get_openai_llm(model_name, temperature, max_tokens)
                except Exception as retry_e:
                    self._mark_provider_failed("openai", str(retry_e))
                    fallback = self._get_fallback_provider("openai")
                    if fallback:
                        self.current_provider = fallback
                        return self.get_llm(None, temperature, max_tokens)
                    raise retry_e
        
        elif self.current_provider == "huggingface" and self.hf_tokens and HF_AVAILABLE:
            try:
                llm = self._get_hf_llm(model_name, temperature, max_tokens)
                print(" Using Hugging Face API")
                return llm
            except Exception as e:
                print(f"  Hugging Face failed: {e}")
                self._mark_provider_failed("huggingface", str(e))
                fallback = self._get_fallback_provider("huggingface")
                if fallback:
                    print(f" Switching to {fallback}...")
                    self.current_provider = fallback
                    self.last_switch_time = time.time()
                    return self.get_llm(None, temperature, max_tokens)
        
        # Final fallback
        for provider in ["groq", "google", "openai", "huggingface"]:
            if provider != self.current_provider and provider not in self.failed_providers:
                if provider == "groq" and self.groq_keys and GROQ_AVAILABLE:
                    print(f" Final fallback to {provider}...")
                    self.current_provider = provider
                    return self._get_groq_llm("llama-3.1-8b-instant", temperature, max_tokens)
                elif provider == "google" and self.google_keys:
                    print(f" Final fallback to {provider}...")
                    self.current_provider = provider
                    return self._get_google_llm("gemini-2.0-flash", temperature, max_tokens)
                elif provider == "openai" and self.openai_keys:
                    print(f" Final fallback to {provider}...")
                    self.current_provider = provider
                    return self._get_openai_llm("gpt-3.5-turbo", temperature, max_tokens)
                elif provider == "huggingface" and self.hf_tokens and HF_AVAILABLE:
                    print(f" Final fallback to {provider}...")
                    self.current_provider = provider
                    return self._get_hf_llm("HuggingFaceH4/zephyr-7b-beta", temperature, max_tokens)
        
        raise ValueError("No working API providers available")

# Create global instance
_llm_provider = MultiKeyLLMProvider()

# Rate limiting decorator
def rate_limited(max_per_minute=20):
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limited(max_per_minute=15)
def get_llm(model_name=None, temperature=0.7, max_tokens=None):
    return _llm_provider.get_llm(model_name, temperature, max_tokens)

# Convenience functions
def get_debate_llm():
    return get_llm(None, temperature=0.7, max_tokens=150)

def get_summary_llm():
    return get_llm(None, temperature=0.2, max_tokens=200)

def get_judge_llm():
    return get_llm(None, temperature=0.3, max_tokens=300)

def get_current_provider():
    return _llm_provider.current_provider

def get_available_keys_info():
    """Get information about available keys"""
    return {
        "google_keys_total": len(_llm_provider.google_keys),
        "openai_keys_total": len(_llm_provider.openai_keys),
        "groq_keys_total": len(_llm_provider.groq_keys),
        "hf_tokens_total": len(_llm_provider.hf_tokens),
        "failed_keys": len(_llm_provider.failed_keys),
        "failed_providers": list(_llm_provider.failed_providers),
        "current_provider": _llm_provider.current_provider
    }