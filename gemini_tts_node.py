# Gemini_TTS_Node.py
import os
import json
import base64
import tempfile
import torch
import torchaudio
import numpy as np
from io import BytesIO
import google.generativeai as genai

p = os.path.dirname(os.path.realpath(__file__))

def get_config():
    try:
        config_path = os.path.join(p, 'config.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(p, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# Official Gemini TTS voices with exact gender information updated from provided list
GEMINI_VOICES_WITH_GENDER = [
    # Female voices
    ("[F] Aoede", "Aoede"),      # Female - Breezy style
    ("[F] Kore", "Kore"),        # Female - Firm style  
    ("[F] Leda", "Leda"),        # Female - Youthful style
    ("[F] Zephyr", "Zephyr"),    # Female - Bright style
    ("[F] Autonoe", "Autonoe"),  # Female - Bright style
    ("[F] Callirhoe", "Callirhoe"), # Female - Easy-going style
    ("[F] Despina", "Despina"),  # Female - Smooth style
    ("[F] Erinome", "Erinome"),  # Female - Clear style
    ("[F] Gacrux", "Gacrux"),    # Female - Mature style
    ("[F] Laomedeia", "Laomedeia"), # Female - Upbeat style
    ("[F] Pulcherrima", "Pulcherrima"), # Female - Forward style
    ("[F] Sulafat", "Sulafat"),  # Female - Warm style
    ("[F] Vindemiatrix", "Vindemiatrix"), # Female - Gentle style
    ("[F] Achernar", "Achernar"), # Female - Soft style
    
    # Male voices
    ("[M] Puck", "Puck"),        # Male - Upbeat style
    ("[M] Charon", "Charon"),    # Male - Informative style
    ("[M] Fenrir", "Fenrir"),    # Male - Excitable style
    ("[M] Orus", "Orus"),        # Male - Firm style
    ("[M] Achird", "Achird"),    # Male - Friendly style  
    ("[M] Algenib", "Algenib"),  # Male - Gravelly style
    ("[M] Algieba", "Algieba"),  # Male - Smooth style
    ("[M] Alnilam", "Alnilam"),  # Male - Firm style
    ("[M] Enceladus", "Enceladus"), # Male - Breathy style
    ("[M] Iapetus", "Iapetus"),  # Male - Clear style
    ("[M] Rasalgethi", "Rasalgethi"), # Male - Informative style
    ("[M] Sadachbia", "Sadachbia"), # Male - Lively style
    ("[M] Sadaltager", "Sadaltager"), # Male - Knowledgeable style
    ("[M] Schedar", "Schedar"),  # Male - Even style
    ("[M] Umbriel", "Umbriel"),  # Male - Easy-going style
    ("[M] Zubenelgenubi", "Zubenelgenubi"), # Male - Casual style
]

# Extract just the voice names for the API
GEMINI_VOICES_DISPLAY = [display_name for display_name, _ in GEMINI_VOICES_WITH_GENDER]
GEMINI_VOICES_API = [api_name for _, api_name in GEMINI_VOICES_WITH_GENDER]

# Voice characteristics with updated gender information
VOICE_CHARACTERISTICS_UPDATED = {
    # Female voices
    "Aoede": "Female • Breezy and natural",
    "Kore": "Female • Firm and confident", 
    "Leda": "Female • Youthful and energetic",
    "Zephyr": "Female • Bright and cheerful",
    "Autonoe": "Female • Bright and optimistic",
    "Callirhoe": "Female • Easy-going and relaxed",
    "Despina": "Female • Smooth and flowing",
    "Erinome": "Female • Clear and precise",
    "Gacrux": "Female • Mature and experienced",
    "Laomedeia": "Female • Upbeat and lively",
    "Pulcherrima": "Female • Forward and expressive",
    "Sulafat": "Female • Warm and welcoming",
    "Vindemiatrix": "Female • Gentle and kind",
    "Achernar": "Female • Soft and gentle",
    
    # Male voices  
    "Puck": "Male • Upbeat and energetic",
    "Charon": "Male • Informative and clear",
    "Fenrir": "Male • Excitable and dynamic",
    "Orus": "Male • Firm and decisive",
    "Achird": "Male • Friendly and approachable",
    "Algenib": "Male • Gravelly texture",
    "Algieba": "Male • Smooth and pleasant",
    "Alnilam": "Male • Firm and strong",
    "Enceladus": "Male • Breathy and soft",
    "Iapetus": "Male • Clear and articulate",
    "Rasalgethi": "Male • Informative and professional",
    "Sadachbia": "Male • Lively and animated",
    "Sadaltager": "Male • Knowledgeable and authoritative",
    "Schedar": "Male • Even and balanced",
    "Umbriel": "Male • Easy-going and calm",
    "Zubenelgenubi": "Male • Casual and conversational",
}

class GeminiTTS:
    def __init__(self, api_key=None):
        env_key = os.environ.get("GEMINI_API_KEY")

        # Common placeholder values to ignore
        placeholders = {"token_here", "place_token_here", "your_api_key",
                        "api_key_here", "enter_your_key", "<api_key>"}

        if env_key and env_key.lower().strip() not in placeholders:
            self.api_key = env_key
        else:
            self.api_key = api_key
            if self.api_key is None:
                config = get_config()
                self.api_key = config.get("GEMINI_API_KEY")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Say: Hello, this is a test of Gemini text-to-speech.", "multiline": True}),
                "tts_model": (["gemini-2.5-pro-preview-tts", "gemini-2.5-flash-preview-tts"], {"default": "gemini-2.5-pro-preview-tts"}),
                "voice": (GEMINI_VOICES_DISPLAY, {"default": "[M] Puck"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "auto_fallback_to_flash": ("BOOLEAN", {"default": True}),
                "retry_delay": ("INT", {"default": 30, "min": 10, "max": 120}),
                "use_paid_tier": ("BOOLEAN", {"default": False}),
                "billing_project_id": ("STRING", {"default": ""}),
                "aggressive_retry": ("BOOLEAN", {"default": False}),
                "show_voice_info": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status")
    FUNCTION = "generate_speech"
    CATEGORY = "Gemini TTS"

    def generate_speech(self, prompt, tts_model="gemini-2.5-pro-preview-tts", voice="[M] Puck", 
                       temperature=1.0, api_key="", auto_fallback_to_flash=True, retry_delay=30, 
                       use_paid_tier=False, billing_project_id="", aggressive_retry=False, 
                       show_voice_info=False):
        """Generate speech using Gemini TTS with paid tier support and intelligent fallback"""
        
        # Handle API key with better validation
        if api_key.strip():
            self.api_key = api_key.strip()
            config_data = {
                "GEMINI_API_KEY": self.api_key,
                "use_paid_tier": use_paid_tier,
                "billing_project_id": billing_project_id.strip() if billing_project_id.strip() else None
            }
            save_config(config_data)
            print(f"🔑 Using provided API key: {self.api_key[:15]}...{self.api_key[-5:]}")

        if not self.api_key:
            error_msg = "❌ API key required. Please set GEMINI_API_KEY environment variable or enter in node."
            empty_audio = {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000}
            return (empty_audio, error_msg)

        # Validate API key format
        if not self.api_key.startswith("AIza") or len(self.api_key) < 35:
            error_msg = f"❌ Invalid API key format. Key should start with 'AIza' and be ~39 characters long.\n"
            error_msg += f"Current key: {self.api_key[:15]}... (length: {len(self.api_key)})"
            empty_audio = {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000}
            return (empty_audio, error_msg)

        # Validate billing project ID format if using paid tier
        if use_paid_tier and billing_project_id.strip():
            project_id = billing_project_id.strip()
            # Project IDs should be 6-30 characters, lowercase letters, numbers, and hyphens
            if not project_id.replace('-', '').replace('_', '').isalnum():
                error_msg = f"❌ Invalid project ID format: {project_id}\n"
                error_msg += f"💡 Project IDs should contain only letters, numbers, and hyphens\n"
                error_msg += f"💡 Example: 'my-project-123' or 'project-name'\n"
                error_msg += f"💡 Find your project ID in Google Cloud Console"
                empty_audio = {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000}
                return (empty_audio, error_msg)

        # Convert display name to API name
        voice_api_name = voice
        if voice in GEMINI_VOICES_DISPLAY:
            # Find the corresponding API name
            for display_name, api_name in GEMINI_VOICES_WITH_GENDER:
                if display_name == voice:
                    voice_api_name = api_name
                    break
        
        # Display tier information with voice characteristics
        tier_info = "💰 Paid Tier" if use_paid_tier else "🆓 Free Tier"
        if use_paid_tier and billing_project_id.strip():
            tier_info += f" (Project: {billing_project_id.strip()[:20]}...)"
        
        # Show voice characteristics if requested
        voice_info = ""
        if show_voice_info and voice_api_name in VOICE_CHARACTERISTICS_UPDATED:
            voice_info = f"🎭 {voice}: {VOICE_CHARACTERISTICS_UPDATED[voice_api_name]}"
        
        print(f"🎙️ Generating TTS: Model={tts_model}, Voice={voice} -> {voice_api_name}, Temp={temperature}")
        print(f"📝 Prompt: {prompt[:100]}...")
        print(f"🔑 API key (partial): {self.api_key[:15]}...{self.api_key[-5:]} (length: {len(self.api_key)})")
        print(f"🏪 Billing: {tier_info}")
        if voice_info:
            print(f"🎭 {voice_info}")
        
        # Determine retry behavior based on tier
        max_retries = 5 if (use_paid_tier or aggressive_retry) else 1
        
        # Try the requested model first
        try:
            return self.try_official_tts(prompt, tts_model, voice_api_name, temperature, use_paid_tier, 
                                       billing_project_id.strip(), max_retries, show_voice_info)
        except Exception as error:
            error_str = str(error)
            print(f"⚠️ {tts_model} failed: {error_str}")
            
            # Handle rate limiting with paid tier awareness
            if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str):
                return self.handle_rate_limiting(error_str, tts_model, prompt, voice_api_name, temperature, 
                                               auto_fallback_to_flash, retry_delay, use_paid_tier, 
                                               billing_project_id.strip(), max_retries, show_voice_info)
            
            # Handle API key errors
            elif "API key not valid" in error_str or "INVALID_ARGUMENT" in error_str:
                print("🔄 Falling back to working Gemini model for TTS simulation...")
                return self.fallback_tts_simulation(prompt, voice_api_name, temperature)
            
            # Handle billing/quota errors
            elif "PERMISSION_DENIED" in error_str or "billing" in error_str.lower() or "USER_PROJECT_DENIED" in error_str or "not found or deleted" in error_str:
                return self.handle_billing_error(error_str, use_paid_tier, billing_project_id.strip())
            
            # Handle other errors
            else:
                return self.handle_complete_failure(error_str, retry_delay, tts_model)

    def try_official_tts(self, prompt, tts_model, voice, temperature, use_paid_tier=False, 
                        billing_project_id="", max_retries=1, show_voice_info=False):
        """Try the official TTS API with paid tier support"""
        import requests
        import json
        import time
        
        # Construct URL with paid tier considerations
        base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{tts_model}:generateContent"
        
        url = f"{base_url}?key={self.api_key}"
        
        headers = {"Content-Type": "application/json", "User-Agent": "ComfyUI-Gemini-TTS/1.0"}
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice
                        }
                    }
                }
            }
        }
        
        print(f"🌐 Making REST request to: {url[:80]}...?key=***")
        print(f"📦 Request data: Model={tts_model}, Voice={voice}, Temp={temperature}")
        
        for attempt in range(max_retries):
            try:
                timeout = 60 if use_paid_tier else 30
                response = requests.post(url, headers=headers, json=data, timeout=timeout)
                print(f"📊 Response status: {response.status_code} (attempt {attempt + 1}/{max_retries})")
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    if ("candidates" in response_data and len(response_data["candidates"]) > 0 and
                        "content" in response_data["candidates"][0] and
                        "parts" in response_data["candidates"][0]["content"] and
                        len(response_data["candidates"][0]["content"]["parts"]) > 0):
                        
                        part = response_data["candidates"][0]["content"]["parts"][0]
                        
                        if "inlineData" in part and "data" in part["inlineData"]:
                            audio_data_b64 = part["inlineData"]["data"]
                            audio_data = base64.b64decode(audio_data_b64)
                            
                            # Convert PCM data to tensor
                            audio_np = np.frombuffer(audio_data, dtype=np.int16)
                            audio_float = audio_np.astype(np.float32) / 32768.0
                            waveform = torch.from_numpy(audio_float).unsqueeze(0)
                            
                            audio_dict = {
                                "waveform": waveform.unsqueeze(0),
                                "sample_rate": 24000
                            }
                            
                            tier_label = "💰 Paid" if use_paid_tier else "🆓 Free"
                            success_msg = f"✅ REST TTS Success: {tts_model} with {voice} voice\n"
                            success_msg += f"🏪 Tier: {tier_label} | 📊 Generated {len(audio_float)} samples at 24kHz"
                            
                            if show_voice_info and voice in VOICE_CHARACTERISTICS_UPDATED:
                                success_msg += f"\n🎭 Voice: {VOICE_CHARACTERISTICS_UPDATED[voice]}"
                            
                            return (audio_dict, success_msg)
                        else:
                            raise Exception("No audio data found in REST response")
                    else:
                        raise Exception("Invalid REST response structure")
                        
                elif response.status_code == 429:
                    error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                    raise Exception(f"Rate limit (429): {error_data}")
                    
                elif response.status_code == 403:
                    error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                    if use_paid_tier:
                        raise Exception(f"Billing/Permission error (403): Check billing project '{billing_project_id}' and API access. {error_data}")
                    else:
                        raise Exception(f"Permission denied (403): {error_data}")
                    
                else:
                    error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                    if attempt < max_retries - 1 and response.status_code >= 500:
                        backoff_time = 2 ** attempt
                        print(f"⚠️ Server error {response.status_code}, retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)
                        continue
                    else:
                        raise Exception(f"REST API error {response.status_code}: {error_data}")
                        
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    backoff_time = 2 ** attempt
                    print(f"⚠️ Request timeout, retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    continue
                else:
                    raise Exception("Request timeout after retries")
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    backoff_time = 2 ** attempt
                    print(f"⚠️ Request error: {e}, retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    continue
                else:
                    raise Exception(f"Request failed: {e}")
        
        raise Exception("Max retries exceeded")

    def handle_rate_limiting(self, error_str, tts_model, prompt, voice, temperature, 
                           auto_fallback_to_flash, retry_delay, use_paid_tier, 
                           billing_project_id, max_retries, show_voice_info):
        """Handle rate limiting with paid tier awareness"""
        
        if use_paid_tier:
            error_msg = f"⚠️ Unexpected rate limit on paid tier: {tts_model}\n"
            error_msg += f"💰 Billing Project: {billing_project_id or 'default'}\n"
            error_msg += f"💡 Check billing project configuration and quotas\n"
            error_msg += f"⏰ Retry in {retry_delay} seconds"
            empty_audio = {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000}
            return (empty_audio, error_msg)
        
        if "pro" in tts_model.lower():
            if auto_fallback_to_flash:
                try:
                    flash_model = "gemini-2.5-flash-preview-tts"
                    result = self.try_official_tts(prompt, flash_model, voice, temperature, 
                                                 use_paid_tier, billing_project_id, max_retries, show_voice_info)
                    audio, original_msg = result
                    fallback_msg = f"⚠️ Fallback Success: Used Flash model (Pro was rate limited)\n"
                    fallback_msg += f"🎙️ Voice: {voice} (Flash quality)\n" 
                    fallback_msg += f"💡 Consider upgrading to paid tier for consistent Pro access\n"
                    fallback_msg += f"💰 Paid Pro: $1.00 input + $20.00 output per 1M tokens\n"
                    fallback_msg += f"💰 Paid Flash: $0.50 input + $10.00 output per 1M tokens\n"
                    fallback_msg += f"📊 Generated audio at 24kHz"
                    return (audio, fallback_msg)
                except Exception as flash_error:
                    flash_error_str = str(flash_error)
                    if "429" in flash_error_str or "RESOURCE_EXHAUSTED" in flash_error_str:
                        return self.fallback_tts_simulation(prompt, voice, temperature, both_models_exhausted=True)
                    else:
                        return self.handle_complete_failure(flash_error_str, retry_delay, "both models")
            else:
                error_msg = f"🚫 {tts_model} API quota exceeded (Free Tier)\n"
                error_msg += f"💰 Upgrade to paid tier for higher quotas:\n"
                if "pro" in tts_model.lower():
                    error_msg += f"  • Pro TTS: $1.00 input + $20.00 output per 1M tokens\n"
                else:
                    error_msg += f"  • Flash TTS: $0.50 input + $10.00 output per 1M tokens\n"
                error_msg += f"⏰ Try again in {retry_delay} seconds"
                empty_audio = {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000}
                return (empty_audio, error_msg)
        else:
            return self.fallback_tts_simulation(prompt, voice, temperature, both_models_exhausted=True)

    def handle_billing_error(self, error_str, use_paid_tier, billing_project_id):
        """Handle billing and permission errors"""
        if use_paid_tier:
            if "not found or deleted" in error_str:
                error_msg = f"🚫 Billing Project Error\n"
                error_msg += f"🏗️ Project ID: {billing_project_id or 'Not specified'}\n"
                error_msg += f"❌ Project not found or deleted\n\n"
                error_msg += f"🔧 Solutions:\n"
                error_msg += f"  • Verify project ID in Google Cloud Console\n"
                error_msg += f"  • Ensure project exists and is active\n"
                error_msg += f"  • Check if project was accidentally deleted\n"
                error_msg += f"  • Use project ID (not project name)\n\n"
                error_msg += f"💡 For free tier, leave billing project empty"
            elif "USER_PROJECT_DENIED" in error_str:
                error_msg = f"🚫 Project Access Denied\n"
                error_msg += f"🏗️ Project ID: {billing_project_id or 'Not specified'}\n"
                error_msg += f"❌ API key doesn't have access to this project\n\n"
                error_msg += f"🔧 Solutions:\n"
                error_msg += f"  • Ensure API key was created in this project\n"
                error_msg += f"  • Grant API key access to the project\n"
                error_msg += f"  • Check IAM permissions\n"
                error_msg += f"  • Try using the project where API key was created\n\n"
                error_msg += f"💡 For free tier, leave billing project empty"
            else:
                error_msg = f"💳 Billing Configuration Error\n"
                error_msg += f"🏗️ Project: {billing_project_id or 'Not specified'}\n"
                error_msg += f"🔧 Check:\n"
                error_msg += f"  • Billing is enabled on the project\n"
                error_msg += f"  • Gemini API is enabled in the project\n"
                error_msg += f"  • API key has access to the billing project\n"
                error_msg += f"  • Project ID is correct\n"
                error_msg += f"💡 Visit Google Cloud Console to verify billing settings"
        else:
            error_msg = f"🚫 API Access Error (Free Tier)\n"
            error_msg += f"💡 This might be resolved by:\n"
            error_msg += f"  • Upgrading to paid tier\n"
            error_msg += f"  • Checking API permissions\n"
            error_msg += f"  • Verifying account status"
        
        error_msg += f"\n🔧 Original error: {error_str[:100]}..."
        empty_audio = {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000}
        return (empty_audio, error_msg)

    def fallback_tts_simulation(self, prompt, voice, temperature, both_models_exhausted=False):
        """Enhanced fallback using the working Gemini model to simulate TTS"""
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            if both_models_exhausted:
                simulation_type = "Both TTS models are temporarily out of quota"
            else:
                simulation_type = "TTS preview models need special access"
            
            tts_prompt = f"""
            You are simulating the {voice} voice from Gemini TTS for text-to-speech generation.
            
            {simulation_type}, so provide detailed voice acting instructions for:
            VOICE: {voice}
            TEXT: "{prompt}"
            STYLE: Temperature {temperature} (0.0=consistent, 2.0=creative)
            
            Provide voice characteristics for {voice} including tone, pace, and delivery style.
            """
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024
            )
            
            response = model.generate_content(tts_prompt, generation_config=generation_config)
            
            # Create placeholder audio
            words = prompt.replace("Say:", "").replace("Say ", "").strip().split()
            estimated_duration = max(2.0, len(words) * 0.4)
            samples = int(24000 * estimated_duration)
            
            import math
            t = torch.linspace(0, estimated_duration, samples)
            placeholder_tone = 0.001 * torch.sin(2 * math.pi * 440 * t)
            placeholder_waveform = placeholder_tone.unsqueeze(0)
            
            audio_dict = {
                "waveform": placeholder_waveform.unsqueeze(0),
                "sample_rate": 24000
            }
            
            if both_models_exhausted:
                fallback_msg = f"🚫 Both TTS Models Exhausted - Voice Simulation Mode\n"
                fallback_msg += f"🎭 Requested Voice: {voice} (characteristics preserved)\n"
                fallback_msg += f"⏰ Quotas renew: Per-minute (60s) | Daily (24h)\n"
                fallback_msg += f"💡 Voice Instructions: {response.text[:150]}..."
            else:
                fallback_msg = f"⚠️ TTS Simulation Mode (Preview models need access)\n"
                fallback_msg += f"🎭 Requested Voice: {voice}\n"
                fallback_msg += f"📝 Voice Instructions: {response.text[:150]}..."
            
            return (audio_dict, fallback_msg)
            
        except Exception as fallback_error:
            error_msg = f"❌ Complete system failure: {str(fallback_error)}\n"
            error_msg += f"🎭 Requested Voice: {voice} (preserved in message)"
            empty_audio = {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000}
            return (empty_audio, error_msg)

    def calculate_pricing_estimate(self, prompt, tts_model, use_paid_tier):
        """Calculate estimated pricing for the TTS request"""
        if not use_paid_tier:
            return "Free tier - no charges"
        
        input_chars = len(prompt)
        estimated_input_tokens = input_chars / 4
        estimated_audio_seconds = len(prompt.split()) * 0.4
        estimated_output_tokens = estimated_audio_seconds * 1000
        
        if "pro" in tts_model.lower():
            input_cost_per_1m = 1.00
            output_cost_per_1m = 20.00
        else:
            input_cost_per_1m = 0.50
            output_cost_per_1m = 10.00
        
        input_cost = (estimated_input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (estimated_output_tokens / 1_000_000) * output_cost_per_1m
        total_cost = input_cost + output_cost
        
        return f"Estimated cost: ~${total_cost:.4f} (Input: {estimated_input_tokens:.0f} tokens, Output: {estimated_output_tokens:.0f} tokens)"
    
    def calculate_actual_cost(self, prompt, audio_samples, tts_model):
        """Calculate more accurate cost based on actual audio output"""
        input_chars = len(prompt)
        estimated_input_tokens = input_chars / 4
        audio_seconds = audio_samples / 24000
        estimated_output_tokens = audio_seconds * 1000
        
        if "pro" in tts_model.lower():
            input_cost_per_1m = 1.00
            output_cost_per_1m = 20.00
        else:
            input_cost_per_1m = 0.50
            output_cost_per_1m = 10.00
        
        input_cost = (estimated_input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (estimated_output_tokens / 1_000_000) * output_cost_per_1m
        total_cost = input_cost + output_cost
        
        return total_cost

    def handle_complete_failure(self, error_str, retry_delay, tts_model):
        """Handle complete TTS failure with helpful messaging"""
        error_msg = f"❌ TTS failed: {tts_model}\n"
        if "429" in error_str:
            error_msg += f"⏰ Rate limited - try again in {retry_delay} seconds\n"
        error_msg += f"🔧 Error: {error_str[:150]}..."
        empty_audio = {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000}
        return (empty_audio, error_msg)

NODE_CLASS_MAPPINGS = {
    "GeminiTTS": GeminiTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiTTS": "🎙️ Gemini Text-to-Speech",
}