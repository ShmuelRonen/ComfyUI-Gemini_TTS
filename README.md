# 🎙️ ComfyUI-Gemini_TTS
A powerful ComfyUI custom node that brings Google's Gemini TTS capabilities directly to your workflow. Generate high-quality speech with 30+ voices supporting both free and paid tiers.

## ✨ Features

- **30+ Premium Voices**: Male and female voices with unique characteristics
- **Dual Tier Support**: Free tier with generous limits + Paid tier for production use  
- **Smart Fallback**: Automatic model switching when quotas are reached
- **Voice Characteristics**: Detailed voice info with personality descriptions
- **Flexible Configuration**: Environment variables, node parameters, or config file
- **Robust Error Handling**: Clear error messages and automatic retry logic
- **Real-time Pricing**: Cost estimates for paid tier usage

## 🚀 Quick Start

### 1. Installation

1. **Clone or download** this repository to your ComfyUI custom nodes folder:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone [repository-url] gemini-tts-node
   ```

2. **Install dependencies**:
   ```bash
   cd gemini-tts-node
   pip install google-generativeai requests torch torchaudio numpy
   ```

3. **Restart ComfyUI** - The node will appear as "🎙️ Gemini Text-to-Speech"

### 2. Get Your API Key

#### Free Tier (Recommended to Start)
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API Key" → "Create API Key" 
4. Select "Create API key in new project"
5. Copy your API key (starts with `AIza...`)

#### Paid Tier (For Production)
See the [Paid Tier Setup](#-paid-tier-setup) section below.

### 3. Configure the Node

**Option A: Environment Variable (Recommended)**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

**Option B: Direct Input**
- Enter your API key directly in the node's `api_key` field
- The node will save it automatically for future use

## 🎭 Available Voices

### Female Voices (14 total)
- **Aoede** - Breezy and natural
- **Kore** - Firm and confident  
- **Leda** - Youthful and energetic
- **Zephyr** - Bright and cheerful
- **Autonoe** - Bright and optimistic
- **Callirhoe** - Easy-going and relaxed
- **Despina** - Smooth and flowing
- **Erinome** - Clear and precise
- **Gacrux** - Mature and experienced
- **Laomedeia** - Upbeat and lively
- **Pulcherrima** - Forward and expressive
- **Sulafat** - Warm and welcoming
- **Vindemiatrix** - Gentle and kind
- **Achernar** - Soft and gentle

### Male Voices (16 total)
- **Puck** - Upbeat and energetic (default)
- **Charon** - Informative and clear
- **Fenrir** - Excitable and dynamic
- **Orus** - Firm and decisive
- **Achird** - Friendly and approachable
- **Algenib** - Gravelly texture
- **Algieba** - Smooth and pleasant
- **Alnilam** - Firm and strong
- **Enceladus** - Breathy and soft
- **Iapetus** - Clear and articulate
- **Rasalgethi** - Informative and professional
- **Sadachbia** - Lively and animated
- **Sadaltager** - Knowledgeable and authoritative
- **Schedar** - Even and balanced
- **Umbriel** - Easy-going and calm
- **Zubenelgenubi** - Casual and conversational

## ⚙️ Node Parameters

### Required Parameters
- **`prompt`**: Text to convert to speech (supports "Say:" prefix)
- **`tts_model`**: Choose between:
  - `gemini-2.5-pro-preview-tts` (Higher quality, slower)
  - `gemini-2.5-flash-preview-tts` (Faster, good quality)
- **`voice`**: Select from 30+ available voices
- **`temperature`**: Control creativity (0.0-2.0, default: 1.0)

### Optional Parameters
- **`api_key`**: Enter API key directly (auto-saved)
- **`auto_fallback_to_flash`**: Auto-switch to Flash if Pro is rate-limited
- **`retry_delay`**: Wait time between retries (10-120 seconds)
- **`use_paid_tier`**: Enable paid billing for higher quotas
- **`billing_project_id`**: Google Cloud project ID for billing
- **`aggressive_retry`**: More retry attempts for better reliability
- **`show_voice_info`**: Display voice characteristics in output

## 💰 Paid Tier Setup

### Why Upgrade to Paid Tier?

| Feature | Free Tier | Paid Tier |
|---------|-----------|-----------|
| **Quota Limits** | Low (good for testing) | High (production ready) |
| **Rate Limits** | Restrictive | Generous |
| **Priority Access** | Standard | Premium |
| **Cost** | Free | ~$0.001-0.02 per request |

### Step-by-Step Paid Setup

#### 1. Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "New Project" or select existing project
3. Enter project name (e.g., "my-gemini-tts")
4. Note your **Project ID** (not the name - this is important!)

#### 2. Enable Billing
1. In Google Cloud Console, go to **Billing**
2. Click "Link a billing account" or "Enable billing"
3. Add a payment method (credit card required)
4. Verify billing is active on your project

#### 3. Enable the Gemini API
1. Go to **APIs & Services > Library**
2. Search for "Generative Language API"
3. Click "Enable" on the Generative Language API
4. Wait for activation (usually instant)

#### 4. Create API Key
1. Go to **APIs & Services > Credentials**
2. Click "Create Credentials" > "API Key"
3. Copy your new API key
4. **Optional**: Restrict the key to "Generative Language API" for security

#### 5. Configure the Node
Set these parameters in the node:
- **`use_paid_tier`**: `True`
- **`billing_project_id`**: Your Project ID from step 1
- **`api_key`**: Your API key from step 4

### 💵 Pricing Information

**Gemini 2.5 Pro TTS**:
- Input: $1.00 per 1M tokens
- Output: $20.00 per 1M tokens
- ~$0.01-0.02 per typical request

**Gemini 2.5 Flash TTS**:
- Input: $0.50 per 1M tokens  
- Output: $10.00 per 1M tokens
- ~$0.005-0.01 per typical request

*Typical 20-word sentence costs less than $0.02*

## 🔧 Troubleshooting

### Common Issues

#### "API key not valid" Error
- **Solution**: Verify your API key starts with `AIza` and is ~39 characters
- **Check**: API key hasn't expired or been deleted
- **Verify**: You're using the correct key from Google AI Studio or Cloud Console

#### "Rate limit exceeded" Error  
- **Free Tier**: Wait 60 seconds or try Flash model
- **Solution**: Enable paid tier for higher quotas
- **Temporary**: Use `auto_fallback_to_flash = True`

#### "Billing project not found" Error
- **Check**: Use Project ID, not project name
- **Verify**: Project exists and billing is enabled
- **Confirm**: API key belongs to the same project

#### "Permission denied" Error
- **Verify**: Generative Language API is enabled
- **Check**: API key has proper permissions
- **Ensure**: Billing is active if using paid tier

### Configuration Files

The node creates a `config.json` file to save your settings:
```json
{
    "GEMINI_API_KEY": "your_key_here",
    "use_paid_tier": true,
    "billing_project_id": "your-project-id"
}
```

### Debug Information

Enable debugging by checking console output:
- **Green ✅**: Successful operations
- **Yellow ⚠️**: Warnings and fallbacks  
- **Red ❌**: Errors requiring attention

## 📝 Usage Examples

### Basic Text-to-Speech
```
Prompt: "Hello, welcome to our presentation today."
Model: gemini-2.5-flash-preview-tts
Voice: [F] Zephyr
Temperature: 1.0
```

### Expressive Reading
```
Prompt: "Say: Once upon a time, in a land far, far away..."
Model: gemini-2.5-pro-preview-tts  
Voice: [M] Charon
Temperature: 1.5
Show Voice Info: True
```

### Production Setup
```
Use Paid Tier: True
Billing Project ID: my-production-project-123
Aggressive Retry: True
Model: gemini-2.5-pro-preview-tts
```

## 🛡️ Security Best Practices

1. **Protect Your API Key**: Never commit API keys to version control
2. **Use Environment Variables**: Set `GEMINI_API_KEY` in your environment
3. **Restrict API Keys**: Limit to specific APIs in Google Cloud Console
4. **Monitor Usage**: Check Google Cloud billing dashboard regularly
5. **Project Isolation**: Use separate projects for development vs production

## 🔄 Updates and Compatibility

- **ComfyUI**: Compatible with latest versions
- **Python**: Requires Python 3.8+
- **Dependencies**: Auto-updated through pip
- **Voice Library**: Automatically synced with Google's latest voices

## 📞 Support

### Common Solutions
1. **Restart ComfyUI** after installation or configuration changes
2. **Check Console Output** for detailed error messages
3. **Verify API Key Format** (should start with `AIza`)
4. **Confirm Project Settings** in Google Cloud Console

### Getting Help
- Check the [troubleshooting section](#-troubleshooting) above
- Review console output for specific error messages
- Verify your Google Cloud project configuration
- Ensure billing is properly enabled for paid tier

## 📜 License

This project is provided as-is for educational and commercial use. Google Gemini API usage is subject to Google's terms of service and pricing.

---

**🎉 Ready to generate amazing speech with Gemini TTS!**

*Last updated: May 2025*
