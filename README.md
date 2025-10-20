# AI-Enabled CRM System - AI Calling Module

A comprehensive AI-powered calling system that provides intelligent voice interactions, natural language processing, and advanced analytics for CRM applications.

## 🚀 Features

### Core AI Capabilities
- **Speech Synthesis (TTS)**: Multiple free and open-source engines (pyttsx3, gTTS, Torch TTS)
- **Speech Recognition (STT)**: Whisper, Google Speech, PocketSphinx support
- **Natural Language Processing**: Intent recognition, sentiment analysis, entity extraction
- **Conversation Management**: Intelligent flow control and state management
- **Multilingual Support**: 20+ languages with cultural adaptation

### Advanced Analytics
- **Real-time Call Analytics**: Performance metrics, quality scores, satisfaction tracking
- **Sentiment Analysis**: Customer emotion detection and trend analysis
- **Predictive Insights**: Success probability, escalation risk, satisfaction prediction
- **Conversation Quality Assessment**: Clarity, engagement, resolution scoring

### Integration & Configuration
- **Dynamic Call Scripts**: Configurable conversation flows and voice settings
- **REST API**: Complete API for backend/frontend integration
- **WebSocket Support**: Real-time communication and updates
- **Webhook Integration**: Event-driven notifications
- **Database Integration**: Call history and analytics storage

## 📁 Project Structure

```
ai_calling_system/
├── __init__.py                 # Package initialization
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── example_usage.py           # Comprehensive usage examples
│
├── ai_calling_agent.py        # Main AI calling orchestrator
├── speech_synthesis.py        # Text-to-speech functionality
├── speech_recognition.py      # Speech-to-text functionality
├── nlp_processor.py           # Natural language processing
├── conversation_manager.py    # Conversation flow management
├── analytics_engine.py        # Analytics and insights generation
├── config_manager.py          # Configuration and call script management
├── multilingual_support.py    # Multi-language support
└── integration_layer.py       # Backend/frontend integration
```

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install additional dependencies for advanced features
pip install googletrans==4.0.0rc1  # For translation
pip install langdetect             # For language detection
pip install flask flask-cors       # For REST API
```

### Optional Dependencies

```bash
# For advanced TTS (Torch TTS)
pip install torch torchaudio transformers

# For spaCy NLP models
python -m spacy download en_core_web_sm

# For Whisper (if not using OpenAI version)
pip install openai-whisper
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from ai_calling_agent import AICallingAgent

async def main():
    # Initialize AI calling agent
    agent = AICallingAgent()
    
    # Define call script
    call_script = {
        'language': 'en',
        'voice_config': {
            'voice': 'default',
            'speed': 1.0,
            'language': 'en'
        },
        'conversation_flow': [
            {
                'ai_message': 'Hello! This is an AI assistant calling to help you with your appointment.',
                'expects_response': True,
                'expected_response_type': 'general'
            },
            {
                'ai_message': 'What time would work best for you tomorrow?',
                'expects_response': True,
                'expected_response_type': 'appointment'
            },
            {
                'ai_message': 'Perfect! I have you scheduled for that time. Thank you!',
                'expects_response': False
            }
        ]
    }
    
    # Initiate call
    session_id = await agent.initiate_call(
        phone_number='+1234567890',
        call_script=call_script,
        contact_name='John Doe'
    )
    
    print(f"Call initiated: {session_id}")

# Run the example
asyncio.run(main())
```

### Using the Integration Layer

```python
from integration_layer import AICallingIntegrationLayer, CallRequest

async def main():
    # Initialize integration layer
    integration = AICallingIntegrationLayer()
    
    # Create call request
    call_request = CallRequest(
        request_id="call_001",
        phone_number="+1234567890",
        contact_name="Jane Smith",
        call_script_id="appointment_booking",
        customizations={
            'voice_settings': {'speed': 1.1, 'emotion': 'friendly'}
        }
    )
    
    # Initiate call
    response = await integration.initiate_call(call_request)
    print(f"Call response: {response}")

asyncio.run(main())
```

## 📚 API Documentation

### REST API Endpoints

#### Health Check
```http
GET /api/v1/health
```
Returns system health status and component information.

#### Create Call
```http
POST /api/v1/calls
Content-Type: application/json

{
    "request_id": "call_001",
    "phone_number": "+1234567890",
    "contact_name": "John Doe",
    "call_script_id": "appointment_booking",
    "customizations": {
        "voice_settings": {"speed": 1.0}
    }
}
```

#### Get Call Status
```http
GET /api/v1/calls/{session_id}
```

#### Cancel Call
```http
POST /api/v1/calls/{session_id}/cancel
```

#### Get Call Analytics
```http
GET /api/v1/calls/{session_id}/analytics
```

#### Get Call History
```http
GET /api/v1/history?limit=100
```

### WebSocket Events

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Listen for call updates
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Call update:', data);
};
```

## 🌍 Multilingual Support

### Supported Languages
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Japanese (ja)
- Korean (ko)
- Chinese Simplified (zh-cn)
- Chinese Traditional (zh-tw)
- Hindi (hi)
- Arabic (ar)
- And more...

### Usage Example

```python
from multilingual_support import MultilingualSupport

# Initialize multilingual support
multilingual = MultilingualSupport()

# Detect language
lang, confidence = await multilingual.detect_language("Hola, ¿cómo estás?")
print(f"Detected: {lang} (confidence: {confidence})")

# Translate text
result = await multilingual.translate_text(
    "Hello, how are you?", 
    target_language="es"
)
print(f"Translation: {result.translated_text}")

# Get localized template
greeting = multilingual.get_localized_template(
    'greeting', 
    'es', 
    contact_name='Juan'
)
print(f"Localized greeting: {greeting}")
```

## 📊 Analytics and Insights

### Real-time Metrics
- Call duration and turn count
- Completion rates and success indicators
- Sentiment trends and emotional peaks
- Quality scores and satisfaction ratings

### Predictive Analytics
- Success probability calculation
- Escalation risk assessment
- Customer satisfaction prediction
- Recommended actions and improvements

### Usage Example

```python
from analytics_engine import AnalyticsEngine

# Initialize analytics engine
analytics = AnalyticsEngine()

# Analyze call session
analytics_data = await analytics.analyze_call(session)

print(f"Quality Score: {analytics_data['call_metrics']['call_quality_score']}")
print(f"Sentiment: {analytics_data['sentiment_metrics']['overall_sentiment']}")
print(f"Success Probability: {analytics_data['predictive_insights']['success_probability']}")
```

## ⚙️ Configuration

### Call Script Templates

```python
from config_manager import ConfigManager, CallScriptTemplate

# Initialize config manager
config_manager = ConfigManager()

# Create custom template
template = CallScriptTemplate(
    template_id='custom_support',
    name='Custom Support Call',
    description='Customized support call flow',
    language='en',
    flow_type='support',
    conversation_flow=[
        # Define conversation turns
    ],
    voice_config=VoiceConfig(
        provider='pyttsx3',
        voice='default',
        speed=1.0,
        language='en'
    ),
    conversation_config={
        'max_turns': 10,
        'timeout_seconds': 300
    }
)

# Create template
config_manager.create_call_script_template(template)
```

### Voice Configuration

```python
from speech_synthesis import VoiceConfig, TTSProvider

# Configure voice settings
voice_config = VoiceConfig(
    provider='pyttsx3',
    voice='default',
    speed=1.2,
    pitch=1.0,
    volume=0.8,
    language='en',
    emotion='friendly'
)

# Use in call script
call_script = {
    'voice_config': asdict(voice_config),
    # ... other settings
}
```

## 🔧 Advanced Features

### Custom NLP Processing

```python
from nlp_processor import NLPProcessor

# Initialize NLP processor
nlp = NLPProcessor(config)

# Process user input
result = await nlp.process_user_input("I'd like to book an appointment")

print(f"Intent: {result.intent.intent.value}")
print(f"Sentiment: {result.sentiment.sentiment.value}")
print(f"Keywords: {result.keywords}")
print(f"Entities: {result.entities}")
```

### Conversation Management

```python
from conversation_manager import ConversationManager

# Initialize conversation manager
conversation_manager = ConversationManager(config)

# Start conversation
context = conversation_manager.start_conversation(
    session_id="session_001",
    contact_name="John Doe",
    phone_number="+1234567890",
    flow_type="appointment_booking"
)

# Process conversation turn
flow_decision = conversation_manager.process_turn(
    session_id="session_001",
    processed_input=user_input
)
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "integration_layer"]
```

### Environment Variables

```bash
# Database configuration
DATABASE_URL=postgresql://user:password@localhost/ai_calling

# API configuration
API_HOST=0.0.0.0
API_PORT=8000

# Telephony integration
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token

# AI model configuration
OPENAI_API_KEY=your_openai_key
WHISPER_MODEL=base
```

## 📈 Performance Optimization

### Recommended Settings

```python
# For high-volume calling
config = {
    'conversation': {
        'max_turns': 15,
        'timeout_seconds': 300,
        'enable_context_memory': True
    },
    'analytics': {
        'enable_real_time': True,
        'enable_predictive_insights': True
    },
    'tts': {
        'default_provider': 'pyttsx3',  # Fastest for local processing
        'cache_audio': True
    }
}
```

### Scaling Considerations

- Use Redis for session storage in multi-instance deployments
- Implement connection pooling for database operations
- Consider using message queues for high-volume scenarios
- Monitor memory usage with large conversation histories

## 🧪 Testing

### Run Examples

```bash
# Run comprehensive examples
python example_usage.py

# Test individual components
python -m ai_calling_system.speech_synthesis
python -m ai_calling_system.speech_recognition
python -m ai_calling_system.nlp_processor
```

### Unit Tests

```bash
# Run tests (when available)
python -m pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the example usage file

## 🔮 Roadmap

- [ ] Advanced voice cloning capabilities
- [ ] Real-time emotion detection
- [ ] Integration with popular CRM systems
- [ ] Advanced conversation AI models
- [ ] Mobile app integration
- [ ] Voice biometrics authentication

---

**Note**: This is the AI part of the CRM system. Your friend will handle the database and backend/frontend integration. The integration layer provides all necessary APIs and webhooks for seamless connection.
