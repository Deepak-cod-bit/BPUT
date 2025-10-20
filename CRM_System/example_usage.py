"""
AI Calling System - Example Usage
=================================

This file demonstrates how to use the AI calling system for various scenarios
including appointment booking, feedback collection, and multilingual calling.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Import AI calling system components
from ai_calling_agent import AICallingAgent, CallSession, CallStatus
from speech_synthesis import SpeechSynthesizer, VoiceConfig, TTSProvider
from speech_recognition import SpeechRecognizer, RecognitionConfig, STTProvider
from nlp_processor import NLPProcessor, ProcessedInput
from conversation_manager import ConversationManager, ConversationState
from analytics_engine import AnalyticsEngine
from config_manager import ConfigManager, CallScriptTemplate
from multilingual_support import MultilingualSupport, LanguageCode
from integration_layer import AICallingIntegrationLayer, CallRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_basic_ai_calling():
    """Example: Basic AI calling functionality"""
    print("=" * 60)
    print("EXAMPLE 1: Basic AI Calling")
    print("=" * 60)
    
    # Initialize AI calling agent
    config = {
        'tts': {'default_provider': 'pyttsx3'},
        'stt': {'default_provider': 'whisper'},
        'nlp': {'enable_sentiment_analysis': True}
    }
    
    agent = AICallingAgent(config)
    
    # Define a simple call script
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
    
    # Add callbacks
    def on_call_started(session):
        print(f"📞 Call started: {session.session_id}")
    
    def on_call_ended(session):
        print(f"✅ Call ended: {session.session_id} - Status: {session.status.value}")
    
    agent.add_callback('call_started', on_call_started)
    agent.add_callback('call_ended', on_call_ended)
    
    # Initiate call
    session_id = await agent.initiate_call(
        phone_number='+1234567890',
        call_script=call_script,
        contact_name='John Doe'
    )
    
    print(f"Call initiated with session ID: {session_id}")
    
    # Wait for call to complete (in real scenario, this would be asynchronous)
    await asyncio.sleep(2)
    
    # Get call status
    session = agent.get_session_status(session_id)
    if session:
        print(f"Call status: {session.status.value}")
        print(f"Conversation turns: {len(session.conversation_log)}")

async def example_multilingual_calling():
    """Example: Multilingual AI calling"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multilingual AI Calling")
    print("=" * 60)
    
    # Initialize multilingual support
    config = {
        'multilingual': {
            'enable_translation': True,
            'default_language': 'en'
        }
    }
    
    multilingual = MultilingualSupport(config)
    
    # Test language detection
    test_texts = [
        "Hello, I'd like to book an appointment",
        "Hola, me gustaría reservar una cita",
        "Bonjour, j'aimerais prendre rendez-vous",
        "こんにちは、予約を取りたいです"
    ]
    
    print("Language Detection:")
    for text in test_texts:
        lang, confidence = await multilingual.detect_language(text)
        print(f"  '{text[:30]}...' -> {lang} (confidence: {confidence:.2f})")
    
    # Test translation
    print("\nTranslation:")
    original = "Hello, I'd like to book an appointment for tomorrow at 2 PM."
    target_languages = ['es', 'fr', 'de', 'ja']
    
    for target_lang in target_languages:
        result = await multilingual.translate_text(original, target_lang)
        print(f"  {target_lang}: {result.translated_text}")
    
    # Test localized templates
    print("\nLocalized Templates:")
    languages = ['en', 'es', 'fr', 'de', 'ja', 'ko', 'zh-cn']
    
    for lang in languages:
        greeting = multilingual.get_localized_template(
            'greeting', 
            lang, 
            contact_name='John'
        )
        print(f"  {lang}: {greeting}")

async def example_analytics_and_insights():
    """Example: Analytics and insights generation"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Analytics and Insights")
    print("=" * 60)
    
    # Initialize analytics engine
    config = {
        'analytics': {
            'enable_real_time': True,
            'enable_predictive_insights': True
        }
    }
    
    analytics = AnalyticsEngine(config)
    
    # Create a mock call session for analysis
    from ai_calling_agent import CallSession, CallStatus
    
    mock_session = CallSession(
        session_id="analytics_test_001",
        phone_number="+1234567890",
        contact_name="Jane Smith",
        call_script={},
        start_time=datetime.now(),
        end_time=datetime.now(),
        status=CallStatus.COMPLETED,
        conversation_log=[
            {
                'turn_number': 1,
                'timestamp': datetime.now().isoformat(),
                'user_input': "Hello, I'd like to book an appointment",
                'intent': 'greeting',
                'sentiment': 'positive',
                'confidence': 0.8
            },
            {
                'turn_number': 2,
                'timestamp': datetime.now().isoformat(),
                'user_input': "Tomorrow at 2 PM works for me",
                'intent': 'appointment',
                'sentiment': 'positive',
                'confidence': 0.9
            },
            {
                'turn_number': 3,
                'timestamp': datetime.now().isoformat(),
                'user_input': "Yes, that's perfect. Thank you!",
                'intent': 'confirmation',
                'sentiment': 'positive',
                'confidence': 0.9
            }
        ],
        analytics={}
    )
    
    # Analyze the call
    analytics_data = await analytics.analyze_call(mock_session)
    
    if analytics_data:
        print("Call Analytics Results:")
        print(f"  Duration: {analytics_data['call_metrics']['duration_seconds']:.1f} seconds")
        print(f"  Turn Count: {analytics_data['call_metrics']['turn_count']}")
        print(f"  Completion Rate: {analytics_data['call_metrics']['completion_rate']:.2f}")
        print(f"  Quality Score: {analytics_data['call_metrics']['call_quality_score']:.2f}")
        print(f"  Satisfaction Score: {analytics_data['call_metrics']['customer_satisfaction_score']:.2f}")
        
        print(f"\nSentiment Analysis:")
        print(f"  Overall Sentiment: {analytics_data['sentiment_metrics']['overall_sentiment']}")
        print(f"  Positive Ratio: {analytics_data['sentiment_metrics']['positive_ratio']:.2f}")
        print(f"  Negative Ratio: {analytics_data['sentiment_metrics']['negative_ratio']:.2f}")
        
        print(f"\nPredictive Insights:")
        print(f"  Success Probability: {analytics_data['predictive_insights']['success_probability']:.2f}")
        print(f"  Escalation Risk: {analytics_data['predictive_insights']['escalation_risk']:.2f}")
        print(f"  Satisfaction Prediction: {analytics_data['predictive_insights']['customer_satisfaction_prediction']:.2f}")
        
        print(f"\nRecommendations:")
        for rec in analytics_data['recommendations']:
            print(f"  - {rec}")

async def example_integration_layer():
    """Example: Integration layer usage"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Integration Layer")
    print("=" * 60)
    
    # Initialize integration layer
    integration = AICallingIntegrationLayer()
    
    # Test system health
    health = await integration.get_system_health()
    print(f"System Health: {health['status']}")
    print(f"Active Calls: {health['active_calls']}")
    
    # Test call script templates
    templates = integration.get_call_script_templates()
    print(f"\nAvailable Call Script Templates: {len(templates)}")
    for template in templates[:3]:
        print(f"  - {template['name']} ({template['template_id']})")
    
    # Create a call request
    call_request = CallRequest(
        request_id="integration_test_001",
        phone_number="+1234567890",
        contact_name="Alice Johnson",
        call_script_id="appointment_booking",
        customizations={
            'voice_settings': {'speed': 1.1, 'emotion': 'friendly'},
            'contact_name': 'Alice Johnson'
        }
    )
    
    print(f"\nCall Request Created:")
    print(f"  Request ID: {call_request.request_id}")
    print(f"  Phone: {call_request.phone_number}")
    print(f"  Contact: {call_request.contact_name}")
    print(f"  Script: {call_request.call_script_id}")
    
    # Test system metrics
    metrics = await integration.get_system_metrics()
    print(f"\nSystem Metrics:")
    print(f"  Active Calls: {metrics['calls']['active']}")
    print(f"  Total Requests: {metrics['calls']['total_requests']}")
    print(f"  Success Rate: {metrics['performance']['success_rate']:.2f}")
    print(f"  Average Duration: {metrics['performance']['average_call_duration']:.1f} seconds")
    
    # Cleanup
    await integration.cleanup()

async def example_custom_call_script():
    """Example: Creating and using custom call scripts"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Call Scripts")
    print("=" * 60)
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Create a custom call script template
    custom_template = CallScriptTemplate(
        template_id='customer_support',
        name='Customer Support Call',
        description='Customer support and issue resolution call flow',
        language='en',
        flow_type='customer_support',
        conversation_flow=[
            {
                'turn_id': 'greeting',
                'ai_message': 'Hello {contact_name}, this is an AI assistant calling from our support team. I understand you may have an issue that needs attention.',
                'expects_response': True,
                'expected_response_type': 'general',
                'timeout_seconds': 10
            },
            {
                'turn_id': 'issue_collection',
                'ai_message': 'Could you please describe the issue you\'re experiencing?',
                'expects_response': True,
                'expected_response_type': 'feedback',
                'timeout_seconds': 30
            },
            {
                'turn_id': 'solution_attempt',
                'ai_message': 'I understand. Let me help you with that. Based on what you\'ve described, I suggest trying {suggested_solution}. Does this make sense?',
                'expects_response': True,
                'expected_response_type': 'confirmation',
                'timeout_seconds': 15
            },
            {
                'turn_id': 'follow_up',
                'ai_message': 'Great! I\'ll also send you an email with detailed instructions. Is there anything else I can help you with today?',
                'expects_response': True,
                'expected_response_type': 'general',
                'timeout_seconds': 10
            },
            {
                'turn_id': 'closing',
                'ai_message': 'Thank you for contacting us! Have a great day!',
                'expects_response': False
            }
        ],
        voice_config=VoiceConfig(
            provider='pyttsx3',
            voice='default',
            speed=0.9,
            language='en',
            emotion='empathetic'
        ),
        conversation_config={
            'max_turns': 12,
            'timeout_seconds': 600,
            'enable_sentiment_analysis': True,
            'escalation_threshold': 0.6
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Create the template
    success = config_manager.create_call_script_template(custom_template)
    print(f"Custom template created: {'Success' if success else 'Failed'}")
    
    # Generate a call script from the template
    call_script = config_manager.generate_call_script(
        'customer_support',
        {
            'voice_settings': {'speed': 0.8, 'emotion': 'empathetic'},
            'flow_customizations': [
                {
                    'turn_id': 'solution_attempt',
                    'suggested_solution': 'restarting your device and clearing the cache'
                }
            ]
        }
    )
    
    if call_script:
        print(f"\nGenerated Call Script:")
        print(f"  Name: {call_script['name']}")
        print(f"  Language: {call_script['language']}")
        print(f"  Flow Type: {call_script['flow_type']}")
        print(f"  Number of Turns: {len(call_script['conversation_flow'])}")
        print(f"  Voice Speed: {call_script['voice_config']['speed']}")
        
        # Show first conversation turn
        first_turn = call_script['conversation_flow'][0]
        print(f"  First Turn: {first_turn['ai_message']}")
    
    # Validate the script
    is_valid, errors = config_manager.validate_call_script(call_script)
    print(f"\nScript Validation: {'Valid' if is_valid else 'Invalid'}")
    if errors:
        print(f"Errors: {errors}")

async def example_advanced_nlp_processing():
    """Example: Advanced NLP processing"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Advanced NLP Processing")
    print("=" * 60)
    
    # Initialize NLP processor
    config = {
        'nlp': {
            'enable_sentiment_analysis': True,
            'enable_intent_recognition': True,
            'enable_entity_extraction': True
        }
    }
    
    nlp_processor = NLPProcessor(config)
    
    # Test various user inputs
    test_inputs = [
        "Hello, I'd like to schedule an appointment for tomorrow at 2 PM",
        "This service is terrible! I'm very disappointed with the quality",
        "Yes, that time works perfectly for me. Thank you so much!",
        "Can you tell me more about your pricing and available services?",
        "I need to cancel my appointment because something came up",
        "Thank you, goodbye! Have a great day!"
    ]
    
    print("NLP Processing Results:")
    print("-" * 40)
    
    for i, text in enumerate(test_inputs, 1):
        print(f"\nInput {i}: '{text}'")
        
        try:
            result = await nlp_processor.process_user_input(text)
            
            print(f"  Intent: {result.intent.intent.value} (confidence: {result.intent.confidence:.2f})")
            print(f"  Sentiment: {result.sentiment.sentiment.value} (confidence: {result.sentiment.confidence:.2f})")
            print(f"  Language: {result.language}")
            print(f"  Keywords: {result.keywords}")
            
            if result.entities:
                print(f"  Entities: {result.entities}")
            
            # Generate response
            response = await nlp_processor.generate_response(
                "I understand your request. Let me help you with that.",
                context=[{'speaker': 'user', 'processed_response': result}]
            )
            print(f"  Generated Response: {response}")
            
        except Exception as e:
            print(f"  Error processing: {e}")

async def main():
    """Run all examples"""
    print("AI CALLING SYSTEM - COMPREHENSIVE EXAMPLES")
    print("=" * 60)
    print("This demonstrates the complete AI calling system capabilities")
    print("including speech synthesis, recognition, NLP, analytics, and more.")
    print("=" * 60)
    
    try:
        # Run all examples
        await example_basic_ai_calling()
        await example_multilingual_calling()
        await example_analytics_and_insights()
        await example_integration_layer()
        await example_custom_call_script()
        await example_advanced_nlp_processing()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe AI calling system is ready for integration with your backend and frontend.")
        print("Key features demonstrated:")
        print("✅ AI-powered voice calling with natural conversation")
        print("✅ Multilingual support with automatic translation")
        print("✅ Advanced analytics and predictive insights")
        print("✅ Flexible call script configuration")
        print("✅ Comprehensive integration layer")
        print("✅ Real-time monitoring and health checks")
        print("\nNext steps:")
        print("1. Install required dependencies: pip install -r requirements.txt")
        print("2. Configure your telephony integration")
        print("3. Set up your database for call storage")
        print("4. Deploy the integration layer API")
        print("5. Connect your frontend to the API endpoints")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        logger.error(f"Example execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
