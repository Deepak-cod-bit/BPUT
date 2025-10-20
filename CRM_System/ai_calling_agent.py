"""
AI Calling Agent - Main orchestrator for AI-powered calling
==========================================================

This module provides the main AI calling agent that orchestrates all AI components
for making intelligent outbound calls with natural language processing.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .speech_synthesis import SpeechSynthesizer
from .speech_recognition import SpeechRecognizer
from .nlp_processor import NLPProcessor
from .conversation_manager import ConversationManager
from .analytics_engine import AnalyticsEngine
from .config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CallStatus(Enum):
    """Call status enumeration"""
    IDLE = "idle"
    INITIATING = "initiating"
    RINGING = "ringing"
    CONNECTED = "connected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class CallSession:
    """Represents a call session"""
    session_id: str
    phone_number: str
    contact_name: Optional[str]
    call_script: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: CallStatus = CallStatus.IDLE
    conversation_log: List[Dict[str, Any]] = None
    analytics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_log is None:
            self.conversation_log = []
        if self.analytics is None:
            self.analytics = {}

class AICallingAgent:
    """
    Main AI Calling Agent that orchestrates all AI components for intelligent calling
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AI Calling Agent
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Initialize AI components
        self.speech_synthesizer = SpeechSynthesizer(self.config)
        self.speech_recognizer = SpeechRecognizer(self.config)
        self.nlp_processor = NLPProcessor(self.config)
        self.conversation_manager = ConversationManager(self.config)
        self.analytics_engine = AnalyticsEngine(self.config)
        
        # Active call sessions
        self.active_sessions: Dict[str, CallSession] = {}
        
        # Event callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'call_started': [],
            'call_ended': [],
            'conversation_turn': [],
            'call_failed': []
        }
        
        logger.info("AI Calling Agent initialized successfully")
    
    def add_callback(self, event: str, callback: Callable):
        """Add event callback"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """Trigger event callbacks"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for event {event}: {e}")
    
    async def initiate_call(self, 
                          phone_number: str, 
                          call_script: Dict[str, Any],
                          contact_name: Optional[str] = None) -> str:
        """
        Initiate an AI-powered call
        
        Args:
            phone_number: Target phone number
            call_script: Call script configuration
            contact_name: Optional contact name
            
        Returns:
            session_id: Unique session identifier
        """
        session_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{phone_number[-4:]}"
        
        # Create call session
        session = CallSession(
            session_id=session_id,
            phone_number=phone_number,
            contact_name=contact_name,
            call_script=call_script,
            start_time=datetime.now(),
            status=CallStatus.INITIATING
        )
        
        self.active_sessions[session_id] = session
        
        try:
            # Update status
            session.status = CallStatus.RINGING
            self._trigger_callback('call_started', session)
            
            # Simulate call initiation (replace with actual telephony integration)
            await self._simulate_call_connection(session)
            
            # Start conversation
            await self._conduct_conversation(session)
            
            # Complete call
            session.status = CallStatus.COMPLETED
            session.end_time = datetime.now()
            
            # Generate analytics
            session.analytics = await self.analytics_engine.analyze_call(session)
            
            self._trigger_callback('call_ended', session)
            
        except Exception as e:
            logger.error(f"Call failed for session {session_id}: {e}")
            session.status = CallStatus.FAILED
            session.end_time = datetime.now()
            self._trigger_callback('call_failed', session, str(e))
        
        finally:
            # Clean up session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        return session_id
    
    async def _simulate_call_connection(self, session: CallSession):
        """Simulate call connection (replace with actual telephony integration)"""
        logger.info(f"Simulating call connection to {session.phone_number}")
        await asyncio.sleep(2)  # Simulate connection time
        session.status = CallStatus.CONNECTED
        logger.info(f"Call connected to {session.phone_number}")
    
    async def _conduct_conversation(self, session: CallSession):
        """Conduct the AI conversation"""
        session.status = CallStatus.IN_PROGRESS
        
        # Get conversation flow from script
        conversation_flow = session.call_script.get('conversation_flow', [])
        
        for turn_index, turn in enumerate(conversation_flow):
            try:
                # Generate AI response
                ai_response = await self._generate_ai_response(session, turn, turn_index)
                
                # Convert to speech
                audio_data = await self.speech_synthesizer.synthesize(
                    text=ai_response,
                    voice_config=session.call_script.get('voice_config', {}),
                    language=session.call_script.get('language', 'en')
                )
                
                # Play audio (simulate - replace with actual audio playback)
                await self._play_audio(audio_data)
                
                # Log conversation turn
                conversation_turn = {
                    'turn_index': turn_index,
                    'speaker': 'ai',
                    'text': ai_response,
                    'timestamp': datetime.now().isoformat(),
                    'audio_duration': len(audio_data) if audio_data else 0
                }
                session.conversation_log.append(conversation_turn)
                
                # Trigger callback
                self._trigger_callback('conversation_turn', session, conversation_turn)
                
                # Wait for user response (simulate - replace with actual speech recognition)
                if turn.get('expects_response', True):
                    user_response = await self._simulate_user_response(session, turn)
                    
                    if user_response:
                        # Process user response with NLP
                        processed_response = await self.nlp_processor.process_user_input(
                            user_response, 
                            context=session.conversation_log
                        )
                        
                        # Log user response
                        user_turn = {
                            'turn_index': turn_index + 0.5,
                            'speaker': 'user',
                            'text': user_response,
                            'processed_response': processed_response,
                            'timestamp': datetime.now().isoformat()
                        }
                        session.conversation_log.append(user_turn)
                        
                        # Check for conversation termination
                        if processed_response.get('intent') == 'end_call':
                            break
                
                # Add delay between turns
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in conversation turn {turn_index}: {e}")
                break
    
    async def _generate_ai_response(self, session: CallSession, turn: Dict, turn_index: int) -> str:
        """Generate AI response using NLP processor"""
        try:
            # Get base response from script
            base_response = turn.get('ai_message', '')
            
            # Personalize response
            if session.contact_name:
                base_response = base_response.replace('{contact_name}', session.contact_name)
            
            # Use NLP to enhance response
            enhanced_response = await self.nlp_processor.generate_response(
                base_response=base_response,
                context=session.conversation_log,
                turn_config=turn
            )
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return turn.get('ai_message', 'Hello, this is an AI assistant calling.')
    
    async def _simulate_user_response(self, session: CallSession, turn: Dict) -> Optional[str]:
        """Simulate user response (replace with actual speech recognition)"""
        # In a real implementation, this would use speech recognition
        # For now, we'll simulate based on the turn configuration
        
        if not turn.get('expects_response', True):
            return None
        
        # Simulate different response types
        response_type = turn.get('expected_response_type', 'general')
        
        simulated_responses = {
            'confirmation': ['yes', 'sure', 'okay', 'that sounds good'],
            'appointment': ['yes, I can meet', 'sure, what time?', 'I\'m available'],
            'feedback': ['it was great', 'good service', 'could be better'],
            'general': ['hello', 'yes', 'no', 'I understand']
        }
        
        responses = simulated_responses.get(response_type, simulated_responses['general'])
        
        # Simulate response delay
        await asyncio.sleep(2)
        
        # Return a random response (in real implementation, this would be speech recognition)
        import random
        return random.choice(responses)
    
    async def _play_audio(self, audio_data: bytes):
        """Play audio data (simulate - replace with actual audio playback)"""
        if audio_data:
            logger.info(f"Playing audio ({len(audio_data)} bytes)")
            # In real implementation, this would play the audio
            await asyncio.sleep(1)  # Simulate audio duration
    
    def get_session_status(self, session_id: str) -> Optional[CallSession]:
        """Get call session status"""
        return self.active_sessions.get(session_id)
    
    def get_all_sessions(self) -> Dict[str, CallSession]:
        """Get all active sessions"""
        return self.active_sessions.copy()
    
    async def cancel_call(self, session_id: str) -> bool:
        """Cancel an active call"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.status = CallStatus.CANCELLED
            session.end_time = datetime.now()
            del self.active_sessions[session_id]
            return True
        return False
    
    async def get_call_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get analytics for a completed call"""
        # In a real implementation, this would retrieve from database
        # For now, return the analytics from the session
        session = self.active_sessions.get(session_id)
        if session and session.analytics:
            return session.analytics
        return None

# Example usage and testing
async def main():
    """Example usage of the AI Calling Agent"""
    
    # Initialize agent
    agent = AICallingAgent()
    
    # Add callbacks
    def on_call_started(session):
        print(f"Call started: {session.session_id} to {session.phone_number}")
    
    def on_call_ended(session):
        print(f"Call ended: {session.session_id} - Status: {session.status}")
    
    agent.add_callback('call_started', on_call_started)
    agent.add_callback('call_ended', on_call_ended)
    
    # Example call script
    call_script = {
        'language': 'en',
        'voice_config': {
            'voice': 'friendly',
            'speed': 1.0,
            'pitch': 1.0
        },
        'conversation_flow': [
            {
                'ai_message': 'Hello {contact_name}, this is an AI assistant calling from our company.',
                'expects_response': True,
                'expected_response_type': 'general'
            },
            {
                'ai_message': 'I\'m calling to confirm your appointment for tomorrow at 2 PM. Is that still convenient for you?',
                'expects_response': True,
                'expected_response_type': 'appointment'
            },
            {
                'ai_message': 'Great! I\'ll send you a confirmation email. Is there anything else I can help you with?',
                'expects_response': True,
                'expected_response_type': 'general'
            },
            {
                'ai_message': 'Thank you for your time. Have a great day!',
                'expects_response': False
            }
        ]
    }
    
    # Make a test call
    session_id = await agent.initiate_call(
        phone_number='+1234567890',
        call_script=call_script,
        contact_name='John Doe'
    )
    
    print(f"Call initiated with session ID: {session_id}")

if __name__ == "__main__":
    asyncio.run(main())
