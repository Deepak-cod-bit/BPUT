"""
AI-Enabled CRM System - AI Calling Module
=========================================

This module provides AI-powered calling capabilities for the CRM system including:
- Speech synthesis (Text-to-Speech)
- Speech recognition (Speech-to-Text) 
- Natural Language Processing
- Conversation management
- Sentiment analysis
- Call analytics and insights

Author: AI Assistant
Version: 1.0.0
"""

from .ai_calling_agent import AICallingAgent
from .speech_synthesis import SpeechSynthesizer
from .speech_recognition import SpeechRecognizer
from .nlp_processor import NLPProcessor
from .conversation_manager import ConversationManager
from .analytics_engine import AnalyticsEngine
from .config_manager import ConfigManager

__all__ = [
    'AICallingAgent',
    'SpeechSynthesizer', 
    'SpeechRecognizer',
    'NLPProcessor',
    'ConversationManager',
    'AnalyticsEngine',
    'ConfigManager'
]

__version__ = '1.0.0'
