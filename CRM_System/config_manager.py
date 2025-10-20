"""
Configuration Manager Module - Dynamic configuration management
============================================================

This module provides comprehensive configuration management for the AI calling system
including dynamic call script configuration, voice settings, and system parameters.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import yaml

logger = logging.getLogger(__name__)

class ConfigType(Enum):
    """Configuration types"""
    SYSTEM = "system"
    VOICE = "voice"
    CONVERSATION = "conversation"
    ANALYTICS = "analytics"
    INTEGRATION = "integration"

@dataclass
class VoiceConfig:
    """Voice configuration parameters"""
    provider: str = "pyttsx3"
    voice: str = "default"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 0.8
    language: str = "en"
    gender: str = "neutral"
    emotion: str = "neutral"
    rate: int = 150
    pause_between_sentences: float = 0.5

@dataclass
class ConversationConfig:
    """Conversation configuration parameters"""
    max_turns: int = 20
    timeout_seconds: int = 300
    escalation_threshold: float = 0.7
    confidence_threshold: float = 0.3
    sentiment_threshold: float = 0.5
    enable_context_memory: bool = True
    context_memory_size: int = 10
    enable_sentiment_analysis: bool = True
    enable_intent_recognition: bool = True
    enable_entity_extraction: bool = True

@dataclass
class AnalyticsConfig:
    """Analytics configuration parameters"""
    enable_real_time: bool = True
    enable_historical_analysis: bool = True
    trend_analysis_days: int = 7
    quality_threshold: float = 0.6
    satisfaction_threshold: float = 0.7
    escalation_risk_threshold: float = 0.5
    enable_predictive_insights: bool = True
    enable_sentiment_tracking: bool = True
    enable_performance_metrics: bool = True

@dataclass
class IntegrationConfig:
    """Integration configuration parameters"""
    enable_telephony: bool = False
    telephony_provider: str = "twilio"
    enable_database: bool = False
    database_url: str = ""
    enable_webhook: bool = False
    webhook_url: str = ""
    enable_api: bool = True
    api_port: int = 8000
    enable_logging: bool = True
    log_level: str = "INFO"

@dataclass
class CallScriptTemplate:
    """Call script template"""
    template_id: str
    name: str
    description: str
    language: str
    flow_type: str
    conversation_flow: List[Dict[str, Any]]
    voice_config: VoiceConfig
    conversation_config: ConversationConfig
    created_at: datetime
    updated_at: datetime
    is_active: bool = True

class ConfigManager:
    """
    Configuration manager for AI calling system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "ai_calling_config.json"
        self.config = {}
        self.call_script_templates: Dict[str, CallScriptTemplate] = {}
        
        # Load configuration
        self._load_config()
        
        # Initialize default configurations
        self._initialize_default_configs()
        
        logger.info("Configuration Manager initialized successfully")
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        self.config = yaml.safe_load(f)
                    else:
                        self.config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.info("No configuration file found, using defaults")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = {}
    
    def _initialize_default_configs(self):
        """Initialize default configurations"""
        if 'system' not in self.config:
            self.config['system'] = {
                'name': 'AI Calling System',
                'version': '1.0.0',
                'debug': False,
                'log_level': 'INFO'
            }
        
        if 'tts' not in self.config:
            self.config['tts'] = {
                'default_provider': 'pyttsx3',
                'providers': {
                    'pyttsx3': {
                        'enabled': True,
                        'voice': 'default',
                        'speed': 1.0,
                        'volume': 0.8
                    },
                    'gtts': {
                        'enabled': True,
                        'language': 'en'
                    },
                    'torch_tts': {
                        'enabled': False,
                        'model': 'microsoft/speecht5_tts'
                    }
                }
            }
        
        if 'stt' not in self.config:
            self.config['stt'] = {
                'default_provider': 'whisper',
                'providers': {
                    'whisper': {
                        'enabled': True,
                        'model': 'base',
                        'language': 'en'
                    },
                    'google': {
                        'enabled': True,
                        'language': 'en'
                    },
                    'sphinx': {
                        'enabled': False,
                        'language': 'en'
                    }
                }
            }
        
        if 'nlp' not in self.config:
            self.config['nlp'] = {
                'spacy_model': 'en_core_web_sm',
                'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'intent_model': None,
                'enable_entity_extraction': True,
                'enable_sentiment_analysis': True,
                'enable_intent_recognition': True
            }
        
        if 'conversation' not in self.config:
            self.config['conversation'] = {
                'max_turns': 20,
                'timeout_seconds': 300,
                'escalation_threshold': 0.7,
                'confidence_threshold': 0.3,
                'sentiment_threshold': 0.5,
                'enable_context_memory': True,
                'context_memory_size': 10
            }
        
        if 'analytics' not in self.config:
            self.config['analytics'] = {
                'enable_real_time': True,
                'enable_historical_analysis': True,
                'trend_analysis_days': 7,
                'quality_threshold': 0.6,
                'satisfaction_threshold': 0.7,
                'escalation_risk_threshold': 0.5,
                'enable_predictive_insights': True
            }
        
        if 'integration' not in self.config:
            self.config['integration'] = {
                'enable_telephony': False,
                'telephony_provider': 'twilio',
                'enable_database': False,
                'database_url': '',
                'enable_webhook': False,
                'webhook_url': '',
                'enable_api': True,
                'api_port': 8000,
                'enable_logging': True,
                'log_level': 'INFO'
            }
        
        # Load default call script templates
        self._load_default_call_scripts()
    
    def _load_default_call_scripts(self):
        """Load default call script templates"""
        default_scripts = {
            'appointment_booking': CallScriptTemplate(
                template_id='appointment_booking',
                name='Appointment Booking',
                description='Standard appointment booking conversation flow',
                language='en',
                flow_type='appointment_booking',
                conversation_flow=[
                    {
                        'turn_id': 'greeting',
                        'ai_message': 'Hello {contact_name}, this is an AI assistant calling from our company. I\'m calling to help you schedule an appointment.',
                        'expects_response': True,
                        'expected_response_type': 'general',
                        'timeout_seconds': 10
                    },
                    {
                        'turn_id': 'collect_details',
                        'ai_message': 'What time would work best for you? I have availability tomorrow and the day after.',
                        'expects_response': True,
                        'expected_response_type': 'appointment',
                        'timeout_seconds': 15
                    },
                    {
                        'turn_id': 'confirm_details',
                        'ai_message': 'Perfect! I have you down for {appointment_time}. Is this correct?',
                        'expects_response': True,
                        'expected_response_type': 'confirmation',
                        'timeout_seconds': 10
                    },
                    {
                        'turn_id': 'closing',
                        'ai_message': 'Excellent! I\'ll send you a confirmation email shortly. Is there anything else I can help you with?',
                        'expects_response': True,
                        'expected_response_type': 'general',
                        'timeout_seconds': 10
                    },
                    {
                        'turn_id': 'end_call',
                        'ai_message': 'Thank you for your time! Have a great day!',
                        'expects_response': False
                    }
                ],
                voice_config=VoiceConfig(
                    provider='pyttsx3',
                    voice='default',
                    speed=1.0,
                    language='en',
                    emotion='friendly'
                ),
                conversation_config=ConversationConfig(
                    max_turns=10,
                    timeout_seconds=300,
                    enable_sentiment_analysis=True
                ),
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            'feedback_collection': CallScriptTemplate(
                template_id='feedback_collection',
                name='Feedback Collection',
                description='Customer feedback collection conversation flow',
                language='en',
                flow_type='feedback_collection',
                conversation_flow=[
                    {
                        'turn_id': 'greeting',
                        'ai_message': 'Hello {contact_name}, this is an AI assistant calling from our company. I\'m calling to get your feedback about our recent service.',
                        'expects_response': True,
                        'expected_response_type': 'general',
                        'timeout_seconds': 10
                    },
                    {
                        'turn_id': 'collect_feedback',
                        'ai_message': 'How would you rate your overall experience with our service on a scale of 1 to 10?',
                        'expects_response': True,
                        'expected_response_type': 'feedback',
                        'timeout_seconds': 15
                    },
                    {
                        'turn_id': 'detailed_feedback',
                        'ai_message': 'Thank you for that rating. Could you tell me what we did well and what we could improve?',
                        'expects_response': True,
                        'expected_response_type': 'feedback',
                        'timeout_seconds': 20
                    },
                    {
                        'turn_id': 'closing',
                        'ai_message': 'Thank you so much for your valuable feedback! We really appreciate you taking the time to share your thoughts.',
                        'expects_response': True,
                        'expected_response_type': 'general',
                        'timeout_seconds': 10
                    },
                    {
                        'turn_id': 'end_call',
                        'ai_message': 'Have a wonderful day! Goodbye!',
                        'expects_response': False
                    }
                ],
                voice_config=VoiceConfig(
                    provider='pyttsx3',
                    voice='default',
                    speed=0.9,
                    language='en',
                    emotion='grateful'
                ),
                conversation_config=ConversationConfig(
                    max_turns=8,
                    timeout_seconds=400,
                    enable_sentiment_analysis=True
                ),
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            'appointment_reminder': CallScriptTemplate(
                template_id='appointment_reminder',
                name='Appointment Reminder',
                description='Appointment reminder conversation flow',
                language='en',
                flow_type='appointment_reminder',
                conversation_flow=[
                    {
                        'turn_id': 'greeting',
                        'ai_message': 'Hello {contact_name}, this is an AI assistant calling to remind you about your appointment tomorrow at {appointment_time}.',
                        'expects_response': True,
                        'expected_response_type': 'general',
                        'timeout_seconds': 10
                    },
                    {
                        'turn_id': 'confirm_attendance',
                        'ai_message': 'Are you still able to make it to this appointment?',
                        'expects_response': True,
                        'expected_response_type': 'confirmation',
                        'timeout_seconds': 10
                    },
                    {
                        'turn_id': 'handle_response',
                        'ai_message': 'Great! We\'ll see you tomorrow at {appointment_time}. If you need to reschedule, please call us at your earliest convenience.',
                        'expects_response': False
                    },
                    {
                        'turn_id': 'end_call',
                        'ai_message': 'Thank you and have a great day!',
                        'expects_response': False
                    }
                ],
                voice_config=VoiceConfig(
                    provider='pyttsx3',
                    voice='default',
                    speed=1.0,
                    language='en',
                    emotion='professional'
                ),
                conversation_config=ConversationConfig(
                    max_turns=6,
                    timeout_seconds=200,
                    enable_sentiment_analysis=True
                ),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        }
        
        for template_id, template in default_scripts.items():
            self.call_script_templates[template_id] = template
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return self.config.copy()
    
    def get_config_section(self, section: str) -> Dict[str, Any]:
        """Get specific configuration section"""
        return self.config.get(section, {}).copy()
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> bool:
        """
        Update configuration section
        
        Args:
            section: Configuration section name
            updates: Updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if section not in self.config:
                self.config[section] = {}
            
            self.config[section].update(updates)
            self._save_config()
            logger.info(f"Configuration section '{section}' updated")
            return True
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config, f, indent=2, default=str)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def create_call_script_template(self, template: CallScriptTemplate) -> bool:
        """
        Create a new call script template
        
        Args:
            template: Call script template to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.call_script_templates[template.template_id] = template
            self._save_call_script_templates()
            logger.info(f"Call script template '{template.template_id}' created")
            return True
        except Exception as e:
            logger.error(f"Error creating call script template: {e}")
            return False
    
    def get_call_script_template(self, template_id: str) -> Optional[CallScriptTemplate]:
        """Get call script template by ID"""
        return self.call_script_templates.get(template_id)
    
    def get_all_call_script_templates(self) -> Dict[str, CallScriptTemplate]:
        """Get all call script templates"""
        return self.call_script_templates.copy()
    
    def update_call_script_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update call script template
        
        Args:
            template_id: Template ID to update
            updates: Updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if template_id not in self.call_script_templates:
                return False
            
            template = self.call_script_templates[template_id]
            
            # Update template fields
            for key, value in updates.items():
                if hasattr(template, key):
                    setattr(template, key, value)
            
            template.updated_at = datetime.now()
            self._save_call_script_templates()
            logger.info(f"Call script template '{template_id}' updated")
            return True
        except Exception as e:
            logger.error(f"Error updating call script template: {e}")
            return False
    
    def delete_call_script_template(self, template_id: str) -> bool:
        """
        Delete call script template
        
        Args:
            template_id: Template ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if template_id in self.call_script_templates:
                del self.call_script_templates[template_id]
                self._save_call_script_templates()
                logger.info(f"Call script template '{template_id}' deleted")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting call script template: {e}")
            return False
    
    def _save_call_script_templates(self):
        """Save call script templates to file"""
        try:
            templates_file = self.config_path.replace('.json', '_templates.json').replace('.yaml', '_templates.yaml')
            
            # Convert templates to serializable format
            serializable_templates = {}
            for template_id, template in self.call_script_templates.items():
                serializable_templates[template_id] = asdict(template)
                # Convert datetime objects to strings
                serializable_templates[template_id]['created_at'] = template.created_at.isoformat()
                serializable_templates[template_id]['updated_at'] = template.updated_at.isoformat()
            
            with open(templates_file, 'w', encoding='utf-8') as f:
                if templates_file.endswith('.yaml') or templates_file.endswith('.yml'):
                    yaml.dump(serializable_templates, f, default_flow_style=False, indent=2)
                else:
                    json.dump(serializable_templates, f, indent=2, default=str)
            
            logger.info(f"Call script templates saved to {templates_file}")
        except Exception as e:
            logger.error(f"Error saving call script templates: {e}")
    
    def generate_call_script(self, 
                           template_id: str, 
                           customizations: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Generate a call script from template with customizations
        
        Args:
            template_id: Template ID to use
            customizations: Customizations to apply
            
        Returns:
            Generated call script or None if template not found
        """
        template = self.get_call_script_template(template_id)
        if not template:
            return None
        
        # Start with template data
        script = {
            'template_id': template.template_id,
            'name': template.name,
            'description': template.description,
            'language': template.language,
            'flow_type': template.flow_type,
            'conversation_flow': template.conversation_flow.copy(),
            'voice_config': asdict(template.voice_config),
            'conversation_config': asdict(template.conversation_config),
            'created_at': datetime.now().isoformat()
        }
        
        # Apply customizations
        if customizations:
            for key, value in customizations.items():
                if key in script:
                    script[key] = value
                elif key == 'voice_settings' and 'voice_config' in script:
                    script['voice_config'].update(value)
                elif key == 'conversation_settings' and 'conversation_config' in script:
                    script['conversation_config'].update(value)
                elif key == 'flow_customizations' and 'conversation_flow' in script:
                    # Apply flow customizations
                    for flow_customization in value:
                        turn_id = flow_customization.get('turn_id')
                        if turn_id:
                            for turn in script['conversation_flow']:
                                if turn.get('turn_id') == turn_id:
                                    turn.update(flow_customization)
                                    break
        
        return script
    
    def validate_call_script(self, script: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate call script configuration
        
        Args:
            script: Call script to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Required fields
        required_fields = ['template_id', 'name', 'language', 'conversation_flow']
        for field in required_fields:
            if field not in script:
                errors.append(f"Missing required field: {field}")
        
        # Validate conversation flow
        if 'conversation_flow' in script:
            flow = script['conversation_flow']
            if not isinstance(flow, list):
                errors.append("conversation_flow must be a list")
            else:
                for i, turn in enumerate(flow):
                    if not isinstance(turn, dict):
                        errors.append(f"Turn {i} must be a dictionary")
                    else:
                        if 'ai_message' not in turn:
                            errors.append(f"Turn {i} missing required field: ai_message")
                        if 'expects_response' not in turn:
                            errors.append(f"Turn {i} missing required field: expects_response")
        
        # Validate voice config
        if 'voice_config' in script:
            voice_config = script['voice_config']
            if not isinstance(voice_config, dict):
                errors.append("voice_config must be a dictionary")
            else:
                if 'provider' not in voice_config:
                    errors.append("voice_config missing required field: provider")
                if 'language' not in voice_config:
                    errors.append("voice_config missing required field: language")
        
        return len(errors) == 0, errors
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'hi', 'ar',
            'nl', 'sv', 'da', 'no', 'fi', 'pl', 'tr', 'th', 'vi', 'id', 'ms'
        ]
    
    def get_supported_voice_providers(self) -> List[str]:
        """Get list of supported voice providers"""
        return ['pyttsx3', 'gtts', 'torch_tts', 'edge_tts']
    
    def get_supported_stt_providers(self) -> List[str]:
        """Get list of supported STT providers"""
        return ['whisper', 'google', 'sphinx', 'azure', 'bing']
    
    def export_config(self, file_path: str) -> bool:
        """
        Export configuration to file
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                'config': self.config,
                'call_script_templates': {
                    template_id: asdict(template) for template_id, template in self.call_script_templates.items()
                },
                'exported_at': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Configuration exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, file_path: str) -> bool:
        """
        Import configuration from file
        
        Args:
            file_path: Path to import file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    import_data = yaml.safe_load(f)
                else:
                    import_data = json.load(f)
            
            if 'config' in import_data:
                self.config.update(import_data['config'])
            
            if 'call_script_templates' in import_data:
                for template_id, template_data in import_data['call_script_templates'].items():
                    # Convert string dates back to datetime objects
                    template_data['created_at'] = datetime.fromisoformat(template_data['created_at'])
                    template_data['updated_at'] = datetime.fromisoformat(template_data['updated_at'])
                    
                    template = CallScriptTemplate(**template_data)
                    self.call_script_templates[template_id] = template
            
            self._save_config()
            self._save_call_script_templates()
            
            logger.info(f"Configuration imported from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False

# Example usage and testing
def main():
    """Example usage of the Configuration Manager"""
    
    # Initialize configuration manager
    config_manager = ConfigManager("test_config.json")
    
    print("Configuration Manager Test:")
    print("=" * 50)
    
    # Get current configuration
    config = config_manager.get_config()
    print(f"System name: {config['system']['name']}")
    print(f"TTS provider: {config['tts']['default_provider']}")
    print(f"STT provider: {config['stt']['default_provider']}")
    
    # Update configuration
    config_manager.update_config('tts', {'default_provider': 'gtts'})
    print(f"Updated TTS provider: {config_manager.get_config_section('tts')['default_provider']}")
    
    # Get call script templates
    templates = config_manager.get_all_call_script_templates()
    print(f"\nAvailable call script templates: {list(templates.keys())}")
    
    # Generate a call script
    script = config_manager.generate_call_script(
        'appointment_booking',
        {
            'voice_settings': {'speed': 1.2, 'emotion': 'excited'},
            'flow_customizations': [
                {
                    'turn_id': 'greeting',
                    'ai_message': 'Hello {contact_name}! Exciting news - I\'m calling to help you schedule an appointment!'
                }
            ]
        }
    )
    
    if script:
        print(f"\nGenerated script name: {script['name']}")
        print(f"Voice speed: {script['voice_config']['speed']}")
        print(f"Number of conversation turns: {len(script['conversation_flow'])}")
    
    # Validate script
    is_valid, errors = config_manager.validate_call_script(script)
    print(f"\nScript validation: {'Valid' if is_valid else 'Invalid'}")
    if errors:
        print(f"Errors: {errors}")
    
    # Export configuration
    config_manager.export_config("exported_config.json")
    print("\nConfiguration exported successfully")
    
    print("\nConfiguration Manager test completed")

if __name__ == "__main__":
    main()
