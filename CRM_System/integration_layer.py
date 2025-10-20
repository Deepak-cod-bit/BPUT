"""
Integration Layer Module - Easy integration with backend/frontend
===============================================================

This module provides a comprehensive integration layer for connecting the AI calling system
with external backends, frontends, and third-party services.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid

# Import AI calling system components
from .ai_calling_agent import AICallingAgent, CallSession, CallStatus
from .speech_synthesis import SpeechSynthesizer, VoiceConfig
from .speech_recognition import SpeechRecognizer, RecognitionConfig
from .nlp_processor import NLPProcessor, ProcessedInput
from .conversation_manager import ConversationManager, ConversationContext
from .analytics_engine import AnalyticsEngine, AnalyticsReport
from .config_manager import ConfigManager, CallScriptTemplate
from .multilingual_support import MultilingualSupport
from .whatsapp_integration import WhatsAppIntegration
from .email_integration import EmailIntegration

logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Integration types"""
    REST_API = "rest_api"
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"
    FILE_SYSTEM = "file_system"

@dataclass
class IntegrationConfig:
    """Integration configuration"""
    integration_type: IntegrationType
    enabled: bool
    config: Dict[str, Any]
    endpoints: List[str] = None
    authentication: Dict[str, Any] = None

@dataclass
class CallRequest:
    """Call request from external system"""
    request_id: str
    phone_number: str
    contact_name: Optional[str]
    call_script_id: str
    customizations: Dict[str, Any] = None
    priority: int = 1
    scheduled_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None

@dataclass
class CallResponse:
    """Call response to external system"""
    request_id: str
    session_id: str
    status: str
    message: str
    estimated_duration: Optional[int] = None
    created_at: datetime = None

@dataclass
class CallUpdate:
    """Call status update"""
    session_id: str
    status: str
    progress: float
    message: str
    timestamp: datetime
    data: Dict[str, Any] = None

class AICallingIntegrationLayer:
    """
    Integration layer for AI calling system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize integration layer
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Initialize AI calling system
        self.ai_calling_agent = AICallingAgent(self.config)
        self.speech_synthesizer = SpeechSynthesizer(self.config)
        self.speech_recognizer = SpeechRecognizer(self.config)
        self.nlp_processor = NLPProcessor(self.config)
        self.conversation_manager = ConversationManager(self.config)
        self.analytics_engine = AnalyticsEngine(self.config)
        self.multilingual_support = MultilingualSupport(self.config)
        
        # Initialize messaging integrations
        self.whatsapp_integration = WhatsAppIntegration(self.config)
        self.email_integration = EmailIntegration(self.config)
        
        # Integration configurations
        self.integration_configs: Dict[str, IntegrationConfig] = {}
        self._load_integration_configs()
        
        # Call tracking
        self.active_calls: Dict[str, CallSession] = {}
        self.call_requests: Dict[str, CallRequest] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'call_started': [],
            'call_ended': [],
            'call_failed': [],
            'call_updated': [],
            'analytics_ready': []
        }
        
        # Webhook endpoints
        self.webhook_endpoints: Dict[str, str] = {}
        
        logger.info("AI Calling Integration Layer initialized successfully")
    
    def _load_integration_configs(self):
        """Load integration configurations"""
        integration_config = self.config.get('integration', {})
        
        # REST API configuration
        if integration_config.get('enable_api', True):
            self.integration_configs['rest_api'] = IntegrationConfig(
                integration_type=IntegrationType.REST_API,
                enabled=True,
                config={
                    'port': integration_config.get('api_port', 8000),
                    'host': integration_config.get('api_host', '0.0.0.0'),
                    'cors_enabled': True,
                    'rate_limiting': True
                },
                endpoints=['/api/v1/calls', '/api/v1/analytics', '/api/v1/health']
            )
        
        # WebSocket configuration
        if integration_config.get('enable_websocket', False):
            self.webhook_endpoints['websocket'] = integration_config.get('websocket_url', 'ws://localhost:8000/ws')
        
        # Webhook configuration
        if integration_config.get('enable_webhook', False):
            self.webhook_endpoints['webhook'] = integration_config.get('webhook_url', '')
        
        # Database configuration
        if integration_config.get('enable_database', False):
            self.integration_configs['database'] = IntegrationConfig(
                integration_type=IntegrationType.DATABASE,
                enabled=True,
                config={
                    'url': integration_config.get('database_url', ''),
                    'type': integration_config.get('database_type', 'postgresql')
                }
            )
    
    def add_event_handler(self, event: str, handler: Callable):
        """Add event handler"""
        if event in self.event_handlers:
            self.event_handlers[event].append(handler)
    
    def _trigger_event(self, event: str, *args, **kwargs):
        """Trigger event handlers"""
        for handler in self.event_handlers.get(event, []):
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event handler for {event}: {e}")
    
    async def initiate_call(self, call_request: CallRequest) -> CallResponse:
        """
        Initiate a call from external request
        
        Args:
            call_request: Call request from external system
            
        Returns:
            CallResponse object
        """
        try:
            # Get call script template
            call_script_template = self.config_manager.get_call_script_template(call_request.call_script_id)
            if not call_script_template:
                return CallResponse(
                    request_id=call_request.request_id,
                    session_id="",
                    status="failed",
                    message=f"Call script template '{call_request.call_script_id}' not found"
                )
            
            # Generate call script with customizations
            call_script = self.config_manager.generate_call_script(
                call_request.call_script_id,
                call_request.customizations or {}
            )
            
            # Create session ID
            session_id = f"call_{uuid.uuid4().hex[:8]}"
            
            # Store call request
            self.call_requests[call_request.request_id] = call_request
            
            # Initiate call
            actual_session_id = await self.ai_calling_agent.initiate_call(
                phone_number=call_request.phone_number,
                call_script=call_script,
                contact_name=call_request.contact_name
            )
            
            # Store active call
            if actual_session_id in self.ai_calling_agent.active_sessions:
                self.active_calls[actual_session_id] = self.ai_calling_agent.active_sessions[actual_session_id]
            
            # Trigger event
            self._trigger_event('call_started', actual_session_id, call_request)
            
            return CallResponse(
                request_id=call_request.request_id,
                session_id=actual_session_id,
                status="initiated",
                message="Call initiated successfully",
                estimated_duration=call_script.get('estimated_duration', 300),
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error initiating call: {e}")
            return CallResponse(
                request_id=call_request.request_id,
                session_id="",
                status="failed",
                message=f"Failed to initiate call: {str(e)}"
            )
    
    async def get_call_status(self, session_id: str) -> Optional[CallUpdate]:
        """
        Get call status update
        
        Args:
            session_id: Call session ID
            
        Returns:
            CallUpdate object or None if not found
        """
        session = self.ai_calling_agent.get_session_status(session_id)
        if not session:
            return None
        
        # Calculate progress
        progress = self._calculate_call_progress(session)
        
        # Get status message
        status_message = self._get_status_message(session)
        
        return CallUpdate(
            session_id=session_id,
            status=session.status.value,
            progress=progress,
            message=status_message,
            timestamp=datetime.now(),
            data={
                'turn_count': len(session.conversation_log),
                'duration': (datetime.now() - session.start_time).total_seconds() if session.start_time else 0,
                'contact_name': session.contact_name,
                'phone_number': session.phone_number
            }
        )
    
    def _calculate_call_progress(self, session: CallSession) -> float:
        """Calculate call progress percentage"""
        if session.status == CallStatus.COMPLETED:
            return 100.0
        elif session.status == CallStatus.FAILED:
            return 0.0
        elif session.status == CallStatus.CANCELLED:
            return 0.0
        else:
            # Estimate progress based on conversation turns
            max_turns = session.call_script.get('max_turns', 10)
            current_turns = len(session.conversation_log)
            return min(90.0, (current_turns / max_turns) * 100)
    
    def _get_status_message(self, session: CallSession) -> str:
        """Get human-readable status message"""
        status_messages = {
            CallStatus.IDLE: "Call is idle",
            CallStatus.INITIATING: "Initiating call...",
            CallStatus.RINGING: "Call is ringing...",
            CallStatus.CONNECTED: "Call connected",
            CallStatus.IN_PROGRESS: "Conversation in progress",
            CallStatus.COMPLETED: "Call completed successfully",
            CallStatus.FAILED: "Call failed",
            CallStatus.CANCELLED: "Call cancelled"
        }
        return status_messages.get(session.status, "Unknown status")
    
    async def cancel_call(self, session_id: str) -> bool:
        """
        Cancel an active call
        
        Args:
            session_id: Call session ID
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        success = await self.ai_calling_agent.cancel_call(session_id)
        if success:
            self._trigger_event('call_cancelled', session_id)
        return success
    
    async def get_call_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get call analytics
        
        Args:
            session_id: Call session ID
            
        Returns:
            Analytics data or None if not found
        """
        session = self.ai_calling_agent.get_session_status(session_id)
        if not session:
            return None
        
        # Get analytics from analytics engine
        analytics_data = await self.analytics_engine.analyze_call(session)
        
        # Trigger event
        self._trigger_event('analytics_ready', session_id, analytics_data)
        
        return analytics_data
    
    async def get_all_active_calls(self) -> List[CallUpdate]:
        """Get all active calls"""
        active_calls = []
        
        for session_id in self.ai_calling_agent.get_all_sessions():
            call_update = await self.get_call_status(session_id)
            if call_update:
                active_calls.append(call_update)
        
        return active_calls
    
    async def get_call_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get call history
        
        Args:
            limit: Maximum number of calls to return
            
        Returns:
            List of call history entries
        """
        # In a real implementation, this would query a database
        # For now, return recent analytics data
        recent_data = self.analytics_engine.historical_data[-limit:]
        
        history = []
        for entry in recent_data:
            history.append({
                'session_id': entry['session_id'],
                'timestamp': entry['timestamp'],
                'duration': entry['metrics']['duration_seconds'],
                'status': 'completed',
                'quality_score': entry['metrics']['call_quality_score'],
                'satisfaction_score': entry['metrics']['customer_satisfaction_score']
            })
        
        return history
    
    def create_call_script_template(self, template_data: Dict[str, Any]) -> bool:
        """
        Create a new call script template
        
        Args:
            template_data: Template data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            template = CallScriptTemplate(**template_data)
            return self.config_manager.create_call_script_template(template)
        except Exception as e:
            logger.error(f"Error creating call script template: {e}")
            return False
    
    def get_call_script_templates(self) -> List[Dict[str, Any]]:
        """Get all call script templates"""
        templates = self.config_manager.get_all_call_script_templates()
        return [asdict(template) for template in templates.values()]
    
    def update_call_script_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update call script template
        
        Args:
            template_id: Template ID
            updates: Updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        return self.config_manager.update_call_script_template(template_id, updates)
    
    def delete_call_script_template(self, template_id: str) -> bool:
        """
        Delete call script template
        
        Args:
            template_id: Template ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.config_manager.delete_call_script_template(template_id)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'active_calls': len(self.active_calls),
            'total_requests': len(self.call_requests)
        }
        
        # Check AI components
        try:
            # Test speech synthesis
            test_audio = await self.speech_synthesizer.synthesize("Test", language="en")
            health['components']['speech_synthesis'] = {
                'status': 'healthy',
                'test_result': f"Generated {len(test_audio)} bytes of audio"
            }
        except Exception as e:
            health['components']['speech_synthesis'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        try:
            # Test NLP processing
            test_input = await self.nlp_processor.process_user_input("Hello, how are you?")
            health['components']['nlp_processing'] = {
                'status': 'healthy',
                'test_result': f"Processed intent: {test_input.intent.intent.value}"
            }
        except Exception as e:
            health['components']['nlp_processing'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        # Check integrations
        for integration_name, integration_config in self.integration_configs.items():
            health['components'][integration_name] = {
                'status': 'enabled' if integration_config.enabled else 'disabled',
                'type': integration_config.integration_type.value
            }
        
        return health
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'calls': {
                'active': len(self.active_calls),
                'total_requests': len(self.call_requests),
                'completed_today': len([c for c in self.analytics_engine.historical_data 
                                      if c['timestamp'].date() == datetime.now().date()])
            },
            'performance': {
                'average_call_duration': 0,
                'success_rate': 0,
                'escalation_rate': 0
            }
        }
        
        # Calculate performance metrics from historical data
        if self.analytics_engine.historical_data:
            recent_data = self.analytics_engine.historical_data[-100:]  # Last 100 calls
            
            durations = [entry['metrics']['duration_seconds'] for entry in recent_data]
            if durations:
                metrics['performance']['average_call_duration'] = sum(durations) / len(durations)
            
            completion_rates = [entry['metrics']['completion_rate'] for entry in recent_data]
            if completion_rates:
                metrics['performance']['success_rate'] = sum(completion_rates) / len(completion_rates)
            
            escalation_counts = [entry['metrics']['escalation_count'] for entry in recent_data]
            if escalation_counts:
                metrics['performance']['escalation_rate'] = sum(escalation_counts) / len(escalation_counts)
        
        return metrics
    
    def export_configuration(self, file_path: str) -> bool:
        """Export system configuration"""
        return self.config_manager.export_config(file_path)
    
    def import_configuration(self, file_path: str) -> bool:
        """Import system configuration"""
        return self.config_manager.import_config(file_path)
    
    # WhatsApp Integration Methods
    async def connect_whatsapp(self) -> bool:
        """Connect to WhatsApp Web"""
        return await self.whatsapp_integration.connect_whatsapp_web()
    
    async def send_whatsapp_message(self, phone_number: str, message: str) -> bool:
        """Send WhatsApp message"""
        return await self.whatsapp_integration.send_message(phone_number, message)
    
    async def send_ai_whatsapp_message(self, phone_number: str, context: str, message_type: str = "general") -> bool:
        """Send AI-generated WhatsApp message"""
        return await self.whatsapp_integration.send_ai_generated_message(phone_number, context, message_type)
    
    async def send_bulk_whatsapp_messages(self, phone_numbers: List[str], message: str) -> Dict[str, bool]:
        """Send bulk WhatsApp messages"""
        return await self.whatsapp_integration.send_bulk_messages(phone_numbers, message)
    
    # Email Integration Methods
    async def send_email(self, to_email: str, subject: str, body: str, is_html: bool = False) -> bool:
        """Send email"""
        return await self.email_integration.send_email(to_email, subject, body, is_html)
    
    async def send_template_email(self, to_email: str, template_id: str, variables: Dict[str, str]) -> bool:
        """Send email using template"""
        return await self.email_integration.send_template_email(to_email, template_id, variables)
    
    async def send_ai_email(self, to_email: str, context: str, email_type: str = "general") -> bool:
        """Send AI-generated email"""
        return await self.email_integration.send_ai_generated_email(to_email, context, email_type)
    
    async def send_bulk_emails(self, email_list: List[Dict[str, str]], template_id: str) -> Dict[str, bool]:
        """Send bulk emails using template"""
        return await self.email_integration.send_bulk_emails(email_list, template_id)
    
    # Multi-channel messaging
    async def send_multi_channel_message(self, 
                                       contact_info: Dict[str, Any], 
                                       message: str,
                                       channels: List[str] = None) -> Dict[str, bool]:
        """
        Send message across multiple channels
        
        Args:
            contact_info: Dictionary with contact information
            message: Message to send
            channels: List of channels to use (whatsapp, email, call)
            
        Returns:
            Dictionary mapping channels to success status
        """
        if channels is None:
            channels = ['whatsapp', 'email']
        
        results = {}
        
        # Send WhatsApp message
        if 'whatsapp' in channels and 'phone_number' in contact_info:
            whatsapp_success = await self.send_whatsapp_message(
                contact_info['phone_number'], 
                message
            )
            results['whatsapp'] = whatsapp_success
        
        # Send email
        if 'email' in channels and 'email' in contact_info:
            email_success = await self.send_email(
                contact_info['email'],
                "Important Message",
                message
            )
            results['email'] = email_success
        
        # Initiate call
        if 'call' in channels and 'phone_number' in contact_info:
            call_request = CallRequest(
                request_id=f"multi_channel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                phone_number=contact_info['phone_number'],
                contact_name=contact_info.get('name'),
                call_script_id='general_info',
                customizations={'message': message}
            )
            call_response = await self.initiate_call(call_request)
            results['call'] = call_response.status == "initiated"
        
        return results
    
    async def cleanup(self):
        """Cleanup resources"""
        # Cancel all active calls
        for session_id in list(self.active_calls.keys()):
            await self.cancel_call(session_id)
        
        # Cleanup AI components
        self.speech_synthesizer.cleanup()
        self.speech_recognizer.cleanup()
        
        # Disconnect messaging services
        await self.whatsapp_integration.disconnect()
        
        logger.info("Integration layer cleanup completed")

# REST API Integration (Flask-based)
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

if FLASK_AVAILABLE:
    def create_flask_app(integration_layer: AICallingIntegrationLayer) -> Flask:
        """Create Flask app for REST API integration"""
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/api/v1/health', methods=['GET'])
        async def health_check():
            """Health check endpoint"""
            health = await integration_layer.get_system_health()
            return jsonify(health)
        
        @app.route('/api/v1/calls', methods=['POST'])
        async def create_call():
            """Create a new call"""
            try:
                data = request.get_json()
                call_request = CallRequest(**data)
                response = await integration_layer.initiate_call(call_request)
                return jsonify(asdict(response))
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @app.route('/api/v1/calls/<session_id>', methods=['GET'])
        async def get_call_status(session_id: str):
            """Get call status"""
            call_update = await integration_layer.get_call_status(session_id)
            if call_update:
                return jsonify(asdict(call_update))
            return jsonify({'error': 'Call not found'}), 404
        
        @app.route('/api/v1/calls/<session_id>/cancel', methods=['POST'])
        async def cancel_call(session_id: str):
            """Cancel a call"""
            success = await integration_layer.cancel_call(session_id)
            return jsonify({'success': success})
        
        @app.route('/api/v1/calls/<session_id>/analytics', methods=['GET'])
        async def get_call_analytics(session_id: str):
            """Get call analytics"""
            analytics = await integration_layer.get_call_analytics(session_id)
            if analytics:
                return jsonify(analytics)
            return jsonify({'error': 'Analytics not available'}), 404
        
        @app.route('/api/v1/calls', methods=['GET'])
        async def get_active_calls():
            """Get all active calls"""
            calls = await integration_layer.get_all_active_calls()
            return jsonify([asdict(call) for call in calls])
        
        @app.route('/api/v1/history', methods=['GET'])
        async def get_call_history():
            """Get call history"""
            limit = request.args.get('limit', 100, type=int)
            history = await integration_layer.get_call_history(limit)
            return jsonify(history)
        
        @app.route('/api/v1/templates', methods=['GET'])
        def get_templates():
            """Get call script templates"""
            templates = integration_layer.get_call_script_templates()
            return jsonify(templates)
        
        @app.route('/api/v1/templates', methods=['POST'])
        def create_template():
            """Create call script template"""
            data = request.get_json()
            success = integration_layer.create_call_script_template(data)
            return jsonify({'success': success})
        
        @app.route('/api/v1/templates/<template_id>', methods=['PUT'])
        def update_template(template_id: str):
            """Update call script template"""
            data = request.get_json()
            success = integration_layer.update_call_script_template(template_id, data)
            return jsonify({'success': success})
        
        @app.route('/api/v1/templates/<template_id>', methods=['DELETE'])
        def delete_template(template_id: str):
            """Delete call script template"""
            success = integration_layer.delete_call_script_template(template_id)
            return jsonify({'success': success})
        
        @app.route('/api/v1/metrics', methods=['GET'])
        async def get_metrics():
            """Get system metrics"""
            metrics = await integration_layer.get_system_metrics()
            return jsonify(metrics)
        
        return app

# Example usage and testing
async def main():
    """Example usage of the Integration Layer"""
    
    # Initialize integration layer
    integration = AICallingIntegrationLayer()
    
    print("AI Calling Integration Layer Test:")
    print("=" * 50)
    
    # Test system health
    health = await integration.get_system_health()
    print(f"System Status: {health['status']}")
    print(f"Active Calls: {health['active_calls']}")
    
    # Test call script templates
    templates = integration.get_call_script_templates()
    print(f"\nAvailable Templates: {len(templates)}")
    for template in templates[:3]:  # Show first 3
        print(f"- {template['name']} ({template['template_id']})")
    
    # Test creating a call request
    call_request = CallRequest(
        request_id="test_request_001",
        phone_number="+1234567890",
        contact_name="John Doe",
        call_script_id="appointment_booking",
        customizations={
            'voice_settings': {'speed': 1.1},
            'contact_name': 'John Doe'
        }
    )
    
    print(f"\nCreating call request: {call_request.request_id}")
    
    # Note: In a real scenario, you would initiate the call
    # For testing, we'll just show the structure
    print("Call request structure:")
    print(f"- Phone: {call_request.phone_number}")
    print(f"- Contact: {call_request.contact_name}")
    print(f"- Script: {call_request.call_script_id}")
    
    # Test system metrics
    metrics = await integration.get_system_metrics()
    print(f"\nSystem Metrics:")
    print(f"- Active Calls: {metrics['calls']['active']}")
    print(f"- Total Requests: {metrics['calls']['total_requests']}")
    print(f"- Success Rate: {metrics['performance']['success_rate']:.2f}")
    
    # Test configuration export
    export_success = integration.export_configuration("test_export.json")
    print(f"\nConfiguration Export: {'Success' if export_success else 'Failed'}")
    
    # Cleanup
    await integration.cleanup()
    
    print("\nIntegration Layer test completed")

if __name__ == "__main__":
    asyncio.run(main())
