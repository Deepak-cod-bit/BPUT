"""
Conversation Manager Module - Conversation flow management
========================================================

This module manages conversation flows, state transitions, and context
maintenance for AI calling interactions.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import uuid

from .nlp_processor import ProcessedInput, IntentType, SentimentType

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """Conversation states"""
    INITIAL = "initial"
    GREETING = "greeting"
    MAIN_TASK = "main_task"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    HANDLING_OBJECTION = "handling_objection"
    CLOSING = "closing"
    COMPLETED = "completed"
    FAILED = "failed"

class FlowAction(Enum):
    """Flow actions"""
    CONTINUE = "continue"
    REPEAT = "repeat"
    CLARIFY = "clarify"
    ESCALATE = "escalate"
    END_CALL = "end_call"
    TRANSFER = "transfer"

@dataclass
class ConversationContext:
    """Conversation context data"""
    session_id: str
    contact_name: Optional[str]
    phone_number: str
    current_state: ConversationState
    previous_state: Optional[ConversationState]
    state_history: List[ConversationState]
    conversation_log: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    collected_data: Dict[str, Any]
    start_time: datetime
    last_activity: datetime
    turn_count: int
    success_indicators: List[str]
    failure_indicators: List[str]

@dataclass
class FlowDecision:
    """Flow decision result"""
    next_state: ConversationState
    action: FlowAction
    response_template: str
    confidence: float
    reasoning: str
    requires_human: bool
    escalation_reason: Optional[str]

class ConversationManager:
    """
    Manages conversation flows and state transitions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize conversation manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.conversation_config = config.get('conversation', {})
        
        # Active conversations
        self.active_conversations: Dict[str, ConversationContext] = {}
        
        # Flow definitions
        self.flow_definitions = self._load_flow_definitions()
        
        # State transition rules
        self.transition_rules = self._load_transition_rules()
        
        # Response templates
        self.response_templates = self._load_response_templates()
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'state_changed': [],
            'flow_decision': [],
            'escalation_required': [],
            'conversation_completed': []
        }
        
        logger.info("Conversation Manager initialized successfully")
    
    def _load_flow_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load conversation flow definitions"""
        return {
            'appointment_booking': {
                'initial_state': ConversationState.GREETING,
                'states': {
                    ConversationState.GREETING: {
                        'purpose': 'Welcome and introduce purpose',
                        'expected_intents': [IntentType.GREETING, IntentType.APPOINTMENT],
                        'success_conditions': ['user_acknowledges', 'user_engages'],
                        'next_states': [ConversationState.MAIN_TASK]
                    },
                    ConversationState.MAIN_TASK: {
                        'purpose': 'Collect appointment details',
                        'expected_intents': [IntentType.APPOINTMENT, IntentType.CONFIRMATION],
                        'success_conditions': ['date_provided', 'time_provided'],
                        'next_states': [ConversationState.CONFIRMATION, ConversationState.CLARIFICATION]
                    },
                    ConversationState.CLARIFICATION: {
                        'purpose': 'Clarify unclear information',
                        'expected_intents': [IntentType.APPOINTMENT, IntentType.QUESTION],
                        'success_conditions': ['information_clarified'],
                        'next_states': [ConversationState.MAIN_TASK, ConversationState.CONFIRMATION]
                    },
                    ConversationState.CONFIRMATION: {
                        'purpose': 'Confirm appointment details',
                        'expected_intents': [IntentType.CONFIRMATION, IntentType.CANCELLATION],
                        'success_conditions': ['user_confirms'],
                        'next_states': [ConversationState.CLOSING, ConversationState.MAIN_TASK]
                    },
                    ConversationState.CLOSING: {
                        'purpose': 'End conversation positively',
                        'expected_intents': [IntentType.END_CALL, IntentType.GREETING],
                        'success_conditions': ['user_satisfied'],
                        'next_states': [ConversationState.COMPLETED]
                    }
                }
            },
            'feedback_collection': {
                'initial_state': ConversationState.GREETING,
                'states': {
                    ConversationState.GREETING: {
                        'purpose': 'Welcome and explain feedback purpose',
                        'expected_intents': [IntentType.GREETING, IntentType.FEEDBACK],
                        'success_conditions': ['user_acknowledges'],
                        'next_states': [ConversationState.MAIN_TASK]
                    },
                    ConversationState.MAIN_TASK: {
                        'purpose': 'Collect detailed feedback',
                        'expected_intents': [IntentType.FEEDBACK, IntentType.COMPLAINT],
                        'success_conditions': ['feedback_provided'],
                        'next_states': [ConversationState.CLOSING, ConversationState.CLARIFICATION]
                    },
                    ConversationState.CLOSING: {
                        'purpose': 'Thank user and end conversation',
                        'expected_intents': [IntentType.END_CALL],
                        'success_conditions': ['user_satisfied'],
                        'next_states': [ConversationState.COMPLETED]
                    }
                }
            }
        }
    
    def _load_transition_rules(self) -> Dict[Tuple[ConversationState, IntentType], FlowDecision]:
        """Load state transition rules"""
        return {
            # Greeting state transitions
            (ConversationState.GREETING, IntentType.GREETING): FlowDecision(
                next_state=ConversationState.MAIN_TASK,
                action=FlowAction.CONTINUE,
                response_template="greeting_acknowledged",
                confidence=0.9,
                reasoning="User acknowledged greeting",
                requires_human=False,
                escalation_reason=None
            ),
            (ConversationState.GREETING, IntentType.APPOINTMENT): FlowDecision(
                next_state=ConversationState.MAIN_TASK,
                action=FlowAction.CONTINUE,
                response_template="direct_to_main_task",
                confidence=0.8,
                reasoning="User went directly to main task",
                requires_human=False,
                escalation_reason=None
            ),
            
            # Main task state transitions
            (ConversationState.MAIN_TASK, IntentType.APPOINTMENT): FlowDecision(
                next_state=ConversationState.CONFIRMATION,
                action=FlowAction.CONTINUE,
                response_template="appointment_details_collected",
                confidence=0.8,
                reasoning="Appointment details provided",
                requires_human=False,
                escalation_reason=None
            ),
            (ConversationState.MAIN_TASK, IntentType.QUESTION): FlowDecision(
                next_state=ConversationState.CLARIFICATION,
                action=FlowAction.CLARIFY,
                response_template="clarification_needed",
                confidence=0.7,
                reasoning="User needs clarification",
                requires_human=False,
                escalation_reason=None
            ),
            
            # Confirmation state transitions
            (ConversationState.CONFIRMATION, IntentType.CONFIRMATION): FlowDecision(
                next_state=ConversationState.CLOSING,
                action=FlowAction.CONTINUE,
                response_template="confirmation_acknowledged",
                confidence=0.9,
                reasoning="User confirmed details",
                requires_human=False,
                escalation_reason=None
            ),
            (ConversationState.CONFIRMATION, IntentType.CANCELLATION): FlowDecision(
                next_state=ConversationState.MAIN_TASK,
                action=FlowAction.CONTINUE,
                response_template="cancellation_handled",
                confidence=0.8,
                reasoning="User wants to change details",
                requires_human=False,
                escalation_reason=None
            ),
            
            # End call transitions
            (ConversationState.CLOSING, IntentType.END_CALL): FlowDecision(
                next_state=ConversationState.COMPLETED,
                action=FlowAction.END_CALL,
                response_template="call_ended",
                confidence=0.9,
                reasoning="User wants to end call",
                requires_human=False,
                escalation_reason=None
            ),
            
            # Complaint handling
            (ConversationState.MAIN_TASK, IntentType.COMPLAINT): FlowDecision(
                next_state=ConversationState.HANDLING_OBJECTION,
                action=FlowAction.ESCALATE,
                response_template="complaint_acknowledged",
                confidence=0.8,
                reasoning="User has complaint - may need human intervention",
                requires_human=True,
                escalation_reason="User complaint detected"
            )
        }
    
    def _load_response_templates(self) -> Dict[str, str]:
        """Load response templates"""
        return {
            'greeting_acknowledged': "Thank you! I'm here to help you with your appointment. What time would work best for you?",
            'direct_to_main_task': "I'd be happy to help you schedule an appointment. What time works for you?",
            'appointment_details_collected': "Perfect! I have your appointment details. Let me confirm: {appointment_details}. Is this correct?",
            'clarification_needed': "I want to make sure I understand correctly. Could you please clarify {unclear_aspect}?",
            'confirmation_acknowledged': "Excellent! Your appointment is confirmed for {appointment_time}. I'll send you a confirmation email shortly.",
            'cancellation_handled': "No problem at all. Let's find a better time for you. What would work better?",
            'call_ended': "Thank you for your time! Have a great day!",
            'complaint_acknowledged': "I'm sorry to hear about this issue. Let me connect you with someone who can help resolve this immediately.",
            'escalation_message': "I'm transferring you to a human agent who can better assist you with this matter.",
            'fallback_response': "I understand. Let me help you with that. Could you please tell me more about what you need?"
        }
    
    def add_callback(self, event: str, callback: Callable):
        """Add conversation callback"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """Trigger conversation callbacks"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for event {event}: {e}")
    
    def start_conversation(self, 
                          session_id: str,
                          contact_name: Optional[str],
                          phone_number: str,
                          flow_type: str = 'appointment_booking') -> ConversationContext:
        """
        Start a new conversation
        
        Args:
            session_id: Unique session identifier
            contact_name: Contact name
            phone_number: Phone number
            flow_type: Type of conversation flow
            
        Returns:
            ConversationContext object
        """
        if flow_type not in self.flow_definitions:
            raise ValueError(f"Unknown flow type: {flow_type}")
        
        flow_def = self.flow_definitions[flow_type]
        initial_state = ConversationState(flow_def['initial_state'])
        
        context = ConversationContext(
            session_id=session_id,
            contact_name=contact_name,
            phone_number=phone_number,
            current_state=initial_state,
            previous_state=None,
            state_history=[initial_state],
            conversation_log=[],
            user_preferences={},
            collected_data={},
            start_time=datetime.now(),
            last_activity=datetime.now(),
            turn_count=0,
            success_indicators=[],
            failure_indicators=[]
        )
        
        self.active_conversations[session_id] = context
        
        logger.info(f"Started conversation {session_id} with flow {flow_type}")
        return context
    
    def process_turn(self, 
                    session_id: str, 
                    processed_input: ProcessedInput) -> FlowDecision:
        """
        Process a conversation turn and determine next action
        
        Args:
            session_id: Session identifier
            processed_input: Processed user input
            
        Returns:
            FlowDecision object
        """
        if session_id not in self.active_conversations:
            raise ValueError(f"Unknown session: {session_id}")
        
        context = self.active_conversations[session_id]
        
        # Update context
        context.turn_count += 1
        context.last_activity = datetime.now()
        
        # Add to conversation log
        turn_log = {
            'turn_number': context.turn_count,
            'timestamp': datetime.now().isoformat(),
            'user_input': processed_input.original_text,
            'intent': processed_input.intent.intent.value,
            'sentiment': processed_input.sentiment.sentiment.value,
            'confidence': processed_input.confidence
        }
        context.conversation_log.append(turn_log)
        
        # Determine flow decision
        flow_decision = self._determine_flow_decision(context, processed_input)
        
        # Update context state
        if flow_decision.next_state != context.current_state:
            context.previous_state = context.current_state
            context.current_state = flow_decision.next_state
            context.state_history.append(flow_decision.next_state)
            
            self._trigger_callback('state_changed', context, flow_decision)
        
        # Update collected data
        self._update_collected_data(context, processed_input)
        
        # Check for escalation
        if flow_decision.requires_human:
            self._trigger_callback('escalation_required', context, flow_decision)
        
        # Check for completion
        if flow_decision.next_state == ConversationState.COMPLETED:
            self._trigger_callback('conversation_completed', context)
        
        self._trigger_callback('flow_decision', context, flow_decision)
        
        return flow_decision
    
    def _determine_flow_decision(self, 
                                context: ConversationContext, 
                                processed_input: ProcessedInput) -> FlowDecision:
        """Determine flow decision based on current state and input"""
        
        # Check for direct state transitions
        transition_key = (context.current_state, processed_input.intent.intent)
        if transition_key in self.transition_rules:
            decision = self.transition_rules[transition_key]
            
            # Customize response template
            decision.response_template = self._customize_response_template(
                decision.response_template, 
                context, 
                processed_input
            )
            
            return decision
        
        # Check for sentiment-based transitions
        if processed_input.sentiment.sentiment == SentimentType.NEGATIVE:
            return FlowDecision(
                next_state=ConversationState.HANDLING_OBJECTION,
                action=FlowAction.ESCALATE,
                response_template="complaint_acknowledged",
                confidence=0.7,
                reasoning="Negative sentiment detected",
                requires_human=True,
                escalation_reason="Negative sentiment"
            )
        
        # Check for low confidence
        if processed_input.confidence < 0.3:
            return FlowDecision(
                next_state=ConversationState.CLARIFICATION,
                action=FlowAction.CLARIFY,
                response_template="clarification_needed",
                confidence=0.5,
                reasoning="Low confidence in understanding",
                requires_human=False,
                escalation_reason=None
            )
        
        # Default fallback
        return FlowDecision(
            next_state=context.current_state,
            action=FlowAction.CONTINUE,
            response_template="fallback_response",
            confidence=0.3,
            reasoning="No specific transition rule matched",
            requires_human=False,
            escalation_reason=None
        )
    
    def _customize_response_template(self, 
                                   template_name: str, 
                                   context: ConversationContext,
                                   processed_input: ProcessedInput) -> str:
        """Customize response template with context data"""
        template = self.response_templates.get(template_name, template_name)
        
        # Replace placeholders
        replacements = {
            '{contact_name}': context.contact_name or 'there',
            '{appointment_details}': self._format_appointment_details(context.collected_data),
            '{appointment_time}': context.collected_data.get('appointment_time', 'the scheduled time'),
            '{unclear_aspect}': 'what you just said'
        }
        
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, str(value))
        
        return template
    
    def _format_appointment_details(self, collected_data: Dict[str, Any]) -> str:
        """Format appointment details for display"""
        details = []
        
        if 'date' in collected_data:
            details.append(f"date: {collected_data['date']}")
        if 'time' in collected_data:
            details.append(f"time: {collected_data['time']}")
        if 'service' in collected_data:
            details.append(f"service: {collected_data['service']}")
        
        return ', '.join(details) if details else 'the details we discussed'
    
    def _update_collected_data(self, 
                              context: ConversationContext, 
                              processed_input: ProcessedInput):
        """Update collected data based on user input"""
        
        # Extract entities and add to collected data
        for entity_type, entities in processed_input.entities.items():
            if entity_type not in context.collected_data:
                context.collected_data[entity_type] = []
            context.collected_data[entity_type].extend(entities)
        
        # Extract specific information based on intent
        if processed_input.intent.intent == IntentType.APPOINTMENT:
            # Extract time and date information
            text = processed_input.cleaned_text.lower()
            
            # Simple time extraction (in real implementation, use more sophisticated NLP)
            time_patterns = [
                r'(\d{1,2}):?(\d{2})?\s*(am|pm)',
                r'(\d{1,2})\s*(am|pm)',
                r'(morning|afternoon|evening)'
            ]
            
            for pattern in time_patterns:
                import re
                match = re.search(pattern, text)
                if match:
                    context.collected_data['time'] = match.group(0)
                    break
            
            # Simple date extraction
            date_patterns = [
                r'(tomorrow|today|yesterday)',
                r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
                r'(\d{1,2})/(\d{1,2})/(\d{4})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match:
                    context.collected_data['date'] = match.group(0)
                    break
        
        # Update user preferences based on sentiment
        if processed_input.sentiment.sentiment == SentimentType.POSITIVE:
            context.user_preferences['satisfaction'] = 'high'
        elif processed_input.sentiment.sentiment == SentimentType.NEGATIVE:
            context.user_preferences['satisfaction'] = 'low'
    
    def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context by session ID"""
        return self.active_conversations.get(session_id)
    
    def end_conversation(self, session_id: str, reason: str = "completed") -> Optional[ConversationContext]:
        """End a conversation"""
        if session_id in self.active_conversations:
            context = self.active_conversations[session_id]
            context.current_state = ConversationState.COMPLETED
            del self.active_conversations[session_id]
            
            logger.info(f"Ended conversation {session_id}: {reason}")
            return context
        return None
    
    def get_conversation_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation summary"""
        context = self.get_conversation_context(session_id)
        if not context:
            return None
        
        return {
            'session_id': context.session_id,
            'contact_name': context.contact_name,
            'phone_number': context.phone_number,
            'duration': (context.last_activity - context.start_time).total_seconds(),
            'turn_count': context.turn_count,
            'final_state': context.current_state.value,
            'state_history': [state.value for state in context.state_history],
            'collected_data': context.collected_data,
            'user_preferences': context.user_preferences,
            'conversation_log': context.conversation_log
        }
    
    def get_active_conversations(self) -> List[str]:
        """Get list of active conversation session IDs"""
        return list(self.active_conversations.keys())

# Example usage and testing
async def main():
    """Example usage of the Conversation Manager"""
    
    config = {
        'conversation': {
            'max_turns': 20,
            'timeout_seconds': 300
        }
    }
    
    manager = ConversationManager(config)
    
    # Add callbacks
    def on_state_changed(context, decision):
        print(f"State changed to: {context.current_state.value}")
    
    def on_escalation_required(context, decision):
        print(f"Escalation required: {decision.escalation_reason}")
    
    manager.add_callback('state_changed', on_state_changed)
    manager.add_callback('escalation_required', on_escalation_required)
    
    # Start a conversation
    session_id = "test_session_001"
    context = manager.start_conversation(
        session_id=session_id,
        contact_name="John Doe",
        phone_number="+1234567890",
        flow_type="appointment_booking"
    )
    
    print(f"Started conversation: {session_id}")
    print(f"Initial state: {context.current_state.value}")
    
    # Simulate conversation turns
    from .nlp_processor import ProcessedInput, IntentResult, SentimentResult, IntentType, SentimentType
    
    test_turns = [
        ("Hello, I'd like to book an appointment", IntentType.GREETING, SentimentType.POSITIVE),
        ("Tomorrow at 2 PM works for me", IntentType.APPOINTMENT, SentimentType.POSITIVE),
        ("Yes, that's correct", IntentType.CONFIRMATION, SentimentType.POSITIVE),
        ("Thank you, goodbye!", IntentType.END_CALL, SentimentType.POSITIVE)
    ]
    
    for i, (text, intent, sentiment) in enumerate(test_turns):
        print(f"\nTurn {i+1}: '{text}'")
        
        # Create processed input
        processed_input = ProcessedInput(
            original_text=text,
            cleaned_text=text.lower(),
            intent=IntentResult(intent, 0.8, {}, []),
            sentiment=SentimentResult(sentiment, 0.8, {}, "neutral"),
            entities={},
            keywords=[],
            language="en",
            confidence=0.8
        )
        
        # Process turn
        decision = manager.process_turn(session_id, processed_input)
        
        print(f"Decision: {decision.action.value}")
        print(f"Next state: {decision.next_state.value}")
        print(f"Response: {decision.response_template}")
        print(f"Requires human: {decision.requires_human}")
    
    # Get conversation summary
    summary = manager.get_conversation_summary(session_id)
    print(f"\nConversation Summary:")
    print(f"Duration: {summary['duration']:.1f} seconds")
    print(f"Turns: {summary['turn_count']}")
    print(f"Final state: {summary['final_state']}")
    print(f"Collected data: {summary['collected_data']}")
    
    # End conversation
    manager.end_conversation(session_id, "test completed")
    print(f"\nConversation ended")

if __name__ == "__main__":
    asyncio.run(main())
