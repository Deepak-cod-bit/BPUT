"""
NLP Processor Module - Natural Language Processing functionality
==============================================================

This module provides natural language processing capabilities including:
- Intent recognition
- Sentiment analysis
- Entity extraction
- Response generation
- Conversation management
"""

import asyncio
import logging
import re
import json
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentiment
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Intent types for conversation management"""
    GREETING = "greeting"
    APPOINTMENT = "appointment"
    CONFIRMATION = "confirmation"
    CANCELLATION = "cancellation"
    RESCHEDULING = "rescheduling"
    FEEDBACK = "feedback"
    COMPLAINT = "complaint"
    QUESTION = "question"
    END_CALL = "end_call"
    UNKNOWN = "unknown"

class SentimentType(Enum):
    """Sentiment types"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

@dataclass
class IntentResult:
    """Intent recognition result"""
    intent: IntentType
    confidence: float
    entities: Dict[str, Any]
    keywords: List[str]

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    sentiment: SentimentType
    confidence: float
    scores: Dict[str, float]
    emotional_tone: str

@dataclass
class ProcessedInput:
    """Processed user input"""
    original_text: str
    cleaned_text: str
    intent: IntentResult
    sentiment: SentimentResult
    entities: Dict[str, Any]
    keywords: List[str]
    language: str
    confidence: float

class NLPProcessor:
    """
    Natural Language Processor for conversation management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NLP processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.nlp_config = config.get('nlp', {})
        
        # Initialize NLP components
        self._initialize_components()
        
        # Intent patterns and keywords
        self.intent_patterns = self._load_intent_patterns()
        
        # Response templates
        self.response_templates = self._load_response_templates()
        
        logger.info("NLP Processor initialized successfully")
    
    def _initialize_components(self):
        """Initialize NLP components"""
        
        # Initialize NLTK
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data
                self._download_nltk_data()
                
                self.sia = SentimentIntensityAnalyzer()
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
                
                logger.info("NLTK components initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK: {e}")
        
        # Initialize TextBlob
        if TEXTBLOB_AVAILABLE:
            logger.info("TextBlob initialized")
        
        # Initialize VADER
        if VADER_AVAILABLE:
            self.vader_analyzer = VaderSentiment()
            logger.info("VADER sentiment analyzer initialized")
        
        # Initialize spaCy
        if SPACY_AVAILABLE:
            try:
                model_name = self.nlp_config.get('spacy_model', 'en_core_web_sm')
                self.nlp = spacy.load(model_name)
                logger.info(f"spaCy model '{model_name}' loaded")
            except OSError:
                logger.warning("spaCy model not found, using basic processing")
                self.nlp = None
            except Exception as e:
                logger.warning(f"Failed to initialize spaCy: {e}")
                self.nlp = None
        
        # Initialize Transformers
        if TRANSFORMERS_AVAILABLE:
            try:
                self._initialize_transformers()
                logger.info("Transformers models initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Transformers: {e}")
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
    
    def _initialize_transformers(self):
        """Initialize Transformers models"""
        try:
            # Sentiment analysis model
            sentiment_model = self.nlp_config.get('sentiment_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=sentiment_model,
                return_all_scores=True
            )
            
            # Intent classification model (if available)
            intent_model = self.nlp_config.get('intent_model')
            if intent_model:
                self.intent_pipeline = pipeline(
                    "text-classification",
                    model=intent_model,
                    return_all_scores=True
                )
            else:
                self.intent_pipeline = None
                
        except Exception as e:
            logger.warning(f"Failed to load Transformers models: {e}")
            self.sentiment_pipeline = None
            self.intent_pipeline = None
    
    def _load_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """Load intent recognition patterns"""
        return {
            IntentType.GREETING: [
                r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                r'\b(how are you|how do you do)\b'
            ],
            IntentType.APPOINTMENT: [
                r'\b(appointment|meeting|schedule|book|reserve)\b',
                r'\b(when|what time|available|free)\b'
            ],
            IntentType.CONFIRMATION: [
                r'\b(yes|yeah|sure|okay|ok|confirm|confirmed)\b',
                r'\b(that works|sounds good|perfect|great)\b'
            ],
            IntentType.CANCELLATION: [
                r'\b(cancel|cancelled|no|not|unable|can\'t)\b',
                r'\b(not available|busy|conflict)\b'
            ],
            IntentType.RESCHEDULING: [
                r'\b(reschedule|change|move|different time)\b',
                r'\b(another time|later|earlier)\b'
            ],
            IntentType.FEEDBACK: [
                r'\b(feedback|review|rating|experience)\b',
                r'\b(how was|what did you think)\b'
            ],
            IntentType.COMPLAINT: [
                r'\b(problem|issue|complaint|wrong|bad|terrible)\b',
                r'\b(not working|broken|disappointed)\b'
            ],
            IntentType.QUESTION: [
                r'\b(what|how|when|where|why|who)\b',
                r'\b(can you|could you|would you)\b'
            ],
            IntentType.END_CALL: [
                r'\b(bye|goodbye|see you|talk to you later)\b',
                r'\b(that\'s all|nothing else|thank you)\b'
            ]
        }
    
    def _load_response_templates(self) -> Dict[IntentType, List[str]]:
        """Load response templates for different intents"""
        return {
            IntentType.GREETING: [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Good to hear from you! How may I assist?"
            ],
            IntentType.APPOINTMENT: [
                "I'd be happy to help you with scheduling. What time works best for you?",
                "Let me check our availability. When would you like to meet?",
                "I can help you book an appointment. What's your preferred time?"
            ],
            IntentType.CONFIRMATION: [
                "Perfect! I'll confirm that for you.",
                "Great! I've got that noted.",
                "Excellent! That's all set up."
            ],
            IntentType.CANCELLATION: [
                "I understand. Let me help you with that.",
                "No problem at all. I'll take care of it.",
                "I'll cancel that for you right away."
            ],
            IntentType.RESCHEDULING: [
                "I can help you reschedule. What time would work better?",
                "No problem! When would you prefer to meet?",
                "Let me help you find a better time."
            ],
            IntentType.FEEDBACK: [
                "I'd love to hear your feedback!",
                "Please tell me about your experience.",
                "Your feedback is very important to us."
            ],
            IntentType.COMPLAINT: [
                "I'm sorry to hear about this issue. Let me help resolve it.",
                "I understand your concern. Let me see what I can do.",
                "I apologize for the inconvenience. How can I make this right?"
            ],
            IntentType.QUESTION: [
                "I'd be happy to answer your question.",
                "What would you like to know?",
                "I'm here to help with any questions you have."
            ],
            IntentType.END_CALL: [
                "Thank you for calling! Have a great day!",
                "It was great talking with you. Take care!",
                "Thanks for your time. Goodbye!"
            ]
        }
    
    async def process_user_input(self, 
                               text: str, 
                               context: Optional[List[Dict]] = None) -> ProcessedInput:
        """
        Process user input with full NLP pipeline
        
        Args:
            text: User input text
            context: Previous conversation context
            
        Returns:
            ProcessedInput object
        """
        try:
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            # Detect language
            language = self._detect_language(cleaned_text)
            
            # Extract entities
            entities = await self._extract_entities(cleaned_text)
            
            # Extract keywords
            keywords = self._extract_keywords(cleaned_text)
            
            # Recognize intent
            intent_result = await self._recognize_intent(cleaned_text, context)
            
            # Analyze sentiment
            sentiment_result = await self._analyze_sentiment(cleaned_text)
            
            # Calculate overall confidence
            confidence = (intent_result.confidence + sentiment_result.confidence) / 2
            
            return ProcessedInput(
                original_text=text,
                cleaned_text=cleaned_text,
                intent=intent_result,
                sentiment=sentiment_result,
                entities=entities,
                keywords=keywords,
                language=language,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            # Return fallback result
            return ProcessedInput(
                original_text=text,
                cleaned_text=text,
                intent=IntentResult(IntentType.UNKNOWN, 0.0, {}, []),
                sentiment=SentimentResult(SentimentType.NEUTRAL, 0.5, {}, "neutral"),
                entities={},
                keywords=[],
                language="en",
                confidence=0.0
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        return text
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                return blob.detect_language()
            except:
                pass
        
        # Fallback to English
        return "en"
    
    async def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text"""
        entities = {}
        
        # Use spaCy if available
        if SPACY_AVAILABLE and self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entity_type = ent.label_
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].append(ent.text)
            except Exception as e:
                logger.warning(f"spaCy entity extraction failed: {e}")
        
        # Use NLTK as fallback
        elif NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        entity_type = chunk.label()
                        if entity_type not in entities:
                            entities[entity_type] = []
                        entities[entity_type].append(' '.join([token for token, pos in chunk.leaves()]))
            except Exception as e:
                logger.warning(f"NLTK entity extraction failed: {e}")
        
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        keywords = []
        
        if NLTK_AVAILABLE:
            try:
                # Tokenize and remove stop words
                tokens = word_tokenize(text)
                tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
                
                # Lemmatize tokens
                lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
                
                # Get most common words
                from collections import Counter
                word_freq = Counter(lemmatized)
                keywords = [word for word, freq in word_freq.most_common(5)]
                
            except Exception as e:
                logger.warning(f"Keyword extraction failed: {e}")
        
        return keywords
    
    async def _recognize_intent(self, text: str, context: Optional[List[Dict]] = None) -> IntentResult:
        """Recognize intent from text"""
        
        # Try Transformers model first
        if TRANSFORMERS_AVAILABLE and self.intent_pipeline:
            try:
                result = self.intent_pipeline(text)
                if result and len(result) > 0:
                    best_intent = result[0]
                    intent_name = best_intent['label']
                    confidence = best_intent['score']
                    
                    # Map to our intent types
                    intent_type = self._map_intent_name(intent_name)
                    return IntentResult(intent_type, confidence, {}, [])
            except Exception as e:
                logger.warning(f"Transformers intent recognition failed: {e}")
        
        # Fallback to pattern matching
        return self._pattern_based_intent_recognition(text)
    
    def _map_intent_name(self, intent_name: str) -> IntentType:
        """Map intent name to IntentType enum"""
        intent_mapping = {
            'greeting': IntentType.GREETING,
            'appointment': IntentType.APPOINTMENT,
            'confirmation': IntentType.CONFIRMATION,
            'cancellation': IntentType.CANCELLATION,
            'rescheduling': IntentType.RESCHEDULING,
            'feedback': IntentType.FEEDBACK,
            'complaint': IntentType.COMPLAINT,
            'question': IntentType.QUESTION,
            'end_call': IntentType.END_CALL
        }
        return intent_mapping.get(intent_name.lower(), IntentType.UNKNOWN)
    
    def _pattern_based_intent_recognition(self, text: str) -> IntentResult:
        """Pattern-based intent recognition"""
        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0
        matched_keywords = []
        
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    confidence = len(matches) / len(patterns)
                    if confidence > best_confidence:
                        best_intent = intent_type
                        best_confidence = confidence
                        matched_keywords = matches
        
        return IntentResult(best_intent, best_confidence, {}, matched_keywords)
    
    async def _analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of text"""
        
        # Try Transformers model first
        if TRANSFORMERS_AVAILABLE and self.sentiment_pipeline:
            try:
                result = self.sentiment_pipeline(text)
                if result and len(result) > 0:
                    scores = {item['label']: item['score'] for item in result[0]}
                    
                    # Find best sentiment
                    best_sentiment = max(scores.items(), key=lambda x: x[1])
                    sentiment_type = self._map_sentiment_label(best_sentiment[0])
                    confidence = best_sentiment[1]
                    
                    return SentimentResult(sentiment_type, confidence, scores, "neutral")
            except Exception as e:
                logger.warning(f"Transformers sentiment analysis failed: {e}")
        
        # Fallback to VADER
        if VADER_AVAILABLE:
            try:
                scores = self.vader_analyzer.polarity_scores(text)
                
                # Determine sentiment
                if scores['compound'] >= 0.05:
                    sentiment_type = SentimentType.POSITIVE
                elif scores['compound'] <= -0.05:
                    sentiment_type = SentimentType.NEGATIVE
                else:
                    sentiment_type = SentimentType.NEUTRAL
                
                confidence = abs(scores['compound'])
                
                return SentimentResult(sentiment_type, confidence, scores, "neutral")
            except Exception as e:
                logger.warning(f"VADER sentiment analysis failed: {e}")
        
        # Fallback to TextBlob
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    sentiment_type = SentimentType.POSITIVE
                elif polarity < -0.1:
                    sentiment_type = SentimentType.NEGATIVE
                else:
                    sentiment_type = SentimentType.NEUTRAL
                
                confidence = abs(polarity)
                
                return SentimentResult(sentiment_type, confidence, {"polarity": polarity}, "neutral")
            except Exception as e:
                logger.warning(f"TextBlob sentiment analysis failed: {e}")
        
        # Default fallback
        return SentimentResult(SentimentType.NEUTRAL, 0.5, {}, "neutral")
    
    def _map_sentiment_label(self, label: str) -> SentimentType:
        """Map sentiment label to SentimentType enum"""
        label_mapping = {
            'positive': SentimentType.POSITIVE,
            'negative': SentimentType.NEGATIVE,
            'neutral': SentimentType.NEUTRAL,
            'mixed': SentimentType.MIXED
        }
        return label_mapping.get(label.lower(), SentimentType.NEUTRAL)
    
    async def generate_response(self, 
                              base_response: str,
                              context: Optional[List[Dict]] = None,
                              turn_config: Optional[Dict] = None) -> str:
        """
        Generate enhanced response based on context and configuration
        
        Args:
            base_response: Base response template
            context: Conversation context
            turn_config: Turn-specific configuration
            
        Returns:
            Enhanced response string
        """
        try:
            # Start with base response
            response = base_response
            
            # Apply personalization based on context
            if context:
                response = self._personalize_response(response, context)
            
            # Apply turn-specific enhancements
            if turn_config:
                response = self._apply_turn_enhancements(response, turn_config)
            
            # Add emotional tone if configured
            emotional_tone = turn_config.get('emotional_tone', 'neutral') if turn_config else 'neutral'
            response = self._apply_emotional_tone(response, emotional_tone)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return base_response
    
    def _personalize_response(self, response: str, context: List[Dict]) -> str:
        """Personalize response based on conversation context"""
        # Extract user preferences from context
        user_sentiment = self._get_context_sentiment(context)
        user_intent = self._get_context_intent(context)
        
        # Adjust response based on sentiment
        if user_sentiment == SentimentType.NEGATIVE:
            response = f"I understand your concern. {response}"
        elif user_sentiment == SentimentType.POSITIVE:
            response = f"Great! {response}"
        
        return response
    
    def _get_context_sentiment(self, context: List[Dict]) -> SentimentType:
        """Get overall sentiment from context"""
        if not context:
            return SentimentType.NEUTRAL
        
        sentiments = []
        for turn in context:
            if turn.get('speaker') == 'user' and 'processed_response' in turn:
                processed = turn['processed_response']
                if hasattr(processed, 'sentiment'):
                    sentiments.append(processed.sentiment.sentiment)
        
        if not sentiments:
            return SentimentType.NEUTRAL
        
        # Return most common sentiment
        from collections import Counter
        sentiment_counts = Counter(sentiments)
        return sentiment_counts.most_common(1)[0][0]
    
    def _get_context_intent(self, context: List[Dict]) -> IntentType:
        """Get most recent intent from context"""
        if not context:
            return IntentType.UNKNOWN
        
        for turn in reversed(context):
            if turn.get('speaker') == 'user' and 'processed_response' in turn:
                processed = turn['processed_response']
                if hasattr(processed, 'intent'):
                    return processed.intent.intent
        
        return IntentType.UNKNOWN
    
    def _apply_turn_enhancements(self, response: str, turn_config: Dict) -> str:
        """Apply turn-specific enhancements"""
        # Add urgency if specified
        if turn_config.get('urgent', False):
            response = f"Important: {response}"
        
        # Add confirmation request if specified
        if turn_config.get('requires_confirmation', False):
            response = f"{response} Can you confirm this?"
        
        return response
    
    def _apply_emotional_tone(self, response: str, tone: str) -> str:
        """Apply emotional tone to response"""
        tone_modifiers = {
            'friendly': lambda x: f"😊 {x}",
            'professional': lambda x: f"Thank you for your inquiry. {x}",
            'empathetic': lambda x: f"I understand. {x}",
            'excited': lambda x: f"Great news! {x}",
            'concerned': lambda x: f"I'm sorry to hear that. {x}"
        }
        
        modifier = tone_modifiers.get(tone.lower())
        if modifier:
            return modifier(response)
        
        return response

# Example usage and testing
async def main():
    """Example usage of the NLP Processor"""
    
    config = {
        'nlp': {
            'spacy_model': 'en_core_web_sm',
            'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        }
    }
    
    processor = NLPProcessor(config)
    
    # Test cases
    test_inputs = [
        "Hello, I'd like to schedule an appointment for tomorrow",
        "This service is terrible! I'm very disappointed",
        "Yes, that time works perfectly for me",
        "Can you tell me more about your services?",
        "Thank you, goodbye!"
    ]
    
    print("Testing NLP Processing:")
    print("=" * 50)
    
    for text in test_inputs:
        print(f"\nInput: '{text}'")
        
        try:
            result = await processor.process_user_input(text)
            
            print(f"Intent: {result.intent.intent.value} (confidence: {result.intent.confidence:.2f})")
            print(f"Sentiment: {result.sentiment.sentiment.value} (confidence: {result.sentiment.confidence:.2f})")
            print(f"Keywords: {result.keywords}")
            print(f"Language: {result.language}")
            
            # Generate response
            response = await processor.generate_response(
                "I understand your request. Let me help you with that.",
                context=[{'speaker': 'user', 'processed_response': result}]
            )
            print(f"Generated Response: {response}")
            
        except Exception as e:
            print(f"Error processing: {e}")
    
    print("\n" + "=" * 50)
    print("NLP Processing test completed")

if __name__ == "__main__":
    asyncio.run(main())
