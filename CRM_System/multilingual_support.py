"""
Multilingual Support Module - Multi-language support for AI calling
================================================================

This module provides comprehensive multilingual support including:
- Language detection and switching
- Multi-language TTS and STT
- Localized conversation flows
- Cultural adaptation
- Translation services
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os

try:
    from googletrans import Translator
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False

try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)

class LanguageCode(Enum):
    """Supported language codes"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    HINDI = "hi"
    ARABIC = "ar"
    DUTCH = "nl"
    SWEDISH = "sv"
    DANISH = "da"
    NORWEGIAN = "no"
    FINNISH = "fi"
    POLISH = "pl"
    TURKISH = "tr"
    THAI = "th"
    VIETNAMESE = "vi"
    INDONESIAN = "id"
    MALAY = "ms"

@dataclass
class LanguageConfig:
    """Language configuration"""
    code: str
    name: str
    native_name: str
    tts_supported: bool
    stt_supported: bool
    nlp_supported: bool
    rtl: bool = False  # Right-to-left language
    cultural_notes: List[str] = None

@dataclass
class TranslationResult:
    """Translation result"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    detected_language: Optional[str] = None

@dataclass
class LocalizedResponse:
    """Localized response with cultural adaptation"""
    text: str
    language: str
    cultural_context: str
    formality_level: str  # formal, informal, neutral
    emotional_tone: str

class MultilingualSupport:
    """
    Multilingual support for AI calling system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multilingual support
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.multilingual_config = config.get('multilingual', {})
        
        # Initialize language configurations
        self.language_configs = self._load_language_configs()
        
        # Initialize translation service
        self.translator = None
        if GOOGLE_TRANSLATE_AVAILABLE:
            try:
                self.translator = Translator()
                logger.info("Google Translate initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Translate: {e}")
        
        # Load localized templates
        self.localized_templates = self._load_localized_templates()
        
        # Cultural adaptation rules
        self.cultural_rules = self._load_cultural_rules()
        
        logger.info("Multilingual Support initialized successfully")
    
    def _load_language_configs(self) -> Dict[str, LanguageConfig]:
        """Load language configurations"""
        configs = {}
        
        # Define supported languages with their capabilities
        language_data = {
            'en': LanguageConfig('en', 'English', 'English', True, True, True, False, [
                'Direct communication style',
                'Use first names in business',
                'Punctuality is important'
            ]),
            'es': LanguageConfig('es', 'Spanish', 'Español', True, True, True, False, [
                'Formal address (usted) for business',
                'Warm and personal communication',
                'Family references are common'
            ]),
            'fr': LanguageConfig('fr', 'French', 'Français', True, True, True, False, [
                'Formal language in business',
                'Politeness is crucial',
                'Use titles and formal greetings'
            ]),
            'de': LanguageConfig('de', 'German', 'Deutsch', True, True, True, False, [
                'Direct and efficient communication',
                'Formal address (Sie) in business',
                'Punctuality is highly valued'
            ]),
            'it': LanguageConfig('it', 'Italian', 'Italiano', True, True, True, False, [
                'Warm and expressive communication',
                'Formal address in business',
                'Personal relationships matter'
            ]),
            'pt': LanguageConfig('pt', 'Portuguese', 'Português', True, True, True, False, [
                'Warm and friendly communication',
                'Formal address in business',
                'Family and personal life important'
            ]),
            'ru': LanguageConfig('ru', 'Russian', 'Русский', True, True, True, False, [
                'Formal address in business',
                'Direct communication style',
                'Respect for hierarchy'
            ]),
            'ja': LanguageConfig('ja', 'Japanese', '日本語', True, True, True, False, [
                'Very formal language in business',
                'Respect and politeness crucial',
                'Indirect communication style'
            ]),
            'ko': LanguageConfig('ko', 'Korean', '한국어', True, True, True, False, [
                'Formal language in business',
                'Respect for hierarchy',
                'Indirect communication preferred'
            ]),
            'zh-cn': LanguageConfig('zh-cn', 'Chinese (Simplified)', '简体中文', True, True, True, False, [
                'Formal language in business',
                'Respect for authority',
                'Indirect communication style'
            ]),
            'hi': LanguageConfig('hi', 'Hindi', 'हिन्दी', True, True, True, False, [
                'Formal language in business',
                'Respect for elders and authority',
                'Family values important'
            ]),
            'ar': LanguageConfig('ar', 'Arabic', 'العربية', True, True, True, True, [
                'Very formal language in business',
                'Respect for religion and culture',
                'Right-to-left text direction'
            ])
        }
        
        for code, config in language_data.items():
            configs[code] = config
        
        return configs
    
    def _load_localized_templates(self) -> Dict[str, Dict[str, str]]:
        """Load localized response templates"""
        return {
            'greeting': {
                'en': 'Hello {contact_name}, this is an AI assistant calling from our company.',
                'es': 'Hola {contact_name}, soy un asistente de IA llamando de nuestra empresa.',
                'fr': 'Bonjour {contact_name}, je suis un assistant IA appelant de notre entreprise.',
                'de': 'Hallo {contact_name}, ich bin ein KI-Assistent von unserer Firma.',
                'it': 'Ciao {contact_name}, sono un assistente IA che chiama dalla nostra azienda.',
                'pt': 'Olá {contact_name}, sou um assistente de IA ligando da nossa empresa.',
                'ja': 'こんにちは {contact_name}、弊社のAIアシスタントです。',
                'ko': '안녕하세요 {contact_name}, 저희 회사의 AI 어시스턴트입니다.',
                'zh-cn': '您好 {contact_name}，我是我们公司的AI助手。',
                'hi': 'नमस्ते {contact_name}, मैं हमारी कंपनी का AI सहायक हूं।',
                'ar': 'مرحبا {contact_name}، أنا مساعد ذكي من شركتنا.'
            },
            'appointment_booking': {
                'en': 'I\'m calling to help you schedule an appointment. What time would work best for you?',
                'es': 'Llamo para ayudarle a programar una cita. ¿Qué hora le convendría más?',
                'fr': 'J\'appelle pour vous aider à planifier un rendez-vous. Quel moment vous conviendrait le mieux?',
                'de': 'Ich rufe an, um Ihnen bei der Terminvereinbarung zu helfen. Welche Zeit wäre für Sie am besten?',
                'it': 'Chiamo per aiutarla a fissare un appuntamento. Quale orario le andrebbe meglio?',
                'pt': 'Ligo para ajudá-lo a agendar uma consulta. Que horário funcionaria melhor para você?',
                'ja': 'お約束のご予約をお手伝いするためにお電話しています。いつがご都合よろしいでしょうか？',
                'ko': '약속을 잡는 것을 도와드리기 위해 전화드렸습니다. 언제가 가장 편하실까요?',
                'zh-cn': '我打电话是为了帮您安排预约。什么时间对您最合适？',
                'hi': 'मैं आपकी अपॉइंटमेंट बुक करने में मदद के लिए फोन कर रहा हूं। आपके लिए कौन सा समय सबसे अच्छा होगा?',
                'ar': 'أتصل لمساعدتك في تحديد موعد. ما هو الوقت الأنسب لك؟'
            },
            'confirmation': {
                'en': 'Perfect! I have you down for {appointment_time}. Is this correct?',
                'es': '¡Perfecto! Le tengo anotado para {appointment_time}. ¿Es correcto?',
                'fr': 'Parfait! Je vous ai noté pour {appointment_time}. Est-ce correct?',
                'de': 'Perfekt! Ich habe Sie für {appointment_time} eingetragen. Ist das korrekt?',
                'it': 'Perfetto! L\'ho segnata per {appointment_time}. È corretto?',
                'pt': 'Perfeito! Anotei você para {appointment_time}. Está correto?',
                'ja': '完璧です！{appointment_time}にご予約いただきました。これで正しいですか？',
                'ko': '완벽합니다! {appointment_time}에 예약해드렸습니다. 이것이 맞나요?',
                'zh-cn': '完美！我为您预约了{appointment_time}。这样对吗？',
                'hi': 'बिल्कुल सही! मैंने आपको {appointment_time} के लिए नोट कर लिया है। क्या यह सही है?',
                'ar': 'ممتاز! لقد سجلت لك موعد {appointment_time}. هل هذا صحيح؟'
            },
            'closing': {
                'en': 'Thank you for your time! Have a great day!',
                'es': '¡Gracias por su tiempo! ¡Que tenga un buen día!',
                'fr': 'Merci pour votre temps! Passez une excellente journée!',
                'de': 'Vielen Dank für Ihre Zeit! Haben Sie einen schönen Tag!',
                'it': 'Grazie per il suo tempo! Buona giornata!',
                'pt': 'Obrigado pelo seu tempo! Tenha um ótimo dia!',
                'ja': 'お時間をいただき、ありがとうございました！良い一日をお過ごしください！',
                'ko': '시간을 내주셔서 감사합니다! 좋은 하루 되세요!',
                'zh-cn': '感谢您的时间！祝您有美好的一天！',
                'hi': 'आपके समय के लिए धन्यवाद! आपका दिन शुभ हो!',
                'ar': 'شكرا لك على وقتك! أتمنى لك يوما سعيدا!'
            }
        }
    
    def _load_cultural_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load cultural adaptation rules"""
        return {
            'formality': {
                'en': {'default': 'neutral', 'business': 'formal'},
                'es': {'default': 'formal', 'business': 'formal'},
                'fr': {'default': 'formal', 'business': 'formal'},
                'de': {'default': 'formal', 'business': 'formal'},
                'it': {'default': 'formal', 'business': 'formal'},
                'pt': {'default': 'formal', 'business': 'formal'},
                'ja': {'default': 'formal', 'business': 'very_formal'},
                'ko': {'default': 'formal', 'business': 'formal'},
                'zh-cn': {'default': 'formal', 'business': 'formal'},
                'hi': {'default': 'formal', 'business': 'formal'},
                'ar': {'default': 'formal', 'business': 'very_formal'}
            },
            'greeting_style': {
                'en': 'direct',
                'es': 'warm',
                'fr': 'polite',
                'de': 'efficient',
                'it': 'warm',
                'pt': 'friendly',
                'ja': 'respectful',
                'ko': 'respectful',
                'zh-cn': 'respectful',
                'hi': 'respectful',
                'ar': 'respectful'
            },
            'time_references': {
                'en': 'exact',
                'es': 'approximate',
                'fr': 'exact',
                'de': 'exact',
                'it': 'approximate',
                'pt': 'approximate',
                'ja': 'exact',
                'ko': 'exact',
                'zh-cn': 'exact',
                'hi': 'approximate',
                'ar': 'approximate'
            }
        }
    
    async def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of input text
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not text.strip():
            return 'en', 0.0
        
        # Try langdetect first
        if LANGDETECT_AVAILABLE:
            try:
                detected_lang = langdetect.detect(text)
                confidence = 0.8  # langdetect doesn't provide confidence
                return detected_lang, confidence
            except Exception as e:
                logger.warning(f"langdetect failed: {e}")
        
        # Try Google Translate
        if self.translator:
            try:
                detection = self.translator.detect(text)
                return detection.lang, detection.confidence
            except Exception as e:
                logger.warning(f"Google Translate detection failed: {e}")
        
        # Fallback to English
        return 'en', 0.5
    
    async def translate_text(self, 
                           text: str, 
                           target_language: str, 
                           source_language: Optional[str] = None) -> TranslationResult:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (auto-detect if None)
            
        Returns:
            TranslationResult object
        """
        if not text.strip():
            return TranslationResult(text, text, target_language, target_language, 1.0)
        
        # Auto-detect source language if not provided
        if source_language is None:
            source_language, _ = await self.detect_language(text)
        
        # If source and target are the same, return original
        if source_language == target_language:
            return TranslationResult(text, text, source_language, target_language, 1.0)
        
        # Use Google Translate
        if self.translator:
            try:
                result = self.translator.translate(text, src=source_language, dest=target_language)
                return TranslationResult(
                    original_text=text,
                    translated_text=result.text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=getattr(result, 'confidence', 0.8),
                    detected_language=source_language
                )
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                return TranslationResult(text, text, source_language, target_language, 0.0)
        
        # Fallback: return original text
        return TranslationResult(text, text, source_language, target_language, 0.0)
    
    def get_localized_template(self, 
                             template_key: str, 
                             language: str, 
                             **kwargs) -> str:
        """
        Get localized template for specific language
        
        Args:
            template_key: Template key (e.g., 'greeting', 'appointment_booking')
            language: Language code
            **kwargs: Template variables
            
        Returns:
            Localized template string
        """
        # Get template for language, fallback to English
        template = self.localized_templates.get(template_key, {}).get(language)
        if not template:
            template = self.localized_templates.get(template_key, {}).get('en', '')
        
        # Replace template variables
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template variable {e} in {template_key}")
            return template
    
    def create_localized_response(self, 
                                base_text: str, 
                                language: str, 
                                context: str = 'business') -> LocalizedResponse:
        """
        Create culturally adapted response
        
        Args:
            base_text: Base response text
            language: Target language
            context: Context (business, casual, etc.)
            
        Returns:
            LocalizedResponse object
        """
        # Get language configuration
        lang_config = self.language_configs.get(language, self.language_configs['en'])
        
        # Determine formality level
        formality_rules = self.cultural_rules['formality'].get(language, {'default': 'neutral'})
        formality_level = formality_rules.get(context, formality_rules['default'])
        
        # Determine emotional tone based on cultural rules
        greeting_style = self.cultural_rules['greeting_style'].get(language, 'neutral')
        emotional_tone = self._map_greeting_style_to_tone(greeting_style)
        
        # Apply cultural adaptations
        adapted_text = self._apply_cultural_adaptations(base_text, language, formality_level)
        
        return LocalizedResponse(
            text=adapted_text,
            language=language,
            cultural_context=context,
            formality_level=formality_level,
            emotional_tone=emotional_tone
        )
    
    def _map_greeting_style_to_tone(self, greeting_style: str) -> str:
        """Map greeting style to emotional tone"""
        style_mapping = {
            'direct': 'professional',
            'warm': 'friendly',
            'polite': 'respectful',
            'efficient': 'professional',
            'friendly': 'warm',
            'respectful': 'respectful'
        }
        return style_mapping.get(greeting_style, 'neutral')
    
    def _apply_cultural_adaptations(self, 
                                  text: str, 
                                  language: str, 
                                  formality_level: str) -> str:
        """Apply cultural adaptations to text"""
        # This is a simplified version - in practice, you'd have more sophisticated rules
        
        if language == 'ja' and formality_level == 'very_formal':
            # Add honorifics and formal language markers
            text = f"恐れ入りますが、{text}"
        elif language == 'ko' and formality_level == 'formal':
            # Add Korean formal endings
            text = f"{text}습니다"
        elif language == 'ar' and formality_level == 'very_formal':
            # Add Arabic formal greetings
            text = f"بسم الله الرحمن الرحيم، {text}"
        
        return text
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return list(self.language_configs.keys())
    
    def get_language_config(self, language: str) -> Optional[LanguageConfig]:
        """Get language configuration"""
        return self.language_configs.get(language)
    
    def is_language_supported(self, language: str, feature: str = 'tts') -> bool:
        """
        Check if language is supported for specific feature
        
        Args:
            language: Language code
            feature: Feature to check (tts, stt, nlp)
            
        Returns:
            True if supported, False otherwise
        """
        lang_config = self.language_configs.get(language)
        if not lang_config:
            return False
        
        if feature == 'tts':
            return lang_config.tts_supported
        elif feature == 'stt':
            return lang_config.stt_supported
        elif feature == 'nlp':
            return lang_config.nlp_supported
        else:
            return False
    
    def get_cultural_notes(self, language: str) -> List[str]:
        """Get cultural notes for language"""
        lang_config = self.language_configs.get(language)
        if lang_config and lang_config.cultural_notes:
            return lang_config.cultural_notes
        return []
    
    def create_multilingual_call_script(self, 
                                      base_script: Dict[str, Any], 
                                      target_languages: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Create multilingual versions of call script
        
        Args:
            base_script: Base call script
            target_languages: List of target languages
            
        Returns:
            Dictionary mapping language codes to localized scripts
        """
        multilingual_scripts = {}
        
        for language in target_languages:
            if not self.is_language_supported(language, 'tts'):
                logger.warning(f"Language {language} not supported for TTS")
                continue
            
            # Create localized script
            localized_script = base_script.copy()
            
            # Translate conversation flow
            if 'conversation_flow' in localized_script:
                for turn in localized_script['conversation_flow']:
                    if 'ai_message' in turn:
                        # Use localized template if available
                        template_key = turn.get('template_key', 'greeting')
                        localized_message = self.get_localized_template(
                            template_key, 
                            language,
                            contact_name='{contact_name}',
                            appointment_time='{appointment_time}'
                        )
                        turn['ai_message'] = localized_message
            
            # Update language settings
            localized_script['language'] = language
            
            # Update voice config
            if 'voice_config' in localized_script:
                localized_script['voice_config']['language'] = language
            
            multilingual_scripts[language] = localized_script
        
        return multilingual_scripts

# Example usage and testing
async def main():
    """Example usage of the Multilingual Support"""
    
    config = {
        'multilingual': {
            'enable_translation': True,
            'default_language': 'en',
            'fallback_language': 'en'
        }
    }
    
    multilingual = MultilingualSupport(config)
    
    print("Multilingual Support Test:")
    print("=" * 50)
    
    # Test language detection
    test_texts = [
        "Hello, I'd like to book an appointment",
        "Hola, me gustaría reservar una cita",
        "Bonjour, j'aimerais prendre rendez-vous",
        "Hallo, ich möchte einen Termin vereinbaren",
        "こんにちは、予約を取りたいです"
    ]
    
    print("Language Detection:")
    for text in test_texts:
        lang, confidence = await multilingual.detect_language(text)
        print(f"'{text[:30]}...' -> {lang} (confidence: {confidence:.2f})")
    
    # Test translation
    print("\nTranslation:")
    original = "Hello, I'd like to book an appointment for tomorrow at 2 PM."
    target_languages = ['es', 'fr', 'de', 'ja', 'ko']
    
    for target_lang in target_languages:
        result = await multilingual.translate_text(original, target_lang)
        print(f"{target_lang}: {result.translated_text}")
    
    # Test localized templates
    print("\nLocalized Templates:")
    languages = ['en', 'es', 'fr', 'de', 'ja', 'ko', 'zh-cn', 'hi', 'ar']
    
    for lang in languages:
        greeting = multilingual.get_localized_template(
            'greeting', 
            lang, 
            contact_name='John'
        )
        print(f"{lang}: {greeting}")
    
    # Test cultural adaptation
    print("\nCultural Adaptation:")
    base_text = "I'm calling to help you with your appointment."
    
    for lang in ['en', 'ja', 'ko', 'ar']:
        response = multilingual.create_localized_response(base_text, lang, 'business')
        print(f"{lang} ({response.formality_level}): {response.text}")
    
    # Test supported languages
    print(f"\nSupported Languages: {multilingual.get_supported_languages()}")
    
    # Test language support
    print("\nLanguage Support:")
    for lang in ['en', 'es', 'ja', 'ko', 'zh-cn']:
        tts_supported = multilingual.is_language_supported(lang, 'tts')
        stt_supported = multilingual.is_language_supported(lang, 'stt')
        print(f"{lang}: TTS={tts_supported}, STT={stt_supported}")
    
    print("\nMultilingual Support test completed")

if __name__ == "__main__":
    asyncio.run(main())
