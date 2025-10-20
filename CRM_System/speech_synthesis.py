"""
Speech Synthesis Module - Text-to-Speech functionality
====================================================

This module provides text-to-speech capabilities using multiple free and open-source
engines including pyttsx3, gTTS, and optional cloud services.
"""

import asyncio
import logging
import io
import tempfile
import os
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum
import threading
import queue

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import torch
    import torchaudio
    from transformers import AutoProcessor, AutoModel
    TORCH_TTS_AVAILABLE = True
except ImportError:
    TORCH_TTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TTSProvider(Enum):
    """Available TTS providers"""
    PYTTSX3 = "pyttsx3"
    GTTS = "gtts"
    TORCH_TTS = "torch_tts"
    EDGE_TTS = "edge_tts"

@dataclass
class VoiceConfig:
    """Voice configuration parameters"""
    voice: str = "default"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 0.8
    language: str = "en"
    gender: str = "neutral"  # male, female, neutral
    emotion: str = "neutral"  # happy, sad, angry, neutral, excited

class SpeechSynthesizer:
    """
    Speech synthesizer supporting multiple TTS engines
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize speech synthesizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tts_config = config.get('tts', {})
        self.default_provider = TTSProvider(self.tts_config.get('default_provider', 'pyttsx3'))
        
        # Initialize available providers
        self.providers = {}
        self._initialize_providers()
        
        # Audio queue for playback
        self.audio_queue = queue.Queue()
        self.playback_thread = None
        self._start_playback_thread()
        
        logger.info(f"Speech Synthesizer initialized with provider: {self.default_provider}")
    
    def _initialize_providers(self):
        """Initialize available TTS providers"""
        
        # Initialize pyttsx3
        if PYTTSX3_AVAILABLE:
            try:
                self.providers[TTSProvider.PYTTSX3] = pyttsx3.init()
                self._configure_pyttsx3()
                logger.info("pyttsx3 provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize pyttsx3: {e}")
        
        # Initialize gTTS
        if GTTS_AVAILABLE:
            self.providers[TTSProvider.GTTS] = True
            logger.info("gTTS provider initialized")
        
        # Initialize Torch TTS (for advanced models)
        if TORCH_TTS_AVAILABLE:
            try:
                self._initialize_torch_tts()
                logger.info("Torch TTS provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Torch TTS: {e}")
    
    def _configure_pyttsx3(self):
        """Configure pyttsx3 engine"""
        if TTSProvider.PYTTSX3 in self.providers:
            engine = self.providers[TTSProvider.PYTTSX3]
            
            # Get available voices
            voices = engine.getProperty('voices')
            if voices:
                # Set a default voice
                engine.setProperty('voice', voices[0].id)
            
            # Set default properties
            engine.setProperty('rate', 150)  # Speed
            engine.setProperty('volume', 0.8)  # Volume
    
    def _initialize_torch_tts(self):
        """Initialize Torch TTS models"""
        try:
            # Load a lightweight TTS model
            model_name = self.tts_config.get('torch_model', 'microsoft/speecht5_tts')
            self.torch_processor = AutoProcessor.from_pretrained(model_name)
            self.torch_model = AutoModel.from_pretrained(model_name)
            self.providers[TTSProvider.TORCH_TTS] = True
        except Exception as e:
            logger.error(f"Failed to load Torch TTS model: {e}")
            raise
    
    def _start_playback_thread(self):
        """Start audio playback thread"""
        if GTTS_AVAILABLE:
            self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
            self.playback_thread.start()
    
    def _playback_worker(self):
        """Audio playback worker thread"""
        pygame.mixer.init()
        while True:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                if audio_data is None:  # Shutdown signal
                    break
                self._play_audio_data(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in playback worker: {e}")
    
    def _play_audio_data(self, audio_data: bytes):
        """Play audio data using pygame"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                
                pygame.mixer.music.load(temp_file.name)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                # Clean up
                os.unlink(temp_file.name)
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    async def synthesize(self, 
                        text: str, 
                        voice_config: Optional[VoiceConfig] = None,
                        provider: Optional[TTSProvider] = None,
                        language: str = "en") -> bytes:
        """
        Synthesize text to speech
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration
            provider: TTS provider to use
            language: Language code
            
        Returns:
            Audio data as bytes
        """
        if not text.strip():
            return b""
        
        # Use default voice config if not provided
        if voice_config is None:
            voice_config = VoiceConfig(language=language)
        
        # Use default provider if not specified
        if provider is None:
            provider = self.default_provider
        
        # Ensure provider is available
        if provider not in self.providers:
            logger.warning(f"Provider {provider} not available, falling back to default")
            provider = self.default_provider
        
        try:
            if provider == TTSProvider.PYTTSX3:
                return await self._synthesize_pyttsx3(text, voice_config)
            elif provider == TTSProvider.GTTS:
                return await self._synthesize_gtts(text, voice_config)
            elif provider == TTSProvider.TORCH_TTS:
                return await self._synthesize_torch_tts(text, voice_config)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error synthesizing speech with {provider}: {e}")
            # Fallback to available provider
            for fallback_provider in self.providers:
                if fallback_provider != provider:
                    try:
                        return await self.synthesize(text, voice_config, fallback_provider, language)
                    except Exception as fallback_error:
                        logger.error(f"Fallback provider {fallback_provider} also failed: {fallback_error}")
                        continue
            raise Exception("All TTS providers failed")
    
    async def _synthesize_pyttsx3(self, text: str, voice_config: VoiceConfig) -> bytes:
        """Synthesize using pyttsx3"""
        if TTSProvider.PYTTSX3 not in self.providers:
            raise Exception("pyttsx3 not available")
        
        engine = self.providers[TTSProvider.PYTTSX3]
        
        # Configure voice properties
        engine.setProperty('rate', int(150 * voice_config.speed))
        engine.setProperty('volume', voice_config.volume)
        
        # Set voice if available
        voices = engine.getProperty('voices')
        if voices and voice_config.voice != "default":
            for voice in voices:
                if voice_config.voice.lower() in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        
        # Synthesize to file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            engine.save_to_file(text, temp_file.name)
            engine.runAndWait()
            
            # Read the generated audio file
            with open(temp_file.name, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
            
            return audio_data
    
    async def _synthesize_gtts(self, text: str, voice_config: VoiceConfig) -> bytes:
        """Synthesize using Google Text-to-Speech"""
        if not GTTS_AVAILABLE:
            raise Exception("gTTS not available")
        
        # Create gTTS object
        tts = gTTS(text=text, lang=voice_config.language, slow=False)
        
        # Generate audio data
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.read()
    
    async def _synthesize_torch_tts(self, text: str, voice_config: VoiceConfig) -> bytes:
        """Synthesize using Torch TTS models"""
        if not TORCH_TTS_AVAILABLE or TTSProvider.TORCH_TTS not in self.providers:
            raise Exception("Torch TTS not available")
        
        try:
            # Tokenize input text
            inputs = self.torch_processor(text=text, return_tensors="pt")
            
            # Generate speech
            with torch.no_grad():
                generated_speech = self.torch_model.generate_speech(
                    inputs["input_ids"], 
                    inputs["attention_mask"]
                )
            
            # Convert to audio data
            audio_tensor = generated_speech.squeeze().cpu()
            
            # Convert to bytes (simplified - in real implementation, use proper audio encoding)
            audio_data = audio_tensor.numpy().tobytes()
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error in Torch TTS synthesis: {e}")
            raise
    
    async def synthesize_and_play(self, 
                                 text: str, 
                                 voice_config: Optional[VoiceConfig] = None,
                                 provider: Optional[TTSProvider] = None,
                                 language: str = "en") -> None:
        """
        Synthesize text and play it immediately
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration
            provider: TTS provider to use
            language: Language code
        """
        audio_data = await self.synthesize(text, voice_config, provider, language)
        
        if audio_data and GTTS_AVAILABLE:
            # Queue audio for playback
            self.audio_queue.put(audio_data)
        else:
            logger.warning("Audio playback not available")
    
    def get_available_voices(self, provider: Optional[TTSProvider] = None) -> Dict[str, List[str]]:
        """
        Get available voices for each provider
        
        Args:
            provider: Specific provider to check (None for all)
            
        Returns:
            Dictionary mapping provider names to list of available voices
        """
        voices = {}
        
        if provider is None or provider == TTSProvider.PYTTSX3:
            if TTSProvider.PYTTSX3 in self.providers:
                engine = self.providers[TTSProvider.PYTTSX3]
                pyttsx3_voices = engine.getProperty('voices')
                voices['pyttsx3'] = [voice.name for voice in pyttsx3_voices] if pyttsx3_voices else []
        
        if provider is None or provider == TTSProvider.GTTS:
            if GTTS_AVAILABLE:
                # gTTS supports many languages
                voices['gtts'] = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'hi', 'ar']
        
        return voices
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        languages = set()
        
        # Add languages from gTTS
        if GTTS_AVAILABLE:
            languages.update(['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'hi', 'ar'])
        
        # Add languages from pyttsx3 (system dependent)
        if TTSProvider.PYTTSX3 in self.providers:
            languages.add('en')  # pyttsx3 typically supports system language
        
        return list(languages)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.playback_thread and self.playback_thread.is_alive():
            self.audio_queue.put(None)  # Shutdown signal
            self.playback_thread.join(timeout=5)
        
        if TTSProvider.PYTTSX3 in self.providers:
            self.providers[TTSProvider.PYTTSX3].stop()

# Example usage and testing
async def main():
    """Example usage of the Speech Synthesizer"""
    
    config = {
        'tts': {
            'default_provider': 'pyttsx3',
            'torch_model': 'microsoft/speecht5_tts'
        }
    }
    
    synthesizer = SpeechSynthesizer(config)
    
    # Test different providers
    test_text = "Hello, this is a test of the AI calling system."
    
    print("Available voices:")
    voices = synthesizer.get_available_voices()
    for provider, voice_list in voices.items():
        print(f"{provider}: {voice_list}")
    
    print("\nSupported languages:")
    languages = synthesizer.get_supported_languages()
    print(languages)
    
    # Test synthesis
    try:
        print("\nTesting pyttsx3 synthesis...")
        audio_data = await synthesizer.synthesize(
            test_text, 
            VoiceConfig(voice="default", speed=1.0),
            TTSProvider.PYTTSX3
        )
        print(f"Generated audio: {len(audio_data)} bytes")
        
        print("\nTesting gTTS synthesis...")
        audio_data = await synthesizer.synthesize(
            test_text,
            VoiceConfig(language="en"),
            TTSProvider.GTTS
        )
        print(f"Generated audio: {len(audio_data)} bytes")
        
    except Exception as e:
        print(f"Error during synthesis: {e}")
    
    finally:
        synthesizer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
