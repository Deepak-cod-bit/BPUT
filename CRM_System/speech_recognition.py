"""
Speech Recognition Module - Speech-to-Text functionality
======================================================

This module provides speech recognition capabilities using multiple free and open-source
engines including Whisper, SpeechRecognition, and optional cloud services.
"""

import asyncio
import logging
import io
import tempfile
import os
import wave
import threading
import queue
from typing import Dict, Any, Optional, Union, List, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import torch
    import torchaudio
    TORCH_AUDIO_AVAILABLE = True
except ImportError:
    TORCH_AUDIO_AVAILABLE = False

logger = logging.getLogger(__name__)

class STTProvider(Enum):
    """Available Speech-to-Text providers"""
    WHISPER = "whisper"
    GOOGLE = "google"
    SPHINX = "sphinx"
    AZURE = "azure"
    BING = "bing"

@dataclass
class RecognitionConfig:
    """Speech recognition configuration"""
    language: str = "en"
    model_size: str = "base"  # For Whisper: tiny, base, small, medium, large
    timeout: float = 5.0
    phrase_timeout: float = 0.3
    energy_threshold: int = 300
    dynamic_energy_threshold: bool = True
    pause_threshold: float = 0.8
    operation_timeout: float = 10.0

@dataclass
class RecognitionResult:
    """Speech recognition result"""
    text: str
    confidence: float
    language: str
    duration: float
    provider: str
    timestamp: float
    alternatives: List[str] = None

class SpeechRecognizer:
    """
    Speech recognizer supporting multiple STT engines
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize speech recognizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.stt_config = config.get('stt', {})
        self.default_provider = STTProvider(self.stt_config.get('default_provider', 'whisper'))
        
        # Initialize available providers
        self.providers = {}
        self._initialize_providers()
        
        # Audio recording
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        
        # Recognition callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'recognition_started': [],
            'recognition_completed': [],
            'recognition_failed': []
        }
        
        logger.info(f"Speech Recognizer initialized with provider: {self.default_provider}")
    
    def _initialize_providers(self):
        """Initialize available STT providers"""
        
        # Initialize Whisper
        if WHISPER_AVAILABLE:
            try:
                model_size = self.stt_config.get('whisper_model', 'base')
                self.whisper_model = whisper.load_model(model_size)
                self.providers[STTProvider.WHISPER] = True
                logger.info(f"Whisper provider initialized with model: {model_size}")
            except Exception as e:
                logger.warning(f"Failed to initialize Whisper: {e}")
        
        # Initialize SpeechRecognition
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self._configure_speech_recognition()
            self.providers[STTProvider.GOOGLE] = True
            self.providers[STTProvider.SPHINX] = True
            logger.info("SpeechRecognition provider initialized")
    
    def _configure_speech_recognition(self):
        """Configure SpeechRecognition settings"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            return
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Set recognition parameters
        self.recognizer.energy_threshold = self.stt_config.get('energy_threshold', 300)
        self.recognizer.dynamic_energy_threshold = self.stt_config.get('dynamic_energy_threshold', True)
        self.recognizer.pause_threshold = self.stt_config.get('pause_threshold', 0.8)
        self.recognizer.operation_timeout = self.stt_config.get('operation_timeout', 10.0)
    
    def add_callback(self, event: str, callback: Callable):
        """Add recognition callback"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """Trigger recognition callbacks"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for event {event}: {e}")
    
    async def recognize_audio_file(self, 
                                 audio_file_path: str,
                                 recognition_config: Optional[RecognitionConfig] = None,
                                 provider: Optional[STTProvider] = None) -> RecognitionResult:
        """
        Recognize speech from audio file
        
        Args:
            audio_file_path: Path to audio file
            recognition_config: Recognition configuration
            provider: STT provider to use
            
        Returns:
            RecognitionResult object
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        # Use default config if not provided
        if recognition_config is None:
            recognition_config = RecognitionConfig()
        
        # Use default provider if not specified
        if provider is None:
            provider = self.default_provider
        
        # Ensure provider is available
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        
        self._trigger_callback('recognition_started', audio_file_path, provider)
        
        try:
            if provider == STTProvider.WHISPER:
                return await self._recognize_with_whisper(audio_file_path, recognition_config)
            elif provider == STTProvider.GOOGLE:
                return await self._recognize_with_google(audio_file_path, recognition_config)
            elif provider == STTProvider.SPHINX:
                return await self._recognize_with_sphinx(audio_file_path, recognition_config)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error recognizing speech with {provider}: {e}")
            self._trigger_callback('recognition_failed', audio_file_path, str(e))
            raise
    
    async def _recognize_with_whisper(self, audio_file_path: str, config: RecognitionConfig) -> RecognitionResult:
        """Recognize speech using Whisper"""
        if not WHISPER_AVAILABLE or STTProvider.WHISPER not in self.providers:
            raise Exception("Whisper not available")
        
        try:
            # Load and transcribe audio
            result = self.whisper_model.transcribe(
                audio_file_path,
                language=config.language if config.language != "en" else None,
                fp16=False  # Use fp32 for better compatibility
            )
            
            # Extract text and confidence
            text = result["text"].strip()
            segments = result.get("segments", [])
            
            # Calculate average confidence
            if segments:
                confidences = [seg.get("avg_logprob", 0) for seg in segments if "avg_logprob" in seg]
                confidence = np.exp(np.mean(confidences)) if confidences else 0.5
            else:
                confidence = 0.5
            
            # Get duration
            duration = result.get("duration", 0)
            
            # Create result
            recognition_result = RecognitionResult(
                text=text,
                confidence=float(confidence),
                language=config.language,
                duration=float(duration),
                provider="whisper",
                timestamp=0.0  # Whisper doesn't provide timestamp
            )
            
            self._trigger_callback('recognition_completed', recognition_result)
            return recognition_result
            
        except Exception as e:
            logger.error(f"Whisper recognition error: {e}")
            raise
    
    async def _recognize_with_google(self, audio_file_path: str, config: RecognitionConfig) -> RecognitionResult:
        """Recognize speech using Google Speech Recognition"""
        if not SPEECH_RECOGNITION_AVAILABLE or STTProvider.GOOGLE not in self.providers:
            raise Exception("Google Speech Recognition not available")
        
        try:
            # Load audio file
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
            
            # Recognize speech
            result = self.recognizer.recognize_google(
                audio,
                language=config.language,
                show_all=False
            )
            
            # Create result
            recognition_result = RecognitionResult(
                text=result,
                confidence=1.0,  # Google doesn't provide confidence
                language=config.language,
                duration=0.0,  # Would need to calculate from audio
                provider="google",
                timestamp=0.0
            )
            
            self._trigger_callback('recognition_completed', recognition_result)
            return recognition_result
            
        except sr.UnknownValueError:
            raise Exception("Could not understand audio")
        except sr.RequestError as e:
            raise Exception(f"Recognition service error: {e}")
        except Exception as e:
            logger.error(f"Google recognition error: {e}")
            raise
    
    async def _recognize_with_sphinx(self, audio_file_path: str, config: RecognitionConfig) -> RecognitionResult:
        """Recognize speech using PocketSphinx"""
        if not SPEECH_RECOGNITION_AVAILABLE or STTProvider.SPHINX not in self.providers:
            raise Exception("PocketSphinx not available")
        
        try:
            # Load audio file
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
            
            # Recognize speech
            result = self.recognizer.recognize_sphinx(
                audio,
                language=config.language
            )
            
            # Create result
            recognition_result = RecognitionResult(
                text=result,
                confidence=1.0,  # Sphinx doesn't provide confidence
                language=config.language,
                duration=0.0,
                provider="sphinx",
                timestamp=0.0
            )
            
            self._trigger_callback('recognition_completed', recognition_result)
            return recognition_result
            
        except sr.UnknownValueError:
            raise Exception("Could not understand audio")
        except sr.RequestError as e:
            raise Exception(f"Recognition service error: {e}")
        except Exception as e:
            logger.error(f"Sphinx recognition error: {e}")
            raise
    
    async def recognize_microphone(self, 
                                 recognition_config: Optional[RecognitionConfig] = None,
                                 provider: Optional[STTProvider] = None,
                                 timeout: float = 5.0) -> RecognitionResult:
        """
        Recognize speech from microphone
        
        Args:
            recognition_config: Recognition configuration
            provider: STT provider to use
            timeout: Recording timeout in seconds
            
        Returns:
            RecognitionResult object
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            raise Exception("Microphone recognition requires SpeechRecognition library")
        
        # Use default config if not provided
        if recognition_config is None:
            recognition_config = RecognitionConfig()
        
        # Use default provider if not specified
        if provider is None:
            provider = self.default_provider
        
        self._trigger_callback('recognition_started', "microphone", provider)
        
        try:
            # Listen for audio
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout)
            
            # Save audio to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(audio.sample_width)
                    wav_file.setframerate(audio.sample_rate)
                    wav_file.writeframes(audio.frame_data)
                
                # Recognize the audio file
                result = await self.recognize_audio_file(
                    temp_file.name, 
                    recognition_config, 
                    provider
                )
                
                # Clean up
                os.unlink(temp_file.name)
                
                return result
                
        except Exception as e:
            logger.error(f"Microphone recognition error: {e}")
            self._trigger_callback('recognition_failed', "microphone", str(e))
            raise
    
    async def start_continuous_recognition(self, 
                                         recognition_config: Optional[RecognitionConfig] = None,
                                         provider: Optional[STTProvider] = None,
                                         callback: Optional[Callable] = None):
        """
        Start continuous speech recognition
        
        Args:
            recognition_config: Recognition configuration
            provider: STT provider to use
            callback: Callback function for recognition results
        """
        if self.is_recording:
            logger.warning("Continuous recognition already running")
            return
        
        self.is_recording = True
        
        # Start recording thread
        self.recording_thread = threading.Thread(
            target=self._continuous_recognition_worker,
            args=(recognition_config, provider, callback),
            daemon=True
        )
        self.recording_thread.start()
        
        logger.info("Continuous recognition started")
    
    def _continuous_recognition_worker(self, 
                                     recognition_config: Optional[RecognitionConfig],
                                     provider: Optional[STTProvider],
                                     callback: Optional[Callable]):
        """Continuous recognition worker thread"""
        try:
            while self.is_recording:
                try:
                    # Listen for audio
                    with self.microphone as source:
                        audio = self.recognizer.listen(source, timeout=1.0)
                    
                    # Save audio to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        with wave.open(temp_file.name, 'wb') as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(audio.sample_width)
                            wav_file.setframerate(audio.sample_rate)
                            wav_file.writeframes(audio.frame_data)
                        
                        # Recognize audio
                        result = asyncio.run(self.recognize_audio_file(
                            temp_file.name,
                            recognition_config,
                            provider
                        ))
                        
                        # Call callback if provided
                        if callback and result.text.strip():
                            callback(result)
                        
                        # Clean up
                        os.unlink(temp_file.name)
                
                except sr.WaitTimeoutError:
                    # No audio detected, continue
                    continue
                except Exception as e:
                    logger.error(f"Error in continuous recognition: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Continuous recognition worker error: {e}")
        finally:
            self.is_recording = False
    
    def stop_continuous_recognition(self):
        """Stop continuous speech recognition"""
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2)
        logger.info("Continuous recognition stopped")
    
    def get_available_providers(self) -> List[STTProvider]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        languages = set()
        
        # Add common languages
        languages.update([
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'hi', 'ar',
            'nl', 'sv', 'da', 'no', 'fi', 'pl', 'tr', 'th', 'vi', 'id', 'ms'
        ])
        
        return list(languages)
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_continuous_recognition()

# Example usage and testing
async def main():
    """Example usage of the Speech Recognizer"""
    
    config = {
        'stt': {
            'default_provider': 'whisper',
            'whisper_model': 'base',
            'energy_threshold': 300,
            'pause_threshold': 0.8
        }
    }
    
    recognizer = SpeechRecognizer(config)
    
    # Add callbacks
    def on_recognition_completed(result):
        print(f"Recognized: '{result.text}' (confidence: {result.confidence:.2f})")
    
    recognizer.add_callback('recognition_completed', on_recognition_completed)
    
    print("Available providers:")
    providers = recognizer.get_available_providers()
    for provider in providers:
        print(f"- {provider.value}")
    
    print("\nSupported languages:")
    languages = recognizer.get_supported_languages()
    print(languages[:10], "...")  # Show first 10
    
    # Test microphone recognition (if available)
    try:
        print("\nTesting microphone recognition (speak for 5 seconds)...")
        result = await recognizer.recognize_microphone(timeout=5.0)
        print(f"Result: {result.text}")
    except Exception as e:
        print(f"Microphone test failed: {e}")
    
    # Test continuous recognition
    try:
        print("\nStarting continuous recognition (press Ctrl+C to stop)...")
        await recognizer.start_continuous_recognition()
        
        # Run for 10 seconds
        await asyncio.sleep(10)
        
        recognizer.stop_continuous_recognition()
        print("Continuous recognition stopped")
        
    except KeyboardInterrupt:
        recognizer.stop_continuous_recognition()
        print("\nStopped by user")
    except Exception as e:
        print(f"Continuous recognition error: {e}")
    
    finally:
        recognizer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
