"""
WhatsApp Integration Module
==========================

This module provides WhatsApp messaging capabilities for the AI calling system.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class WhatsAppMessage:
    """WhatsApp message structure"""
    message_id: str
    phone_number: str
    message: str
    timestamp: datetime
    message_type: str = "text"  # text, image, document
    status: str = "sent"  # sent, delivered, read

@dataclass
class WhatsAppContact:
    """WhatsApp contact structure"""
    phone_number: str
    name: Optional[str] = None
    last_seen: Optional[datetime] = None
    is_online: bool = False

class WhatsAppIntegration:
    """
    WhatsApp integration for AI calling system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WhatsApp integration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.whatsapp_config = config.get('whatsapp', {})
        
        # WhatsApp Web integration
        self.driver = None
        self.is_connected = False
        
        # Message tracking
        self.sent_messages: List[WhatsAppMessage] = []
        self.contacts: Dict[str, WhatsAppContact] = {}
        
        logger.info("WhatsApp Integration initialized")
    
    async def connect_whatsapp_web(self) -> bool:
        """
        Connect to WhatsApp Web using Selenium
        
        Returns:
            True if connected successfully, False otherwise
        """
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium not available for WhatsApp Web integration")
            return False
        
        try:
            # Initialize Chrome driver
            options = webdriver.ChromeOptions()
            options.add_argument("--user-data-dir=./whatsapp_session")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            self.driver = webdriver.Chrome(options=options)
            self.driver.get("https://web.whatsapp.com/")
            
            # Wait for QR code scan or already logged in
            wait = WebDriverWait(self.driver, 60)
            
            try:
                # Check if already logged in
                wait.until(EC.presence_of_element_located((By.XPATH, '//div[@title="Search input textbox"]')))
                self.is_connected = True
                logger.info("WhatsApp Web connected successfully")
                return True
            except:
                # Wait for QR code scan
                wait.until(EC.presence_of_element_located((By.XPATH, '//div[@data-testid="qr-code"]')))
                logger.info("Please scan QR code in browser")
                
                # Wait for login completion
                wait.until(EC.presence_of_element_located((By.XPATH, '//div[@title="Search input textbox"]')))
                self.is_connected = True
                logger.info("WhatsApp Web connected after QR scan")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to WhatsApp Web: {e}")
            return False
    
    async def send_message(self, phone_number: str, message: str) -> bool:
        """
        Send WhatsApp message
        
        Args:
            phone_number: Recipient phone number (with country code)
            message: Message text to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_connected:
            logger.error("WhatsApp not connected")
            return False
        
        try:
            # Format phone number (remove + and spaces)
            formatted_number = phone_number.replace("+", "").replace(" ", "").replace("-", "")
            
            # Search for contact
            search_box = self.driver.find_element(By.XPATH, '//div[@title="Search input textbox"]')
            search_box.click()
            search_box.send_keys(formatted_number)
            search_box.send_keys(Keys.ENTER)
            
            # Wait for chat to load
            await asyncio.sleep(2)
            
            # Find message input box
            message_box = self.driver.find_element(By.XPATH, '//div[@title="Type a message"]')
            message_box.click()
            message_box.send_keys(message)
            message_box.send_keys(Keys.ENTER)
            
            # Create message record
            message_record = WhatsAppMessage(
                message_id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                phone_number=phone_number,
                message=message,
                timestamp=datetime.now()
            )
            self.sent_messages.append(message_record)
            
            logger.info(f"WhatsApp message sent to {phone_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")
            return False
    
    async def send_ai_generated_message(self, 
                                      phone_number: str, 
                                      context: str,
                                      message_type: str = "appointment_reminder") -> bool:
        """
        Send AI-generated WhatsApp message
        
        Args:
            phone_number: Recipient phone number
            context: Context for message generation
            message_type: Type of message to generate
            
        Returns:
            True if sent successfully, False otherwise
        """
        # Generate AI message based on type
        ai_message = self._generate_ai_message(context, message_type)
        
        # Send the message
        return await self.send_message(phone_number, ai_message)
    
    def _generate_ai_message(self, context: str, message_type: str) -> str:
        """Generate AI message based on context and type"""
        
        message_templates = {
            'appointment_reminder': f"Hi! This is an AI assistant from our company. {context} Your appointment is confirmed. Reply STOP to opt out.",
            'appointment_confirmation': f"Hello! Your appointment has been confirmed. {context} We'll see you soon!",
            'feedback_request': f"Hi! We hope you enjoyed our service. {context} Please share your feedback by replying to this message.",
            'general_info': f"Hello! {context} If you have any questions, feel free to reply to this message.",
            'follow_up': f"Hi! Following up on our recent interaction. {context} Is there anything else we can help you with?"
        }
        
        return message_templates.get(message_type, f"Hello! {context}")
    
    async def get_contact_info(self, phone_number: str) -> Optional[WhatsAppContact]:
        """
        Get WhatsApp contact information
        
        Args:
            phone_number: Contact phone number
            
        Returns:
            WhatsAppContact object or None
        """
        if not self.is_connected:
            return None
        
        try:
            # Search for contact
            search_box = self.driver.find_element(By.XPATH, '//div[@title="Search input textbox"]')
            search_box.click()
            search_box.send_keys(phone_number)
            search_box.send_keys(Keys.ENTER)
            
            await asyncio.sleep(2)
            
            # Try to get contact name
            try:
                contact_name_element = self.driver.find_element(By.XPATH, '//span[@title]')
                contact_name = contact_name_element.get_attribute('title')
            except:
                contact_name = None
            
            # Check if online
            try:
                online_indicator = self.driver.find_element(By.XPATH, '//span[@data-testid="online"]')
                is_online = True
            except:
                is_online = False
            
            contact = WhatsAppContact(
                phone_number=phone_number,
                name=contact_name,
                is_online=is_online
            )
            
            self.contacts[phone_number] = contact
            return contact
            
        except Exception as e:
            logger.error(f"Failed to get contact info: {e}")
            return None
    
    async def send_bulk_messages(self, 
                                phone_numbers: List[str], 
                                message: str) -> Dict[str, bool]:
        """
        Send bulk WhatsApp messages
        
        Args:
            phone_numbers: List of phone numbers
            message: Message to send
            
        Returns:
            Dictionary mapping phone numbers to success status
        """
        results = {}
        
        for phone_number in phone_numbers:
            success = await self.send_message(phone_number, message)
            results[phone_number] = success
            
            # Add delay between messages to avoid spam detection
            await asyncio.sleep(2)
        
        return results
    
    async def schedule_message(self, 
                             phone_number: str, 
                             message: str, 
                             send_time: datetime) -> str:
        """
        Schedule WhatsApp message for later
        
        Args:
            phone_number: Recipient phone number
            message: Message to send
            send_time: When to send the message
            
        Returns:
            Scheduled message ID
        """
        message_id = f"scheduled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # In a real implementation, you would store this in a database
        # and have a background task to send at the scheduled time
        scheduled_message = {
            'message_id': message_id,
            'phone_number': phone_number,
            'message': message,
            'send_time': send_time.isoformat(),
            'status': 'scheduled'
        }
        
        logger.info(f"Message scheduled for {send_time}: {message_id}")
        return message_id
    
    def get_message_history(self, phone_number: str) -> List[WhatsAppMessage]:
        """Get message history for a contact"""
        return [msg for msg in self.sent_messages if msg.phone_number == phone_number]
    
    def get_all_contacts(self) -> List[WhatsAppContact]:
        """Get all contacts"""
        return list(self.contacts.values())
    
    async def disconnect(self):
        """Disconnect from WhatsApp Web"""
        if self.driver:
            self.driver.quit()
            self.is_connected = False
            logger.info("WhatsApp Web disconnected")

# Example usage
async def main():
    """Example usage of WhatsApp integration"""
    
    config = {
        'whatsapp': {
            'enable_web_integration': True,
            'session_data_dir': './whatsapp_session'
        }
    }
    
    whatsapp = WhatsAppIntegration(config)
    
    # Connect to WhatsApp Web
    connected = await whatsapp.connect_whatsapp_web()
    if not connected:
        print("Failed to connect to WhatsApp Web")
        return
    
    # Send a message
    success = await whatsapp.send_message("+1234567890", "Hello from AI calling system!")
    print(f"Message sent: {success}")
    
    # Send AI-generated message
    ai_success = await whatsapp.send_ai_generated_message(
        "+1234567890",
        "Your appointment is confirmed for tomorrow at 2 PM.",
        "appointment_confirmation"
    )
    print(f"AI message sent: {ai_success}")
    
    # Disconnect
    await whatsapp.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
