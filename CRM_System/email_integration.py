"""
Email Integration Module
=======================

This module provides email capabilities for the AI calling system.
"""

import asyncio
import logging
import smtplib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json

logger = logging.getLogger(__name__)

@dataclass
class EmailMessage:
    """Email message structure"""
    message_id: str
    to_email: str
    subject: str
    body: str
    timestamp: datetime
    status: str = "sent"  # sent, delivered, failed
    attachments: List[str] = None

@dataclass
class EmailTemplate:
    """Email template structure"""
    template_id: str
    name: str
    subject_template: str
    body_template: str
    is_html: bool = False

class EmailIntegration:
    """
    Email integration for AI calling system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize email integration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.email_config = config.get('email', {})
        
        # Email settings
        self.smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = self.email_config.get('smtp_port', 587)
        self.username = self.email_config.get('username', '')
        self.password = self.email_config.get('password', '')
        self.from_email = self.email_config.get('from_email', '')
        
        # Message tracking
        self.sent_emails: List[EmailMessage] = []
        self.email_templates: Dict[str, EmailTemplate] = {}
        
        # Load default templates
        self._load_default_templates()
        
        logger.info("Email Integration initialized")
    
    def _load_default_templates(self):
        """Load default email templates"""
        default_templates = {
            'appointment_confirmation': EmailTemplate(
                template_id='appointment_confirmation',
                name='Appointment Confirmation',
                subject_template='Appointment Confirmed - {appointment_time}',
                body_template='''
Dear {contact_name},

Your appointment has been confirmed for {appointment_time}.

Appointment Details:
- Date: {appointment_date}
- Time: {appointment_time}
- Location: {appointment_location}
- Contact: {contact_phone}

If you need to reschedule or have any questions, please contact us.

Best regards,
AI Assistant Team
                ''',
                is_html=False
            ),
            'appointment_reminder': EmailTemplate(
                template_id='appointment_reminder',
                name='Appointment Reminder',
                subject_template='Reminder: Your appointment tomorrow at {appointment_time}',
                body_template='''
Dear {contact_name},

This is a friendly reminder about your appointment tomorrow at {appointment_time}.

Please arrive 10 minutes early and bring any required documents.

If you need to reschedule, please contact us as soon as possible.

Best regards,
AI Assistant Team
                ''',
                is_html=False
            ),
            'feedback_request': EmailTemplate(
                template_id='feedback_request',
                name='Feedback Request',
                subject_template='How was your experience?',
                body_template='''
Dear {contact_name},

We hope you had a great experience with our service. Your feedback is very important to us.

Please take a moment to share your thoughts by replying to this email or visiting our feedback form.

Thank you for choosing us!

Best regards,
AI Assistant Team
                ''',
                is_html=False
            ),
            'call_summary': EmailTemplate(
                template_id='call_summary',
                name='Call Summary',
                subject_template='Summary of our call - {call_date}',
                body_template='''
Dear {contact_name},

Thank you for your time during our call on {call_date}.

Call Summary:
- Duration: {call_duration}
- Topics Discussed: {topics}
- Next Steps: {next_steps}
- Follow-up Required: {follow_up}

If you have any questions about this summary, please don't hesitate to contact us.

Best regards,
AI Assistant Team
                ''',
                is_html=False
            )
        }
        
        for template_id, template in default_templates.items():
            self.email_templates[template_id] = template
    
    async def send_email(self, 
                        to_email: str, 
                        subject: str, 
                        body: str, 
                        is_html: bool = False,
                        attachments: List[str] = None) -> bool:
        """
        Send email
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body
            is_html: Whether body is HTML
            attachments: List of file paths to attach
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add body
            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    try:
                        with open(file_path, "rb") as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                        
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {file_path.split("/")[-1]}'
                        )
                        msg.attach(part)
                    except Exception as e:
                        logger.warning(f"Failed to attach file {file_path}: {e}")
            
            # Connect to server and send
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            
            text = msg.as_string()
            server.sendmail(self.from_email, to_email, text)
            server.quit()
            
            # Record sent email
            email_record = EmailMessage(
                message_id=f"email_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                to_email=to_email,
                subject=subject,
                body=body,
                timestamp=datetime.now(),
                status="sent",
                attachments=attachments or []
            )
            self.sent_emails.append(email_record)
            
            logger.info(f"Email sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    async def send_template_email(self, 
                                 to_email: str, 
                                 template_id: str, 
                                 variables: Dict[str, str]) -> bool:
        """
        Send email using template
        
        Args:
            to_email: Recipient email address
            template_id: Template ID to use
            variables: Variables to substitute in template
            
        Returns:
            True if sent successfully, False otherwise
        """
        template = self.email_templates.get(template_id)
        if not template:
            logger.error(f"Email template '{template_id}' not found")
            return False
        
        # Substitute variables in template
        subject = template.subject_template.format(**variables)
        body = template.body_template.format(**variables)
        
        return await self.send_email(
            to_email=to_email,
            subject=subject,
            body=body,
            is_html=template.is_html
        )
    
    async def send_ai_generated_email(self, 
                                    to_email: str, 
                                    context: str,
                                    email_type: str = "general") -> bool:
        """
        Send AI-generated email
        
        Args:
            to_email: Recipient email address
            context: Context for email generation
            email_type: Type of email to generate
            
        Returns:
            True if sent successfully, False otherwise
        """
        # Generate AI email content
        subject, body = self._generate_ai_email(context, email_type)
        
        return await self.send_email(to_email, subject, body)
    
    def _generate_ai_email(self, context: str, email_type: str) -> tuple:
        """Generate AI email content based on context and type"""
        
        email_templates = {
            'appointment_confirmation': (
                "Appointment Confirmed",
                f"Dear Customer,\n\nYour appointment has been confirmed. {context}\n\nBest regards,\nAI Assistant"
            ),
            'appointment_reminder': (
                "Appointment Reminder",
                f"Dear Customer,\n\nThis is a reminder about your upcoming appointment. {context}\n\nBest regards,\nAI Assistant"
            ),
            'feedback_request': (
                "Your Feedback is Important",
                f"Dear Customer,\n\nWe hope you enjoyed our service. {context} Please share your feedback.\n\nBest regards,\nAI Assistant"
            ),
            'follow_up': (
                "Follow-up on Our Conversation",
                f"Dear Customer,\n\nFollowing up on our recent conversation. {context}\n\nBest regards,\nAI Assistant"
            ),
            'general': (
                "Important Information",
                f"Dear Customer,\n\n{context}\n\nBest regards,\nAI Assistant"
            )
        }
        
        return email_templates.get(email_type, email_templates['general'])
    
    async def send_bulk_emails(self, 
                              email_list: List[Dict[str, str]], 
                              template_id: str) -> Dict[str, bool]:
        """
        Send bulk emails using template
        
        Args:
            email_list: List of dictionaries with 'email' and 'variables' keys
            template_id: Template ID to use
            
        Returns:
            Dictionary mapping email addresses to success status
        """
        results = {}
        
        for email_data in email_list:
            to_email = email_data['email']
            variables = email_data.get('variables', {})
            
            success = await self.send_template_email(to_email, template_id, variables)
            results[to_email] = success
            
            # Add delay between emails to avoid spam detection
            await asyncio.sleep(1)
        
        return results
    
    async def schedule_email(self, 
                           to_email: str, 
                           subject: str, 
                           body: str, 
                           send_time: datetime) -> str:
        """
        Schedule email for later
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body
            send_time: When to send the email
            
        Returns:
            Scheduled email ID
        """
        email_id = f"scheduled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # In a real implementation, you would store this in a database
        # and have a background task to send at the scheduled time
        scheduled_email = {
            'email_id': email_id,
            'to_email': to_email,
            'subject': subject,
            'body': body,
            'send_time': send_time.isoformat(),
            'status': 'scheduled'
        }
        
        logger.info(f"Email scheduled for {send_time}: {email_id}")
        return email_id
    
    def create_email_template(self, template: EmailTemplate) -> bool:
        """
        Create new email template
        
        Args:
            template: EmailTemplate object
            
        Returns:
            True if created successfully, False otherwise
        """
        try:
            self.email_templates[template.template_id] = template
            logger.info(f"Email template '{template.template_id}' created")
            return True
        except Exception as e:
            logger.error(f"Failed to create email template: {e}")
            return False
    
    def get_email_template(self, template_id: str) -> Optional[EmailTemplate]:
        """Get email template by ID"""
        return self.email_templates.get(template_id)
    
    def get_all_templates(self) -> List[EmailTemplate]:
        """Get all email templates"""
        return list(self.email_templates.values())
    
    def get_email_history(self, to_email: str) -> List[EmailMessage]:
        """Get email history for an address"""
        return [email for email in self.sent_emails if email.to_email == to_email]
    
    def get_all_sent_emails(self) -> List[EmailMessage]:
        """Get all sent emails"""
        return self.sent_emails.copy()

# Example usage
async def main():
    """Example usage of email integration"""
    
    config = {
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your_email@gmail.com',
            'password': 'your_app_password',
            'from_email': 'your_email@gmail.com'
        }
    }
    
    email = EmailIntegration(config)
    
    # Send simple email
    success = await email.send_email(
        to_email="customer@example.com",
        subject="Test Email",
        body="This is a test email from AI calling system."
    )
    print(f"Email sent: {success}")
    
    # Send template email
    template_success = await email.send_template_email(
        to_email="customer@example.com",
        template_id="appointment_confirmation",
        variables={
            'contact_name': 'John Doe',
            'appointment_time': '2:00 PM',
            'appointment_date': 'Tomorrow',
            'appointment_location': 'Our Office',
            'contact_phone': '+1234567890'
        }
    )
    print(f"Template email sent: {template_success}")
    
    # Send AI-generated email
    ai_success = await email.send_ai_generated_email(
        to_email="customer@example.com",
        context="Your appointment is confirmed for tomorrow at 2 PM.",
        email_type="appointment_confirmation"
    )
    print(f"AI email sent: {ai_success}")

if __name__ == "__main__":
    asyncio.run(main())
