"""
Analytics Engine Module - Call analytics and insights generation
==============================================================

This module provides comprehensive analytics for AI calling interactions including:
- Call performance metrics
- Sentiment analysis and trends
- Conversation quality assessment
- Predictive insights
- Real-time monitoring
"""

import asyncio
import logging
import json
import statistics
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, Counter

from .ai_calling_agent import CallSession, CallStatus
from .conversation_manager import ConversationContext, ConversationState
from .nlp_processor import SentimentType, IntentType

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of analytics metrics"""
    PERFORMANCE = "performance"
    SENTIMENT = "sentiment"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    PREDICTIVE = "predictive"

@dataclass
class CallMetrics:
    """Call performance metrics"""
    session_id: str
    duration_seconds: float
    turn_count: int
    completion_rate: float
    success_indicators: List[str]
    failure_indicators: List[str]
    escalation_count: int
    human_intervention_required: bool
    call_quality_score: float
    customer_satisfaction_score: float
    agent_efficiency_score: float

@dataclass
class SentimentMetrics:
    """Sentiment analysis metrics"""
    session_id: str
    overall_sentiment: SentimentType
    sentiment_trend: List[Tuple[float, SentimentType]]  # (timestamp, sentiment)
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    sentiment_volatility: float
    emotional_peaks: List[Tuple[float, str]]  # (timestamp, emotion)
    sentiment_keywords: Dict[str, int]

@dataclass
class ConversationQualityMetrics:
    """Conversation quality assessment"""
    session_id: str
    clarity_score: float
    engagement_score: float
    resolution_score: float
    politeness_score: float
    efficiency_score: float
    overall_quality_score: float
    improvement_areas: List[str]
    strengths: List[str]

@dataclass
class PredictiveInsights:
    """Predictive analytics insights"""
    session_id: str
    success_probability: float
    escalation_risk: float
    customer_satisfaction_prediction: float
    recommended_actions: List[str]
    risk_factors: List[str]
    opportunity_indicators: List[str]

@dataclass
class AnalyticsReport:
    """Comprehensive analytics report"""
    session_id: str
    generated_at: datetime
    call_metrics: CallMetrics
    sentiment_metrics: SentimentMetrics
    quality_metrics: ConversationQualityMetrics
    predictive_insights: PredictiveInsights
    summary: str
    recommendations: List[str]

class AnalyticsEngine:
    """
    Analytics engine for AI calling system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize analytics engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.analytics_config = config.get('analytics', {})
        
        # Analytics storage (in production, this would be a database)
        self.call_metrics: Dict[str, CallMetrics] = {}
        self.sentiment_metrics: Dict[str, SentimentMetrics] = {}
        self.quality_metrics: Dict[str, ConversationQualityMetrics] = {}
        self.predictive_insights: Dict[str, PredictiveInsights] = {}
        
        # Historical data for trend analysis
        self.historical_data: List[Dict[str, Any]] = []
        
        # Real-time monitoring
        self.real_time_metrics: Dict[str, Any] = {}
        
        # Quality assessment rules
        self.quality_rules = self._load_quality_rules()
        
        # Sentiment analysis patterns
        self.sentiment_patterns = self._load_sentiment_patterns()
        
        logger.info("Analytics Engine initialized successfully")
    
    def _load_quality_rules(self) -> Dict[str, Any]:
        """Load conversation quality assessment rules"""
        return {
            'clarity_indicators': {
                'positive': ['clear', 'understand', 'explain', 'clarify'],
                'negative': ['confused', 'unclear', 'misunderstand', 'unclear']
            },
            'engagement_indicators': {
                'positive': ['interested', 'engaged', 'participate', 'involved'],
                'negative': ['bored', 'disinterested', 'unresponsive', 'silent']
            },
            'politeness_indicators': {
                'positive': ['please', 'thank you', 'appreciate', 'respectful'],
                'negative': ['rude', 'impolite', 'disrespectful', 'offensive']
            },
            'resolution_indicators': {
                'positive': ['resolved', 'solved', 'completed', 'satisfied'],
                'negative': ['unresolved', 'unsolved', 'incomplete', 'dissatisfied']
            }
        }
    
    def _load_sentiment_patterns(self) -> Dict[str, List[str]]:
        """Load sentiment analysis patterns"""
        return {
            'positive_keywords': [
                'excellent', 'great', 'wonderful', 'amazing', 'fantastic',
                'satisfied', 'happy', 'pleased', 'delighted', 'thrilled'
            ],
            'negative_keywords': [
                'terrible', 'awful', 'horrible', 'disappointed', 'frustrated',
                'angry', 'upset', 'annoyed', 'irritated', 'disgusted'
            ],
            'neutral_keywords': [
                'okay', 'fine', 'acceptable', 'average', 'normal',
                'standard', 'regular', 'typical', 'usual', 'common'
            ]
        }
    
    async def analyze_call(self, session: CallSession) -> Dict[str, Any]:
        """
        Analyze a completed call session
        
        Args:
            session: Call session to analyze
            
        Returns:
            Comprehensive analytics data
        """
        try:
            # Calculate call metrics
            call_metrics = await self._calculate_call_metrics(session)
            self.call_metrics[session.session_id] = call_metrics
            
            # Calculate sentiment metrics
            sentiment_metrics = await self._calculate_sentiment_metrics(session)
            self.sentiment_metrics[session.session_id] = sentiment_metrics
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(session)
            self.quality_metrics[session.session_id] = quality_metrics
            
            # Generate predictive insights
            predictive_insights = await self._generate_predictive_insights(session, call_metrics, sentiment_metrics, quality_metrics)
            self.predictive_insights[session.session_id] = predictive_insights
            
            # Create comprehensive report
            report = await self._create_analytics_report(
                session, call_metrics, sentiment_metrics, quality_metrics, predictive_insights
            )
            
            # Store in historical data
            self.historical_data.append({
                'session_id': session.session_id,
                'timestamp': datetime.now(),
                'metrics': asdict(call_metrics),
                'sentiment': asdict(sentiment_metrics),
                'quality': asdict(quality_metrics),
                'predictions': asdict(predictive_insights)
            })
            
            # Update real-time metrics
            self._update_real_time_metrics(session, call_metrics)
            
            logger.info(f"Analytics completed for session {session.session_id}")
            return asdict(report)
            
        except Exception as e:
            logger.error(f"Error analyzing call {session.session_id}: {e}")
            return {}
    
    async def _calculate_call_metrics(self, session: CallSession) -> CallMetrics:
        """Calculate call performance metrics"""
        
        # Calculate duration
        if session.end_time:
            duration = (session.end_time - session.start_time).total_seconds()
        else:
            duration = 0
        
        # Count turns
        turn_count = len(session.conversation_log)
        
        # Calculate completion rate
        completion_rate = 1.0 if session.status == CallStatus.COMPLETED else 0.0
        
        # Count escalations and human interventions
        escalation_count = sum(1 for turn in session.conversation_log 
                             if turn.get('escalation_required', False))
        human_intervention_required = escalation_count > 0
        
        # Calculate call quality score (0-1)
        call_quality_score = self._calculate_call_quality_score(session)
        
        # Calculate customer satisfaction score (0-1)
        customer_satisfaction_score = self._calculate_satisfaction_score(session)
        
        # Calculate agent efficiency score (0-1)
        agent_efficiency_score = self._calculate_efficiency_score(session, duration, turn_count)
        
        return CallMetrics(
            session_id=session.session_id,
            duration_seconds=duration,
            turn_count=turn_count,
            completion_rate=completion_rate,
            success_indicators=session.analytics.get('success_indicators', []),
            failure_indicators=session.analytics.get('failure_indicators', []),
            escalation_count=escalation_count,
            human_intervention_required=human_intervention_required,
            call_quality_score=call_quality_score,
            customer_satisfaction_score=customer_satisfaction_score,
            agent_efficiency_score=agent_efficiency_score
        )
    
    def _calculate_call_quality_score(self, session: CallSession) -> float:
        """Calculate call quality score based on various factors"""
        score = 0.0
        
        # Base score for completion
        if session.status == CallStatus.COMPLETED:
            score += 0.3
        
        # Score for conversation length (optimal range)
        turn_count = len(session.conversation_log)
        if 3 <= turn_count <= 10:  # Optimal conversation length
            score += 0.2
        elif turn_count > 10:  # Too long
            score += 0.1
        
        # Score for lack of escalations
        escalation_count = sum(1 for turn in session.conversation_log 
                             if turn.get('escalation_required', False))
        if escalation_count == 0:
            score += 0.3
        elif escalation_count == 1:
            score += 0.1
        
        # Score for positive sentiment trend
        positive_turns = sum(1 for turn in session.conversation_log 
                           if turn.get('sentiment') == 'positive')
        if positive_turns > len(session.conversation_log) / 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_satisfaction_score(self, session: CallSession) -> float:
        """Calculate customer satisfaction score"""
        if not session.conversation_log:
            return 0.5
        
        # Analyze sentiment trends
        sentiments = [turn.get('sentiment', 'neutral') for turn in session.conversation_log]
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        total_count = len(sentiments)
        
        if total_count == 0:
            return 0.5
        
        # Calculate satisfaction based on sentiment ratio
        satisfaction = (positive_count - negative_count) / total_count
        return max(0.0, min(1.0, (satisfaction + 1) / 2))  # Normalize to 0-1
    
    def _calculate_efficiency_score(self, session: CallSession, duration: float, turn_count: int) -> float:
        """Calculate agent efficiency score"""
        if duration == 0 or turn_count == 0:
            return 0.5
        
        # Efficiency based on turns per minute
        turns_per_minute = turn_count / (duration / 60)
        
        # Optimal range: 2-5 turns per minute
        if 2 <= turns_per_minute <= 5:
            efficiency = 1.0
        elif turns_per_minute < 2:
            efficiency = turns_per_minute / 2  # Too slow
        else:
            efficiency = max(0.0, 1.0 - (turns_per_minute - 5) / 5)  # Too fast
        
        return efficiency
    
    async def _calculate_sentiment_metrics(self, session: CallSession) -> SentimentMetrics:
        """Calculate sentiment analysis metrics"""
        
        if not session.conversation_log:
            return SentimentMetrics(
                session_id=session.session_id,
                overall_sentiment=SentimentType.NEUTRAL,
                sentiment_trend=[],
                positive_ratio=0.0,
                negative_ratio=0.0,
                neutral_ratio=1.0,
                sentiment_volatility=0.0,
                emotional_peaks=[],
                sentiment_keywords={}
            )
        
        # Extract sentiment data
        sentiments = []
        sentiment_trend = []
        sentiment_keywords = defaultdict(int)
        
        for i, turn in enumerate(session.conversation_log):
            sentiment = turn.get('sentiment', 'neutral')
            timestamp = i  # Use turn index as timestamp
            
            sentiments.append(sentiment)
            sentiment_trend.append((timestamp, SentimentType(sentiment)))
            
            # Extract keywords from user input
            user_input = turn.get('user_input', '')
            for word in user_input.lower().split():
                if word in self.sentiment_patterns['positive_keywords']:
                    sentiment_keywords[word] += 1
                elif word in self.sentiment_patterns['negative_keywords']:
                    sentiment_keywords[word] += 1
        
        # Calculate ratios
        total_turns = len(sentiments)
        positive_ratio = sentiments.count('positive') / total_turns
        negative_ratio = sentiments.count('negative') / total_turns
        neutral_ratio = sentiments.count('neutral') / total_turns
        
        # Determine overall sentiment
        if positive_ratio > negative_ratio and positive_ratio > neutral_ratio:
            overall_sentiment = SentimentType.POSITIVE
        elif negative_ratio > positive_ratio and negative_ratio > neutral_ratio:
            overall_sentiment = SentimentType.NEGATIVE
        else:
            overall_sentiment = SentimentType.NEUTRAL
        
        # Calculate sentiment volatility
        sentiment_volatility = self._calculate_sentiment_volatility(sentiment_trend)
        
        # Find emotional peaks
        emotional_peaks = self._find_emotional_peaks(sentiment_trend)
        
        return SentimentMetrics(
            session_id=session.session_id,
            overall_sentiment=overall_sentiment,
            sentiment_trend=sentiment_trend,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            neutral_ratio=neutral_ratio,
            sentiment_volatility=sentiment_volatility,
            emotional_peaks=emotional_peaks,
            sentiment_keywords=dict(sentiment_keywords)
        )
    
    def _calculate_sentiment_volatility(self, sentiment_trend: List[Tuple[float, SentimentType]]) -> float:
        """Calculate sentiment volatility"""
        if len(sentiment_trend) < 2:
            return 0.0
        
        # Convert sentiments to numeric values
        sentiment_values = []
        for _, sentiment in sentiment_trend:
            if sentiment == SentimentType.POSITIVE:
                sentiment_values.append(1.0)
            elif sentiment == SentimentType.NEGATIVE:
                sentiment_values.append(-1.0)
            else:
                sentiment_values.append(0.0)
        
        # Calculate standard deviation
        if len(sentiment_values) > 1:
            return statistics.stdev(sentiment_values)
        return 0.0
    
    def _find_emotional_peaks(self, sentiment_trend: List[Tuple[float, SentimentType]]) -> List[Tuple[float, str]]:
        """Find emotional peaks in conversation"""
        peaks = []
        
        for i, (timestamp, sentiment) in enumerate(sentiment_trend):
            if sentiment == SentimentType.POSITIVE:
                # Check if this is a peak (higher than neighbors)
                if (i == 0 or sentiment_trend[i-1][1] != SentimentType.POSITIVE) and \
                   (i == len(sentiment_trend)-1 or sentiment_trend[i+1][1] != SentimentType.POSITIVE):
                    peaks.append((timestamp, "positive_peak"))
            elif sentiment == SentimentType.NEGATIVE:
                # Check if this is a peak (lower than neighbors)
                if (i == 0 or sentiment_trend[i-1][1] != SentimentType.NEGATIVE) and \
                   (i == len(sentiment_trend)-1 or sentiment_trend[i+1][1] != SentimentType.NEGATIVE):
                    peaks.append((timestamp, "negative_peak"))
        
        return peaks
    
    async def _calculate_quality_metrics(self, session: CallSession) -> ConversationQualityMetrics:
        """Calculate conversation quality metrics"""
        
        # Analyze conversation text for quality indicators
        all_text = " ".join([turn.get('user_input', '') for turn in session.conversation_log])
        all_text_lower = all_text.lower()
        
        # Calculate clarity score
        clarity_score = self._calculate_clarity_score(all_text_lower)
        
        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(session.conversation_log)
        
        # Calculate resolution score
        resolution_score = self._calculate_resolution_score(session)
        
        # Calculate politeness score
        politeness_score = self._calculate_politeness_score(all_text_lower)
        
        # Calculate efficiency score
        efficiency_score = self._calculate_conversation_efficiency(session)
        
        # Calculate overall quality score
        overall_quality_score = (
            clarity_score + engagement_score + resolution_score + 
            politeness_score + efficiency_score
        ) / 5
        
        # Identify improvement areas and strengths
        improvement_areas = self._identify_improvement_areas(
            clarity_score, engagement_score, resolution_score, 
            politeness_score, efficiency_score
        )
        strengths = self._identify_strengths(
            clarity_score, engagement_score, resolution_score, 
            politeness_score, efficiency_score
        )
        
        return ConversationQualityMetrics(
            session_id=session.session_id,
            clarity_score=clarity_score,
            engagement_score=engagement_score,
            resolution_score=resolution_score,
            politeness_score=politeness_score,
            efficiency_score=efficiency_score,
            overall_quality_score=overall_quality_score,
            improvement_areas=improvement_areas,
            strengths=strengths
        )
    
    def _calculate_clarity_score(self, text: str) -> float:
        """Calculate clarity score based on text analysis"""
        positive_indicators = self.quality_rules['clarity_indicators']['positive']
        negative_indicators = self.quality_rules['clarity_indicators']['negative']
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in text)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text)
        
        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            return 0.5  # Neutral score
        
        return positive_count / total_indicators
    
    def _calculate_engagement_score(self, conversation_log: List[Dict]) -> float:
        """Calculate engagement score based on conversation patterns"""
        if not conversation_log:
            return 0.0
        
        # Factors that indicate engagement
        engagement_factors = 0
        total_factors = 0
        
        # Check for questions and responses
        for turn in conversation_log:
            user_input = turn.get('user_input', '')
            if '?' in user_input:
                engagement_factors += 1
            total_factors += 1
        
        # Check for conversation length (longer = more engaged)
        if len(conversation_log) > 3:
            engagement_factors += 1
        total_factors += 1
        
        return engagement_factors / total_factors if total_factors > 0 else 0.0
    
    def _calculate_resolution_score(self, session: CallSession) -> float:
        """Calculate resolution score based on call outcome"""
        if session.status == CallStatus.COMPLETED:
            return 1.0
        elif session.status == CallStatus.FAILED:
            return 0.0
        else:
            return 0.5
    
    def _calculate_politeness_score(self, text: str) -> float:
        """Calculate politeness score based on text analysis"""
        positive_indicators = self.quality_rules['politeness_indicators']['positive']
        negative_indicators = self.quality_rules['politeness_indicators']['negative']
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in text)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text)
        
        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            return 0.5  # Neutral score
        
        return positive_count / total_indicators
    
    def _calculate_conversation_efficiency(self, session: CallSession) -> float:
        """Calculate conversation efficiency score"""
        if not session.conversation_log:
            return 0.0
        
        # Efficiency based on achieving goals with minimal turns
        turn_count = len(session.conversation_log)
        
        # Optimal range: 3-8 turns
        if 3 <= turn_count <= 8:
            return 1.0
        elif turn_count < 3:
            return turn_count / 3  # Too short
        else:
            return max(0.0, 1.0 - (turn_count - 8) / 10)  # Too long
    
    def _identify_improvement_areas(self, *scores) -> List[str]:
        """Identify areas for improvement based on quality scores"""
        areas = []
        score_names = ['clarity', 'engagement', 'resolution', 'politeness', 'efficiency']
        
        for i, score in enumerate(scores):
            if score < 0.6:  # Threshold for improvement
                areas.append(f"Improve {score_names[i]} (score: {score:.2f})")
        
        return areas
    
    def _identify_strengths(self, *scores) -> List[str]:
        """Identify conversation strengths based on quality scores"""
        strengths = []
        score_names = ['clarity', 'engagement', 'resolution', 'politeness', 'efficiency']
        
        for i, score in enumerate(scores):
            if score > 0.8:  # Threshold for strength
                strengths.append(f"Strong {score_names[i]} (score: {score:.2f})")
        
        return strengths
    
    async def _generate_predictive_insights(self, 
                                          session: CallSession,
                                          call_metrics: CallMetrics,
                                          sentiment_metrics: SentimentMetrics,
                                          quality_metrics: ConversationQualityMetrics) -> PredictiveInsights:
        """Generate predictive insights based on analytics"""
        
        # Calculate success probability
        success_probability = self._calculate_success_probability(call_metrics, sentiment_metrics, quality_metrics)
        
        # Calculate escalation risk
        escalation_risk = self._calculate_escalation_risk(call_metrics, sentiment_metrics)
        
        # Predict customer satisfaction
        satisfaction_prediction = self._predict_customer_satisfaction(sentiment_metrics, quality_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(call_metrics, sentiment_metrics, quality_metrics)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(call_metrics, sentiment_metrics, quality_metrics)
        
        # Identify opportunities
        opportunities = self._identify_opportunities(call_metrics, sentiment_metrics, quality_metrics)
        
        return PredictiveInsights(
            session_id=session.session_id,
            success_probability=success_probability,
            escalation_risk=escalation_risk,
            customer_satisfaction_prediction=satisfaction_prediction,
            recommended_actions=recommendations,
            risk_factors=risk_factors,
            opportunity_indicators=opportunities
        )
    
    def _calculate_success_probability(self, call_metrics: CallMetrics, 
                                     sentiment_metrics: SentimentMetrics, 
                                     quality_metrics: ConversationQualityMetrics) -> float:
        """Calculate probability of call success"""
        factors = [
            call_metrics.completion_rate,
            call_metrics.call_quality_score,
            sentiment_metrics.positive_ratio,
            quality_metrics.overall_quality_score
        ]
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        return sum(factor * weight for factor, weight in zip(factors, weights))
    
    def _calculate_escalation_risk(self, call_metrics: CallMetrics, 
                                 sentiment_metrics: SentimentMetrics) -> float:
        """Calculate risk of escalation"""
        risk_factors = []
        
        # Escalation history
        if call_metrics.escalation_count > 0:
            risk_factors.append(0.8)
        
        # Negative sentiment
        if sentiment_metrics.negative_ratio > 0.3:
            risk_factors.append(0.6)
        
        # High sentiment volatility
        if sentiment_metrics.sentiment_volatility > 0.5:
            risk_factors.append(0.4)
        
        return max(risk_factors) if risk_factors else 0.1
    
    def _predict_customer_satisfaction(self, sentiment_metrics: SentimentMetrics, 
                                     quality_metrics: ConversationQualityMetrics) -> float:
        """Predict customer satisfaction"""
        return (sentiment_metrics.positive_ratio + quality_metrics.overall_quality_score) / 2
    
    def _generate_recommendations(self, call_metrics: CallMetrics, 
                                sentiment_metrics: SentimentMetrics, 
                                quality_metrics: ConversationQualityMetrics) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if call_metrics.call_quality_score < 0.6:
            recommendations.append("Focus on improving call quality through better conversation flow")
        
        if sentiment_metrics.negative_ratio > 0.3:
            recommendations.append("Address customer concerns more proactively")
        
        if quality_metrics.clarity_score < 0.6:
            recommendations.append("Improve communication clarity and explanation")
        
        if call_metrics.escalation_count > 0:
            recommendations.append("Train agents on handling difficult situations")
        
        return recommendations
    
    def _identify_risk_factors(self, call_metrics: CallMetrics, 
                             sentiment_metrics: SentimentMetrics, 
                             quality_metrics: ConversationQualityMetrics) -> List[str]:
        """Identify risk factors"""
        risks = []
        
        if call_metrics.human_intervention_required:
            risks.append("Requires human intervention")
        
        if sentiment_metrics.overall_sentiment == SentimentType.NEGATIVE:
            risks.append("Negative customer sentiment")
        
        if quality_metrics.overall_quality_score < 0.5:
            risks.append("Low conversation quality")
        
        return risks
    
    def _identify_opportunities(self, call_metrics: CallMetrics, 
                              sentiment_metrics: SentimentMetrics, 
                              quality_metrics: ConversationQualityMetrics) -> List[str]:
        """Identify opportunities for improvement"""
        opportunities = []
        
        if call_metrics.agent_efficiency_score > 0.8:
            opportunities.append("High efficiency - can handle more complex cases")
        
        if sentiment_metrics.positive_ratio > 0.7:
            opportunities.append("Positive customer sentiment - opportunity for upselling")
        
        if quality_metrics.overall_quality_score > 0.8:
            opportunities.append("High quality conversation - use as training example")
        
        return opportunities
    
    async def _create_analytics_report(self, 
                                     session: CallSession,
                                     call_metrics: CallMetrics,
                                     sentiment_metrics: SentimentMetrics,
                                     quality_metrics: ConversationQualityMetrics,
                                     predictive_insights: PredictiveInsights) -> AnalyticsReport:
        """Create comprehensive analytics report"""
        
        # Generate summary
        summary = self._generate_summary(call_metrics, sentiment_metrics, quality_metrics)
        
        # Generate recommendations
        recommendations = self._generate_report_recommendations(
            call_metrics, sentiment_metrics, quality_metrics, predictive_insights
        )
        
        return AnalyticsReport(
            session_id=session.session_id,
            generated_at=datetime.now(),
            call_metrics=call_metrics,
            sentiment_metrics=sentiment_metrics,
            quality_metrics=quality_metrics,
            predictive_insights=predictive_insights,
            summary=summary,
            recommendations=recommendations
        )
    
    def _generate_summary(self, call_metrics: CallMetrics, 
                         sentiment_metrics: SentimentMetrics, 
                         quality_metrics: ConversationQualityMetrics) -> str:
        """Generate executive summary"""
        summary_parts = []
        
        # Call outcome
        if call_metrics.completion_rate > 0.8:
            summary_parts.append("Call completed successfully")
        else:
            summary_parts.append("Call did not complete successfully")
        
        # Duration and efficiency
        duration_min = call_metrics.duration_seconds / 60
        summary_parts.append(f"Duration: {duration_min:.1f} minutes with {call_metrics.turn_count} turns")
        
        # Sentiment
        if sentiment_metrics.overall_sentiment == SentimentType.POSITIVE:
            summary_parts.append("Customer sentiment was positive")
        elif sentiment_metrics.overall_sentiment == SentimentType.NEGATIVE:
            summary_parts.append("Customer sentiment was negative")
        else:
            summary_parts.append("Customer sentiment was neutral")
        
        # Quality
        if quality_metrics.overall_quality_score > 0.7:
            summary_parts.append("High conversation quality")
        else:
            summary_parts.append("Conversation quality needs improvement")
        
        return ". ".join(summary_parts) + "."
    
    def _generate_report_recommendations(self, call_metrics: CallMetrics, 
                                       sentiment_metrics: SentimentMetrics, 
                                       quality_metrics: ConversationQualityMetrics,
                                       predictive_insights: PredictiveInsights) -> List[str]:
        """Generate report recommendations"""
        recommendations = []
        
        # Add predictive insights recommendations
        recommendations.extend(predictive_insights.recommended_actions)
        
        # Add quality-based recommendations
        if quality_metrics.improvement_areas:
            recommendations.extend(quality_metrics.improvement_areas)
        
        # Add escalation recommendations
        if call_metrics.escalation_count > 0:
            recommendations.append("Review escalation triggers and improve handling")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _update_real_time_metrics(self, session: CallSession, call_metrics: CallMetrics):
        """Update real-time monitoring metrics"""
        current_time = datetime.now()
        
        # Update daily metrics
        today = current_time.date()
        if 'daily_calls' not in self.real_time_metrics:
            self.real_time_metrics['daily_calls'] = {}
        
        if today not in self.real_time_metrics['daily_calls']:
            self.real_time_metrics['daily_calls'][today] = {
                'total_calls': 0,
                'completed_calls': 0,
                'average_duration': 0,
                'average_quality': 0
            }
        
        daily_metrics = self.real_time_metrics['daily_calls'][today]
        daily_metrics['total_calls'] += 1
        
        if call_metrics.completion_rate > 0.8:
            daily_metrics['completed_calls'] += 1
        
        # Update averages
        daily_metrics['average_duration'] = (
            (daily_metrics['average_duration'] * (daily_metrics['total_calls'] - 1) + 
             call_metrics.duration_seconds) / daily_metrics['total_calls']
        )
        
        daily_metrics['average_quality'] = (
            (daily_metrics['average_quality'] * (daily_metrics['total_calls'] - 1) + 
             call_metrics.call_quality_score) / daily_metrics['total_calls']
        )
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        return self.real_time_metrics.copy()
    
    def get_historical_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get historical trends for specified number of days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_data = [
            entry for entry in self.historical_data 
            if entry['timestamp'] >= cutoff_date
        ]
        
        if not recent_data:
            return {}
        
        # Calculate trends
        trends = {
            'total_calls': len(recent_data),
            'average_duration': statistics.mean([entry['metrics']['duration_seconds'] for entry in recent_data]),
            'average_quality': statistics.mean([entry['metrics']['call_quality_score'] for entry in recent_data]),
            'completion_rate': statistics.mean([entry['metrics']['completion_rate'] for entry in recent_data]),
            'escalation_rate': statistics.mean([1 if entry['metrics']['escalation_count'] > 0 else 0 for entry in recent_data])
        }
        
        return trends
    
    def export_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export analytics data for a specific session"""
        if session_id not in self.call_metrics:
            return None
        
        return {
            'call_metrics': asdict(self.call_metrics[session_id]),
            'sentiment_metrics': asdict(self.sentiment_metrics.get(session_id)),
            'quality_metrics': asdict(self.quality_metrics.get(session_id)),
            'predictive_insights': asdict(self.predictive_insights.get(session_id))
        }

# Example usage and testing
async def main():
    """Example usage of the Analytics Engine"""
    
    config = {
        'analytics': {
            'enable_real_time': True,
            'trend_analysis_days': 7
        }
    }
    
    engine = AnalyticsEngine(config)
    
    # Create a mock call session for testing
    from .ai_calling_agent import CallSession, CallStatus
    from datetime import datetime
    
    session = CallSession(
        session_id="test_session_001",
        phone_number="+1234567890",
        contact_name="John Doe",
        call_script={},
        start_time=datetime.now(),
        end_time=datetime.now(),
        status=CallStatus.COMPLETED,
        conversation_log=[
            {
                'turn_number': 1,
                'timestamp': datetime.now().isoformat(),
                'user_input': "Hello, I'd like to book an appointment",
                'intent': 'greeting',
                'sentiment': 'positive',
                'confidence': 0.8
            },
            {
                'turn_number': 2,
                'timestamp': datetime.now().isoformat(),
                'user_input': "Tomorrow at 2 PM works for me",
                'intent': 'appointment',
                'sentiment': 'positive',
                'confidence': 0.9
            },
            {
                'turn_number': 3,
                'timestamp': datetime.now().isoformat(),
                'user_input': "Yes, that's perfect. Thank you!",
                'intent': 'confirmation',
                'sentiment': 'positive',
                'confidence': 0.9
            }
        ],
        analytics={}
    )
    
    print("Testing Analytics Engine:")
    print("=" * 50)
    
    # Analyze the call
    analytics_data = await engine.analyze_call(session)
    
    if analytics_data:
        print(f"Call Metrics:")
        print(f"- Duration: {analytics_data['call_metrics']['duration_seconds']:.1f} seconds")
        print(f"- Turn Count: {analytics_data['call_metrics']['turn_count']}")
        print(f"- Completion Rate: {analytics_data['call_metrics']['completion_rate']:.2f}")
        print(f"- Quality Score: {analytics_data['call_metrics']['call_quality_score']:.2f}")
        print(f"- Satisfaction Score: {analytics_data['call_metrics']['customer_satisfaction_score']:.2f}")
        
        print(f"\nSentiment Metrics:")
        print(f"- Overall Sentiment: {analytics_data['sentiment_metrics']['overall_sentiment']}")
        print(f"- Positive Ratio: {analytics_data['sentiment_metrics']['positive_ratio']:.2f}")
        print(f"- Negative Ratio: {analytics_data['sentiment_metrics']['negative_ratio']:.2f}")
        
        print(f"\nQuality Metrics:")
        print(f"- Overall Quality: {analytics_data['quality_metrics']['overall_quality_score']:.2f}")
        print(f"- Clarity Score: {analytics_data['quality_metrics']['clarity_score']:.2f}")
        print(f"- Engagement Score: {analytics_data['quality_metrics']['engagement_score']:.2f}")
        
        print(f"\nPredictive Insights:")
        print(f"- Success Probability: {analytics_data['predictive_insights']['success_probability']:.2f}")
        print(f"- Escalation Risk: {analytics_data['predictive_insights']['escalation_risk']:.2f}")
        print(f"- Satisfaction Prediction: {analytics_data['predictive_insights']['customer_satisfaction_prediction']:.2f}")
        
        print(f"\nSummary:")
        print(analytics_data['summary'])
        
        print(f"\nRecommendations:")
        for rec in analytics_data['recommendations']:
            print(f"- {rec}")
    
    # Test real-time metrics
    real_time = engine.get_real_time_metrics()
    print(f"\nReal-time Metrics:")
    print(f"Daily calls: {real_time}")
    
    print("\nAnalytics Engine test completed")

if __name__ == "__main__":
    asyncio.run(main())
