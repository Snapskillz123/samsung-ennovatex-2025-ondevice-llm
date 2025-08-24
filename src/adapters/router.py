"""
Adapter Router for Samsung EnnovateX 2025 AI Challenge
Intelligent routing system to select appropriate adapters for user queries
"""

from typing import List, Dict, Optional, Callable
from enum import Enum
import re
import logging
from dataclasses import dataclass

from .protocol import Domain

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Different routing strategies."""
    UI_CONTEXT = "ui_context"
    KEYWORDS = "keywords" 
    CLASSIFIER = "classifier"
    HYBRID = "hybrid"

@dataclass
class RoutingResult:
    """Result of adapter routing."""
    adapter_ids: List[str]
    confidence_scores: Dict[str, float]
    reasoning: str

class KeywordRouter:
    """Keyword-based adapter routing."""
    
    def __init__(self):
        # Domain-specific keywords
        self.domain_keywords = {
            Domain.COMMUNICATION: [
                # Casual/texting
                "hey", "hi", "hello", "lol", "brb", "omg", "wtf", "tbh", "imo",
                "text", "message", "chat", "whatsapp", "telegram", "dm",
                # Emotions/reactions
                "😂", "😊", "😍", "👍", "❤️", "🔥", "💯",
                # Slang
                "gonna", "wanna", "gotta", "dunno", "kinda", "sorta"
            ],
            
            Domain.CALENDAR: [
                # Time references
                "meet", "meeting", "schedule", "appointment", "calendar",
                "tomorrow", "today", "next week", "monday", "tuesday", 
                "wednesday", "thursday", "friday", "saturday", "sunday",
                "morning", "afternoon", "evening", "pm", "am",
                # Calendar actions
                "book", "reschedule", "cancel", "confirm", "remind",
                "available", "busy", "free", "slot", "time"
            ],
            
            Domain.NOTES: [
                # Note-taking
                "note", "notes", "remember", "jot down", "write down",
                "summary", "summarize", "bullet", "outline", "list",
                "todo", "task", "action item", "follow up",
                # Organization
                "organize", "structure", "categorize", "tag", "label"
            ]
        }
        
        # Compile regex patterns for efficiency
        self.domain_patterns = {}
        for domain, keywords in self.domain_keywords.items():
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.domain_patterns[domain] = re.compile(pattern, re.IGNORECASE)
    
    def route(self, prompt: str, max_adapters: int = 2) -> RoutingResult:
        """Route based on keyword matching."""
        scores = {}
        
        for domain, pattern in self.domain_patterns.items():
            matches = pattern.findall(prompt.lower())
            # Score based on number of matches and total keyword coverage
            score = len(matches) / max(len(prompt.split()), 1)  # Normalize by prompt length
            scores[domain] = score
        
        # Sort by score and take top domains
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_domains = [domain for domain, score in sorted_domains[:max_adapters] if score > 0]
        
        # Convert to adapter IDs (assumes naming convention)
        adapter_ids = []
        for domain in top_domains:
            if domain == Domain.COMMUNICATION:
                adapter_ids.append("comm_v1")
            elif domain == Domain.CALENDAR:
                adapter_ids.append("cal_v1")
            elif domain == Domain.NOTES:
                adapter_ids.append("notes_v1")
        
        confidence_scores = {
            adapter_id: scores.get(self._adapter_to_domain(adapter_id), 0.0)
            for adapter_id in adapter_ids
        }
        
        reasoning = f"Keyword matching found: {[d.value for d in top_domains]}"
        
        return RoutingResult(
            adapter_ids=adapter_ids,
            confidence_scores=confidence_scores,
            reasoning=reasoning
        )
    
    def _adapter_to_domain(self, adapter_id: str) -> Domain:
        """Map adapter ID back to domain."""
        if "comm" in adapter_id:
            return Domain.COMMUNICATION
        elif "cal" in adapter_id:
            return Domain.CALENDAR
        elif "notes" in adapter_id:
            return Domain.NOTES
        else:
            return Domain.GENERAL

class UIContextRouter:
    """UI context-based routing."""
    
    def __init__(self):
        # Map UI contexts to adapter IDs
        self.ui_mappings = {
            "messaging": ["comm_v1"],
            "whatsapp": ["comm_v1"],
            "telegram": ["comm_v1"],
            "chat": ["comm_v1"],
            "calendar": ["cal_v1"],
            "schedule": ["cal_v1"],
            "events": ["cal_v1"],
            "notes": ["notes_v1"],
            "notepad": ["notes_v1"],
            "memo": ["notes_v1"]
        }
    
    def route(self, ui_context: Optional[str]) -> RoutingResult:
        """Route based on UI context."""
        if not ui_context:
            return RoutingResult(
                adapter_ids=[],
                confidence_scores={},
                reasoning="No UI context provided"
            )
        
        context_lower = ui_context.lower()
        adapter_ids = self.ui_mappings.get(context_lower, [])
        
        confidence_scores = {adapter_id: 1.0 for adapter_id in adapter_ids}
        reasoning = f"UI context '{ui_context}' mapped to adapters"
        
        return RoutingResult(
            adapter_ids=adapter_ids,
            confidence_scores=confidence_scores,
            reasoning=reasoning
        )

class HybridRouter:
    """Hybrid router combining multiple strategies."""
    
    def __init__(self):
        self.ui_router = UIContextRouter()
        self.keyword_router = KeywordRouter()
    
    def route(self, 
             prompt: str, 
             ui_context: Optional[str] = None,
             max_adapters: int = 2) -> RoutingResult:
        """Route using hybrid approach."""
        
        # Start with UI context if available
        ui_result = self.ui_router.route(ui_context)
        
        # Add keyword-based routing
        keyword_result = self.keyword_router.route(prompt, max_adapters)
        
        # Combine results, preferring UI context
        combined_adapters = []
        combined_scores = {}
        
        # UI context gets priority
        for adapter_id in ui_result.adapter_ids:
            combined_adapters.append(adapter_id)
            combined_scores[adapter_id] = ui_result.confidence_scores[adapter_id] * 1.5  # Boost UI context
        
        # Add keyword results if not already present
        for adapter_id in keyword_result.adapter_ids:
            if adapter_id not in combined_adapters and len(combined_adapters) < max_adapters:
                combined_adapters.append(adapter_id)
                combined_scores[adapter_id] = keyword_result.confidence_scores[adapter_id]
        
        # Remove duplicates while preserving order
        unique_adapters = []
        seen = set()
        for adapter_id in combined_adapters:
            if adapter_id not in seen:
                unique_adapters.append(adapter_id)
                seen.add(adapter_id)
        
        reasoning = f"Hybrid: UI({ui_result.reasoning}) + Keywords({keyword_result.reasoning})"
        
        return RoutingResult(
            adapter_ids=unique_adapters[:max_adapters],
            confidence_scores=combined_scores,
            reasoning=reasoning
        )

class AdapterRouter:
    """Main router interface."""
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.HYBRID):
        self.strategy = strategy
        
        # Initialize routers
        self.ui_router = UIContextRouter()
        self.keyword_router = KeywordRouter() 
        self.hybrid_router = HybridRouter()
        
        # Routing function map
        self._route_functions = {
            RoutingStrategy.UI_CONTEXT: self._route_ui_context,
            RoutingStrategy.KEYWORDS: self._route_keywords,
            RoutingStrategy.HYBRID: self._route_hybrid
        }
    
    def route(self, 
             prompt: str,
             ui_context: Optional[str] = None,
             max_adapters: int = 2,
             strategy: Optional[RoutingStrategy] = None) -> RoutingResult:
        """Main routing function."""
        
        active_strategy = strategy or self.strategy
        route_func = self._route_functions.get(active_strategy, self._route_hybrid)
        
        result = route_func(prompt, ui_context, max_adapters)
        
        logger.info(f"Routing result: {result.adapter_ids} (strategy: {active_strategy.value})")
        logger.debug(f"Reasoning: {result.reasoning}")
        
        return result
    
    def _route_ui_context(self, prompt: str, ui_context: Optional[str], max_adapters: int) -> RoutingResult:
        """Route using UI context only."""
        return self.ui_router.route(ui_context)
    
    def _route_keywords(self, prompt: str, ui_context: Optional[str], max_adapters: int) -> RoutingResult:
        """Route using keywords only."""
        return self.keyword_router.route(prompt, max_adapters)
    
    def _route_hybrid(self, prompt: str, ui_context: Optional[str], max_adapters: int) -> RoutingResult:
        """Route using hybrid approach."""
        return self.hybrid_router.route(prompt, ui_context, max_adapters)
    
    def explain_routing(self, result: RoutingResult) -> str:
        """Provide human-readable explanation of routing decision."""
        if not result.adapter_ids:
            return "No adapters selected - using base model"
        
        explanations = []
        for adapter_id in result.adapter_ids:
            score = result.confidence_scores.get(adapter_id, 0.0)
            domain = self._adapter_to_domain_name(adapter_id)
            explanations.append(f"{domain} adapter (confidence: {score:.2f})")
        
        return f"Selected: {', '.join(explanations)}. {result.reasoning}"
    
    def _adapter_to_domain_name(self, adapter_id: str) -> str:
        """Convert adapter ID to human-readable domain name."""
        if "comm" in adapter_id:
            return "Communication"
        elif "cal" in adapter_id:
            return "Calendar"
        elif "notes" in adapter_id:
            return "Notes"
        else:
            return "General"

# Factory function for easy instantiation
def create_router(strategy: RoutingStrategy = RoutingStrategy.HYBRID) -> AdapterRouter:
    """Create adapter router with specified strategy."""
    return AdapterRouter(strategy)
