"""
Evaluator Module - LLM-as-Judge and Metric Calculation.

Uses Gemini to evaluate response quality and detect hallucinations.
"""

import logging
import time
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single response."""
    question_id: str
    question: str
    retrieval_mode: str
    
    # Response data
    response: str
    ground_truth: str
    sources: List[Dict]
    response_time_ms: float
    
    # Metrics
    relevance_score: float  # 1-5 scale
    accuracy_score: float   # 0-1 scale
    keyword_coverage: float # 0-1 scale (expected keywords found)
    entity_coverage: float  # 0-1 scale (expected entities found)
    hallucination_detected: bool
    hallucination_details: str = ""
    
    # Context metrics
    context_length: int = 0
    source_count: int = 0
    graph_entities_found: int = 0
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "retrieval_mode": self.retrieval_mode,
            "response": self.response,
            "ground_truth": self.ground_truth,
            "sources": self.sources,
            "response_time_ms": self.response_time_ms,
            "relevance_score": self.relevance_score,
            "accuracy_score": self.accuracy_score,
            "keyword_coverage": self.keyword_coverage,
            "entity_coverage": self.entity_coverage,
            "hallucination_detected": self.hallucination_detected,
            "hallucination_details": self.hallucination_details,
            "context_length": self.context_length,
            "source_count": self.source_count,
            "graph_entities_found": self.graph_entities_found,
            "timestamp": self.timestamp
        }


class Evaluator:
    """
    Evaluator using LLM-as-Judge (Gemini) for response quality assessment.
    """
    
    def __init__(self, llm_generator=None):
        """
        Initialize the evaluator.
        
        Args:
            llm_generator: LLMGenerator instance for LLM-as-Judge scoring
        """
        self.llm_generator = llm_generator
    
    def evaluate_response(
        self,
        question_id: str,
        question: str,
        response: str,
        ground_truth: str,
        expected_entities: List[str],
        expected_keywords: List[str],
        retrieval_mode: str,
        sources: List[Dict],
        response_time_ms: float,
        context_length: int = 0,
        graph_entities_found: int = 0
    ) -> EvaluationResult:
        """
        Evaluate a single response against ground truth.
        
        Returns:
            EvaluationResult with all metrics
        """
        # Calculate keyword coverage
        keyword_coverage = self._calculate_keyword_coverage(
            response, expected_keywords
        )
        
        # Calculate entity coverage
        entity_coverage = self._calculate_entity_coverage(
            response, expected_entities
        )
        
        # Get LLM-as-Judge scores
        relevance_score, accuracy_score = self._llm_judge_scores(
            question, response, ground_truth
        )
        
        # Detect hallucinations
        time.sleep(1)  # Rate limit protection between LLM calls
        hallucination_detected, hallucination_details = self._detect_hallucination(
            question, response, ground_truth, expected_entities
        )
        
        return EvaluationResult(
            question_id=question_id,
            question=question,
            retrieval_mode=retrieval_mode,
            response=response,
            ground_truth=ground_truth,
            sources=sources,
            response_time_ms=response_time_ms,
            relevance_score=relevance_score,
            accuracy_score=accuracy_score,
            keyword_coverage=keyword_coverage,
            entity_coverage=entity_coverage,
            hallucination_detected=hallucination_detected,
            hallucination_details=hallucination_details,
            context_length=context_length,
            source_count=len(sources),
            graph_entities_found=graph_entities_found
        )
    
    def _calculate_keyword_coverage(
        self, 
        response: str, 
        expected_keywords: List[str]
    ) -> float:
        """Calculate what percentage of expected keywords appear in response."""
        if not expected_keywords:
            return 1.0
        
        response_lower = response.lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        return found / len(expected_keywords)
    
    def _calculate_entity_coverage(
        self, 
        response: str, 
        expected_entities: List[str]
    ) -> float:
        """Calculate what percentage of expected entities appear in response."""
        if not expected_entities:
            return 1.0
        
        response_lower = response.lower()
        found = sum(1 for ent in expected_entities if ent.lower() in response_lower)
        return found / len(expected_entities)
    
    def _llm_judge_scores(
        self, 
        question: str, 
        response: str, 
        ground_truth: str
    ) -> tuple:
        """
        Use LLM to judge response quality.
        
        Returns:
            (relevance_score, accuracy_score) - both 0-5 scale
        """
        if not self.llm_generator:
            # Fallback to keyword-based scoring
            return self._fallback_scoring(question, response, ground_truth)
        
        try:
            judge_prompt = f"""You are an expert evaluator. Rate the following response on two dimensions.

QUESTION: {question}

EXPECTED ANSWER (Ground Truth): {ground_truth}

ACTUAL RESPONSE: {response}

Rate the response on these two metrics (1-5 scale):

1. RELEVANCE: Does the response directly address the question?
   1 = Completely irrelevant
   2 = Tangentially related
   3 = Partially addresses the question
   4 = Mostly addresses the question
   5 = Fully addresses the question

2. ACCURACY: Is the information factually correct compared to ground truth?
   1 = Completely wrong or fabricated
   2 = Mostly incorrect
   3 = Partially correct
   4 = Mostly correct with minor errors
   5 = Fully accurate

Respond ONLY with two numbers separated by a comma, like: 4,5
Do not include any other text."""

            judge_response = self.llm_generator.generate(judge_prompt)
            
            # Parse scores
            match = re.search(r'(\d)[,\s]+(\d)', judge_response)
            if match:
                relevance = min(5, max(1, int(match.group(1))))
                accuracy = min(5, max(1, int(match.group(2))))
                return relevance, accuracy / 5.0  # Convert accuracy to 0-1 scale
            else:
                logger.warning(f"Could not parse LLM judge response: {judge_response}")
                return self._fallback_scoring(question, response, ground_truth)
                
        except Exception as e:
            logger.error(f"LLM-as-Judge failed: {e}")
            return self._fallback_scoring(question, response, ground_truth)
    
    def _fallback_scoring(
        self, 
        question: str, 
        response: str, 
        ground_truth: str
    ) -> tuple:
        """Fallback scoring based on string similarity."""
        # Simple word overlap scoring
        response_words = set(response.lower().split())
        ground_truth_words = set(ground_truth.lower().split())
        question_words = set(question.lower().split())
        
        # Relevance: Does response share words with question?
        if question_words:
            relevance_overlap = len(response_words & question_words) / len(question_words)
            relevance = min(5, max(1, int(relevance_overlap * 4) + 1))
        else:
            relevance = 3
        
        # Accuracy: Does response share words with ground truth?
        if ground_truth_words:
            accuracy_overlap = len(response_words & ground_truth_words) / len(ground_truth_words)
        else:
            accuracy_overlap = 0.5
        
        return relevance, accuracy_overlap
    
    def _detect_hallucination(
        self,
        question: str,
        response: str,
        ground_truth: str,
        expected_entities: List[str]
    ) -> tuple:
        """
        Detect if the response contains hallucinated information.
        
        Returns:
            (hallucination_detected: bool, details: str)
        """
        hallucination_indicators = []
        
        # Check for made-up prices (specific numbers not grounded)
        price_pattern = r'₹\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        response_prices = re.findall(price_pattern, response)
        ground_truth_prices = re.findall(price_pattern, ground_truth)
        
        if response_prices and not ground_truth_prices:
            # Response has specific prices but ground truth doesn't
            hallucination_indicators.append(
                f"Specific prices mentioned ({response_prices}) not in ground truth"
            )
        
        # Check for invented entities
        # Common hallucination: mentioning brands/products not in the context
        invented_entities = [
            "XYZ", "ABC Corp", "Made-up Brand", "Generic Product 9000"
        ]
        for ent in invented_entities:
            if ent.lower() in response.lower():
                hallucination_indicators.append(f"Invented entity: {ent}")
        
        # Check for contradictions with ground truth negatives
        negative_indicators = ["not found", "does not exist", "no product", "unavailable"]
        ground_has_negative = any(neg in ground_truth.lower() for neg in negative_indicators)
        response_has_positive = not any(neg in response.lower() for neg in negative_indicators)
        
        if ground_has_negative and response_has_positive:
            if "not" not in response.lower() and "no " not in response.lower():
                hallucination_indicators.append(
                    "Response provides info for non-existent item"
                )
        
        # Use LLM for deeper hallucination detection if available
        if self.llm_generator and not hallucination_indicators:
            try:
                hallucination_check = self._llm_hallucination_check(
                    question, response, ground_truth
                )
                if hallucination_check:
                    hallucination_indicators.append(hallucination_check)
            except Exception as e:
                logger.warning(f"LLM hallucination check failed: {e}")
        
        detected = len(hallucination_indicators) > 0
        details = "; ".join(hallucination_indicators) if detected else ""
        
        return detected, details
    
    def _llm_hallucination_check(
        self,
        question: str,
        response: str,
        ground_truth: str
    ) -> Optional[str]:
        """Use LLM to detect subtle hallucinations."""
        check_prompt = f"""Analyze if this response contains hallucinated (made-up) information.

QUESTION: {question}

GROUND TRUTH: {ground_truth}

RESPONSE TO CHECK: {response}

Does the response contain any information that:
1. Is not supported by the ground truth?
2. Contains fabricated details, numbers, or facts?
3. Makes claims that contradict the ground truth?

If hallucination detected, respond with a brief description of what was hallucinated.
If no hallucination, respond with exactly: NO_HALLUCINATION"""

        result = self.llm_generator.generate(check_prompt)
        
        if "NO_HALLUCINATION" in result.upper():
            return None
        else:
            return result[:200]  # Truncate long responses


def evaluate_response(
    llm_generator,
    question_id: str,
    question: str,
    response: str,
    ground_truth: str,
    expected_entities: List[str],
    expected_keywords: List[str],
    retrieval_mode: str,
    sources: List[Dict],
    response_time_ms: float,
    **kwargs
) -> EvaluationResult:
    """
    Convenience function to evaluate a single response.
    """
    evaluator = Evaluator(llm_generator)
    return evaluator.evaluate_response(
        question_id=question_id,
        question=question,
        response=response,
        ground_truth=ground_truth,
        expected_entities=expected_entities,
        expected_keywords=expected_keywords,
        retrieval_mode=retrieval_mode,
        sources=sources,
        response_time_ms=response_time_ms,
        **kwargs
    )


def aggregate_results(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Aggregate multiple evaluation results into summary statistics.
    
    Args:
        results: List of EvaluationResult objects
        
    Returns:
        Dictionary with aggregate statistics
    """
    if not results:
        return {}
    
    # Group by retrieval mode
    by_mode = {}
    for r in results:
        mode = r.retrieval_mode
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(r)
    
    # Calculate aggregates per mode
    aggregates = {}
    for mode, mode_results in by_mode.items():
        n = len(mode_results)
        aggregates[mode] = {
            "count": n,
            "avg_relevance_score": sum(r.relevance_score for r in mode_results) / n,
            "avg_accuracy_score": sum(r.accuracy_score for r in mode_results) / n,
            "avg_keyword_coverage": sum(r.keyword_coverage for r in mode_results) / n,
            "avg_entity_coverage": sum(r.entity_coverage for r in mode_results) / n,
            "avg_response_time_ms": sum(r.response_time_ms for r in mode_results) / n,
            "avg_source_count": sum(r.source_count for r in mode_results) / n,
            "hallucination_rate": sum(1 for r in mode_results if r.hallucination_detected) / n,
            "avg_graph_entities": sum(r.graph_entities_found for r in mode_results) / n,
        }
    
    return {
        "total_evaluations": len(results),
        "by_mode": aggregates,
        "modes_compared": list(by_mode.keys())
    }
