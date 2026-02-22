"""
Evaluation Module for Research Benchmarking.

Provides ablation study capabilities comparing:
- Vector-only RAG
- Graph-only retrieval
- Hybrid GraphRAG
"""

from .benchmark import BENCHMARK_QUESTIONS, get_benchmark_questions, get_statistics
from .evaluator import Evaluator, evaluate_response
from .ablation import AblationStudy, AblationStudyConfig, run_ablation_study

__all__ = [
    "BENCHMARK_QUESTIONS",
    "get_benchmark_questions",
    "get_statistics",
    "Evaluator",
    "evaluate_response",
    "AblationStudy",
    "AblationStudyConfig",
    "run_ablation_study",
]
