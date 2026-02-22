"""
Ablation Study Runner - Compare Vector-only, Graph-only, and Hybrid RAG.

Runs benchmark questions through each retrieval mode and collects metrics.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .benchmark import (
    BENCHMARK_QUESTIONS, 
    BenchmarkQuestion, 
    get_benchmark_questions,
    get_statistics
)
from .evaluator import Evaluator, EvaluationResult, aggregate_results

logger = logging.getLogger(__name__)


@dataclass
class AblationStudyConfig:
    """Configuration for ablation study."""
    modes: List[str] = field(default_factory=lambda: ["vector_only", "graph_only", "hybrid"])
    questions: List[BenchmarkQuestion] = field(default_factory=lambda: BENCHMARK_QUESTIONS)
    output_dir: str = "./data/evaluation_results"
    save_intermediate: bool = True
    
    
@dataclass
class AblationStudyResults:
    """Complete results of an ablation study."""
    config: Dict[str, Any]
    start_time: str
    end_time: str
    duration_seconds: float
    
    # Results
    all_results: List[Dict]  # All individual EvaluationResults
    aggregated: Dict[str, Any]  # Aggregated statistics by mode
    
    # Comparison tables
    comparison_table: List[Dict]  # Per-question comparison
    summary_table: Dict[str, Any]  # Summary statistics
    
    # Hallucination examples
    hallucination_examples: List[Dict]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "all_results": self.all_results,
            "aggregated": self.aggregated,
            "comparison_table": self.comparison_table,
            "summary_table": self.summary_table,
            "hallucination_examples": self.hallucination_examples
        }
    
    def to_json(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def to_markdown_report(self) -> str:
        """Generate a markdown report for publication."""
        report = []
        
        report.append("# Ablation Study Results: RAG vs GraphRAG\n")
        report.append(f"**Generated:** {self.end_time}\n")
        report.append(f"**Duration:** {self.duration_seconds:.1f} seconds\n")
        report.append(f"**Total Questions:** {len(self.comparison_table)}\n\n")
        
        # Summary Table
        report.append("## Summary Statistics\n")
        report.append("| Metric | Vector-Only | Graph-Only | Hybrid (Ours) |\n")
        report.append("|--------|-------------|------------|---------------|\n")
        
        modes = ["vector_only", "graph_only", "hybrid"]
        agg = self.aggregated.get("by_mode", {})
        
        metrics = [
            ("Avg. Relevance Score (1-5)", "avg_relevance_score", "{:.2f}"),
            ("Avg. Accuracy Score (0-1)", "avg_accuracy_score", "{:.2%}"),
            ("Avg. Keyword Coverage", "avg_keyword_coverage", "{:.2%}"),
            ("Avg. Entity Coverage", "avg_entity_coverage", "{:.2%}"),
            ("Avg. Response Time (ms)", "avg_response_time_ms", "{:.0f}"),
            ("Avg. Source Count", "avg_source_count", "{:.1f}"),
            ("Hallucination Rate", "hallucination_rate", "{:.1%}"),
        ]
        
        for metric_name, metric_key, fmt in metrics:
            row = f"| {metric_name} |"
            for mode in modes:
                if mode in agg:
                    value = agg[mode].get(metric_key, 0)
                    row += f" {fmt.format(value)} |"
                else:
                    row += " N/A |"
            report.append(row + "\n")
        
        report.append("\n")
        
        # Hallucination Examples
        if self.hallucination_examples:
            report.append("## Hallucination Examples\n")
            report.append("Comparing responses where hallucination was detected:\n\n")
            
            for i, example in enumerate(self.hallucination_examples[:5], 1):
                report.append(f"### Example {i}: {example['question'][:60]}...\n")
                report.append(f"**Ground Truth:** {example['ground_truth'][:100]}...\n\n")
                
                for mode_result in example.get('mode_responses', []):
                    mode = mode_result['mode']
                    response = mode_result['response'][:150]
                    hallucinated = "❌ Hallucinated" if mode_result['hallucinated'] else "✅ Accurate"
                    report.append(f"- **{mode}:** {response}... ({hallucinated})\n")
                
                report.append("\n")
        
        # Per-Category Performance
        report.append("## Performance by Question Category\n")
        
        # Group by category
        category_stats = {}
        for row in self.comparison_table:
            cat = row.get("category", "unknown")
            if cat not in category_stats:
                category_stats[cat] = {mode: [] for mode in modes}
            
            for mode in modes:
                if f"{mode}_accuracy" in row:
                    category_stats[cat][mode].append(row[f"{mode}_accuracy"])
        
        report.append("| Category | Vector-Only | Graph-Only | Hybrid | Best Mode |\n")
        report.append("|----------|-------------|------------|--------|----------|\n")
        
        for cat, stats in category_stats.items():
            row = f"| {cat} |"
            best_mode = ""
            best_score = 0
            
            for mode in modes:
                if stats[mode]:
                    avg = sum(stats[mode]) / len(stats[mode])
                    row += f" {avg:.2%} |"
                    if avg > best_score:
                        best_score = avg
                        best_mode = mode
                else:
                    row += " N/A |"
            
            row += f" {best_mode} |"
            report.append(row + "\n")
        
        return "".join(report)


class AblationStudy:
    """
    Run ablation study comparing different retrieval modes.
    """
    
    def __init__(self, rag_chain, llm_generator):
        """
        Initialize ablation study.
        
        Args:
            rag_chain: RAGChain instance with mode support
            llm_generator: LLMGenerator for evaluation
        """
        self.rag_chain = rag_chain
        self.evaluator = Evaluator(llm_generator)
        self.results: List[EvaluationResult] = []
    
    def run(
        self,
        config: AblationStudyConfig = None,
        progress_callback=None
    ) -> AblationStudyResults:
        """
        Run the complete ablation study.
        
        Args:
            config: Study configuration
            progress_callback: Optional callback(current, total, message) for progress
            
        Returns:
            AblationStudyResults with all metrics and comparisons
        """
        if config is None:
            config = AblationStudyConfig()
        
        start_time = datetime.now()
        self.results = []
        
        total_steps = len(config.questions) * len(config.modes)
        current_step = 0
        
        logger.info(f"Starting ablation study: {len(config.questions)} questions × {len(config.modes)} modes")
        
        # Run each question through each mode
        comparison_table = []
        
        for question in config.questions:
            question_comparison = {
                "question_id": question.id,
                "question": question.question,
                "category": question.category.value,
                "difficulty": question.difficulty.value,
                "ground_truth": question.ground_truth
            }
            
            for mode in config.modes:
                current_step += 1
                
                if progress_callback:
                    progress_callback(
                        current_step, 
                        total_steps, 
                        f"Evaluating {question.id} with {mode}"
                    )
                
                logger.info(f"[{current_step}/{total_steps}] {question.id} - {mode}")
                
                try:
                    # Run query with specific mode
                    query_start = time.time()
                    response = self.rag_chain.query(
                        question.question,
                        retrieval_mode=mode
                    )
                    query_time = (time.time() - query_start) * 1000  # ms
                    
                    # Evaluate
                    eval_result = self.evaluator.evaluate_response(
                        question_id=question.id,
                        question=question.question,
                        response=response.get("answer", ""),
                        ground_truth=question.ground_truth,
                        expected_entities=question.expected_entities,
                        expected_keywords=question.expected_keywords,
                        retrieval_mode=mode,
                        sources=response.get("sources", []),
                        response_time_ms=query_time,
                        context_length=len(str(response)),
                        graph_entities_found=response.get("graph_entities_found", 0)
                    )
                    
                    self.results.append(eval_result)
                    
                    # Add to comparison
                    question_comparison[f"{mode}_response"] = response.get("answer", "")[:200]
                    question_comparison[f"{mode}_relevance"] = eval_result.relevance_score
                    question_comparison[f"{mode}_accuracy"] = eval_result.accuracy_score
                    question_comparison[f"{mode}_hallucinated"] = eval_result.hallucination_detected
                    question_comparison[f"{mode}_time_ms"] = query_time
                    
                    # Add delay to respect API rate limits (15 RPM for free tier)
                    time.sleep(2.0)

                    
                except Exception as e:
                    logger.error(f"Error evaluating {question.id} with {mode}: {e}")
                    question_comparison[f"{mode}_error"] = str(e)
            
            comparison_table.append(question_comparison)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Aggregate results
        aggregated = aggregate_results(self.results)
        
        # Find hallucination examples
        hallucination_examples = self._find_hallucination_examples(comparison_table)
        
        # Build summary table
        summary_table = self._build_summary_table(aggregated)
        
        # Create results object
        study_results = AblationStudyResults(
            config={
                "modes": config.modes,
                "question_count": len(config.questions),
                "output_dir": config.output_dir
            },
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            all_results=[r.to_dict() for r in self.results],
            aggregated=aggregated,
            comparison_table=comparison_table,
            summary_table=summary_table,
            hallucination_examples=hallucination_examples
        )
        
        # Save results if configured
        if config.output_dir:
            self._save_results(study_results, config.output_dir)
        
        logger.info(f"Ablation study complete: {duration:.1f}s, {len(self.results)} evaluations")
        
        return study_results
    
    def _find_hallucination_examples(self, comparison_table: List[Dict]) -> List[Dict]:
        """Find examples where hallucination behavior differs between modes."""
        examples = []
        
        for row in comparison_table:
            # Check if any mode hallucinated
            modes_hallucinated = {
                mode: row.get(f"{mode}_hallucinated", False)
                for mode in ["vector_only", "graph_only", "hybrid"]
            }
            
            # Interesting: hybrid didn't hallucinate but others did
            if not modes_hallucinated.get("hybrid", True) and \
               (modes_hallucinated.get("vector_only") or modes_hallucinated.get("graph_only")):
                examples.append({
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "mode_responses": [
                        {
                            "mode": mode,
                            "response": row.get(f"{mode}_response", ""),
                            "hallucinated": modes_hallucinated.get(mode, False)
                        }
                        for mode in ["vector_only", "graph_only", "hybrid"]
                    ]
                })
        
        return examples[:10]  # Return top 10 examples
    
    def _build_summary_table(self, aggregated: Dict) -> Dict:
        """Build a summary comparison table."""
        if not aggregated.get("by_mode"):
            return {}
        
        modes = aggregated["modes_compared"]
        by_mode = aggregated["by_mode"]
        
        # Find winner for each metric
        metrics = [
            "avg_relevance_score",
            "avg_accuracy_score", 
            "avg_keyword_coverage",
            "avg_entity_coverage",
            "avg_response_time_ms",
            "hallucination_rate"
        ]
        
        winners = {}
        for metric in metrics:
            best_mode = None
            best_value = None
            higher_is_better = metric != "hallucination_rate" and metric != "avg_response_time_ms"
            
            for mode in modes:
                value = by_mode[mode].get(metric, 0)
                if best_value is None:
                    best_value = value
                    best_mode = mode
                elif higher_is_better and value > best_value:
                    best_value = value
                    best_mode = mode
                elif not higher_is_better and value < best_value:
                    best_value = value
                    best_mode = mode
            
            winners[metric] = {
                "winner": best_mode,
                "value": best_value
            }
        
        return {
            "winners": winners,
            "hybrid_improvements": self._calculate_improvements(by_mode)
        }
    
    def _calculate_improvements(self, by_mode: Dict) -> Dict:
        """Calculate improvement of hybrid over baselines."""
        if "hybrid" not in by_mode or "vector_only" not in by_mode:
            return {}
        
        hybrid = by_mode["hybrid"]
        vector = by_mode.get("vector_only", {})
        graph = by_mode.get("graph_only", {})
        
        improvements = {}
        
        # Accuracy improvement
        if vector.get("avg_accuracy_score"):
            improvements["accuracy_vs_vector"] = (
                (hybrid.get("avg_accuracy_score", 0) - vector["avg_accuracy_score"]) 
                / vector["avg_accuracy_score"] * 100
            )
        
        if graph.get("avg_accuracy_score"):
            improvements["accuracy_vs_graph"] = (
                (hybrid.get("avg_accuracy_score", 0) - graph["avg_accuracy_score"]) 
                / graph["avg_accuracy_score"] * 100
            )
        
        # Hallucination reduction
        if vector.get("hallucination_rate"):
            improvements["hallucination_reduction_vs_vector"] = (
                (vector["hallucination_rate"] - hybrid.get("hallucination_rate", 0)) 
                / vector["hallucination_rate"] * 100
            )
        
        return improvements
    
    def _save_results(self, results: AblationStudyResults, output_dir: str):
        """Save results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_file = output_path / f"ablation_study_{timestamp}.json"
        results.to_json(str(json_file))
        logger.info(f"Saved JSON results to {json_file}")
        
        # Save Markdown report
        md_file = output_path / f"ablation_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(results.to_markdown_report())
        logger.info(f"Saved Markdown report to {md_file}")


def run_ablation_study(
    rag_chain,
    llm_generator,
    questions: List[BenchmarkQuestion] = None,
    modes: List[str] = None,
    output_dir: str = "./data/evaluation_results"
) -> AblationStudyResults:
    """
    Convenience function to run an ablation study.
    
    Args:
        rag_chain: RAGChain with mode support
        llm_generator: LLMGenerator for evaluation
        questions: Optional subset of questions (defaults to all)
        modes: Optional list of modes (defaults to all three)
        output_dir: Where to save results
        
    Returns:
        AblationStudyResults
    """
    study = AblationStudy(rag_chain, llm_generator)
    
    config = AblationStudyConfig(
        modes=modes or ["vector_only", "graph_only", "hybrid"],
        questions=questions or BENCHMARK_QUESTIONS,
        output_dir=output_dir
    )
    
    return study.run(config)
