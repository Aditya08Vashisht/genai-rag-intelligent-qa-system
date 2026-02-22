"""
Evaluation API Routes - Endpoints for running ablation studies and benchmarks.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel

from ...evaluation import (
    BENCHMARK_QUESTIONS,
    get_benchmark_questions,
    get_statistics,
    AblationStudy,
    AblationStudyConfig,
)
from ...evaluation.benchmark import QuestionCategory, Difficulty

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["Evaluation"])

# Store for ongoing/completed evaluations
evaluation_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_question": "",
    "results": None
}


# ============================================================================
# Request/Response Models
# ============================================================================

class BenchmarkStatsResponse(BaseModel):
    """Statistics about the benchmark dataset."""
    total_questions: int
    by_category: dict
    by_difficulty: dict
    graph_required: int
    graph_optional: int


class QuestionResponse(BaseModel):
    """A single benchmark question."""
    id: str
    question: str
    category: str
    difficulty: str
    ground_truth: str
    expected_entities: List[str]
    expected_keywords: List[str]
    requires_graph: bool


class RunEvaluationRequest(BaseModel):
    """Request to run an evaluation."""
    modes: List[str] = ["vector_only", "graph_only", "hybrid"]
    category: Optional[str] = None
    difficulty: Optional[str] = None
    limit: Optional[int] = None


class EvaluationProgressResponse(BaseModel):
    """Progress of ongoing evaluation."""
    running: bool
    progress: int
    total: int
    current_question: str
    percentage: float


class SingleEvaluationRequest(BaseModel):
    """Request to evaluate a single question across modes."""
    question: str
    ground_truth: Optional[str] = None
    modes: List[str] = ["vector_only", "graph_only", "hybrid"]


# ============================================================================
# Dependencies
# ============================================================================

def get_rag_chain():
    """Get RAG chain instance."""
    from ..main import rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    return rag_chain


def get_llm_generator():
    """Get LLM generator instance."""
    from ..main import rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    return rag_chain.generator


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/benchmark/stats", response_model=BenchmarkStatsResponse)
async def get_benchmark_stats():
    """Get statistics about the benchmark dataset."""
    stats = get_statistics()
    return BenchmarkStatsResponse(**stats)


@router.get("/benchmark/questions")
async def list_benchmark_questions(
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    requires_graph: Optional[bool] = None,
    limit: Optional[int] = None
):
    """List benchmark questions with optional filters."""
    # Parse category enum
    cat_enum = None
    if category:
        try:
            cat_enum = QuestionCategory(category)
        except ValueError:
            raise HTTPException(400, f"Invalid category: {category}")
    
    # Parse difficulty enum
    diff_enum = None
    if difficulty:
        try:
            diff_enum = Difficulty(difficulty)
        except ValueError:
            raise HTTPException(400, f"Invalid difficulty: {difficulty}")
    
    questions = get_benchmark_questions(
        category=cat_enum,
        difficulty=diff_enum,
        requires_graph=requires_graph,
        limit=limit
    )
    
    return {
        "count": len(questions),
        "questions": [q.to_dict() for q in questions]
    }


@router.post("/run")
async def run_evaluation(
    request: RunEvaluationRequest,
    background_tasks: BackgroundTasks,
    rag=Depends(get_rag_chain),
    generator=Depends(get_llm_generator)
):
    """
    Start a full ablation study evaluation.
    This runs in the background and returns immediately.
    """
    global evaluation_state
    
    if evaluation_state["running"]:
        raise HTTPException(400, "Evaluation already in progress")
    
    # Filter questions
    cat_enum = None
    if request.category:
        try:
            cat_enum = QuestionCategory(request.category)
        except ValueError:
            raise HTTPException(400, f"Invalid category: {request.category}")
    
    diff_enum = None
    if request.difficulty:
        try:
            diff_enum = Difficulty(request.difficulty)
        except ValueError:
            raise HTTPException(400, f"Invalid difficulty: {request.difficulty}")
    
    questions = get_benchmark_questions(
        category=cat_enum,
        difficulty=diff_enum,
        limit=request.limit
    )
    
    if not questions:
        raise HTTPException(400, "No questions match the criteria")
    
    # Define progress callback
    def progress_callback(current, total, message):
        global evaluation_state
        evaluation_state["progress"] = current
        evaluation_state["total"] = total
        evaluation_state["current_question"] = message
    
    # Define background task
    async def run_study():
        global evaluation_state
        evaluation_state["running"] = True
        evaluation_state["progress"] = 0
        evaluation_state["total"] = len(questions) * len(request.modes)
        
        try:
            study = AblationStudy(rag, generator)
            config = AblationStudyConfig(
                modes=request.modes,
                questions=questions,
                output_dir="./data/evaluation_results"
            )
            results = study.run(config, progress_callback=progress_callback)
            evaluation_state["results"] = results.to_dict()
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            evaluation_state["results"] = {"error": str(e)}
        finally:
            evaluation_state["running"] = False
    
    # Start background task
    background_tasks.add_task(run_study)
    
    return {
        "status": "started",
        "questions": len(questions),
        "modes": request.modes,
        "total_evaluations": len(questions) * len(request.modes)
    }


@router.get("/progress", response_model=EvaluationProgressResponse)
async def get_evaluation_progress():
    """Get the progress of an ongoing evaluation."""
    total = evaluation_state["total"] or 1
    return EvaluationProgressResponse(
        running=evaluation_state["running"],
        progress=evaluation_state["progress"],
        total=total,
        current_question=evaluation_state["current_question"],
        percentage=(evaluation_state["progress"] / total) * 100
    )


@router.get("/results")
async def get_evaluation_results():
    """Get the results of the last completed evaluation."""
    if evaluation_state["running"]:
        return {
            "status": "running",
            "progress": evaluation_state["progress"],
            "total": evaluation_state["total"]
        }
    
    if evaluation_state["results"] is None:
        return {
            "status": "no_results",
            "message": "No evaluation has been run yet"
        }
    
    return {
        "status": "complete",
        "results": evaluation_state["results"]
    }


@router.get("/results/summary")
async def get_results_summary():
    """Get a summary of the evaluation results for display."""
    if evaluation_state["results"] is None:
        raise HTTPException(404, "No evaluation results available")
    
    results = evaluation_state["results"]
    
    # Extract key metrics
    aggregated = results.get("aggregated", {})
    by_mode = aggregated.get("by_mode", {})
    
    summary = {
        "total_evaluations": aggregated.get("total_evaluations", 0),
        "modes": list(by_mode.keys()),
        "comparison": {}
    }
    
    metrics = [
        ("relevance", "avg_relevance_score", True),
        ("accuracy", "avg_accuracy_score", True),
        ("response_time_ms", "avg_response_time_ms", False),
        ("hallucination_rate", "hallucination_rate", False),
        ("source_count", "avg_source_count", True),
    ]
    
    for metric_name, metric_key, higher_is_better in metrics:
        metric_data = {}
        best_mode = None
        best_value = None
        
        for mode in by_mode:
            value = by_mode[mode].get(metric_key, 0)
            metric_data[mode] = round(value, 3)
            
            if best_value is None:
                best_value = value
                best_mode = mode
            elif higher_is_better and value > best_value:
                best_value = value
                best_mode = mode
            elif not higher_is_better and value < best_value:
                best_value = value
                best_mode = mode
        
        summary["comparison"][metric_name] = {
            "values": metric_data,
            "best_mode": best_mode
        }
    
    return summary


@router.post("/single")
async def evaluate_single_question(
    request: SingleEvaluationRequest,
    rag=Depends(get_rag_chain),
    generator=Depends(get_llm_generator)
):
    """
    Evaluate a single question across all modes.
    Useful for quick testing and demonstration.
    """
    import time
    from ...evaluation.evaluator import Evaluator
    
    evaluator = Evaluator(generator)
    results = {}
    
    for mode in request.modes:
        start = time.time()
        response = rag.query(request.question, retrieval_mode=mode)
        elapsed = (time.time() - start) * 1000
        
        # Basic evaluation if ground truth provided
        if request.ground_truth:
            eval_result = evaluator.evaluate_response(
                question_id="custom",
                question=request.question,
                response=response.get("answer", ""),
                ground_truth=request.ground_truth,
                expected_entities=[],
                expected_keywords=[],
                retrieval_mode=mode,
                sources=response.get("sources", []),
                response_time_ms=elapsed,
                graph_entities_found=response.get("graph_entities_found", 0)
            )
            
            results[mode] = {
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
                "response_time_ms": round(elapsed, 1),
                "relevance_score": eval_result.relevance_score,
                "accuracy_score": round(eval_result.accuracy_score, 2),
                "hallucination_detected": eval_result.hallucination_detected
            }
        else:
            results[mode] = {
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
                "response_time_ms": round(elapsed, 1),
                "graph_entities_found": response.get("graph_entities_found", 0)
            }
    
    return {
        "question": request.question,
        "ground_truth": request.ground_truth,
        "results": results
    }


@router.get("/report")
async def get_markdown_report():
    """Generate a markdown report of the evaluation results."""
    if evaluation_state["results"] is None:
        raise HTTPException(404, "No evaluation results available")
    
    from ...evaluation.ablation import AblationStudyResults
    
    # Reconstruct results object
    results_dict = evaluation_state["results"]
    
    # Generate markdown
    report = []
    report.append("# Ablation Study Results: RAG vs GraphRAG\n\n")
    
    aggregated = results_dict.get("aggregated", {})
    by_mode = aggregated.get("by_mode", {})
    
    if not by_mode:
        return {"report": "# No results available\n\nRun an evaluation first."}
    
    # Summary table
    report.append("## Summary Statistics\n\n")
    report.append("| Metric | Vector-Only | Graph-Only | Hybrid (Ours) |\n")
    report.append("|--------|-------------|------------|---------------|\n")
    
    modes = ["vector_only", "graph_only", "hybrid"]
    metrics = [
        ("Avg. Relevance Score (1-5)", "avg_relevance_score", "{:.2f}"),
        ("Avg. Accuracy Score", "avg_accuracy_score", "{:.1%}"),
        ("Avg. Response Time (ms)", "avg_response_time_ms", "{:.0f}"),
        ("Hallucination Rate", "hallucination_rate", "{:.1%}"),
        ("Avg. Source Count", "avg_source_count", "{:.1f}"),
    ]
    
    for metric_name, metric_key, fmt in metrics:
        row = f"| {metric_name} |"
        for mode in modes:
            if mode in by_mode:
                value = by_mode[mode].get(metric_key, 0)
                row += f" {fmt.format(value)} |"
            else:
                row += " N/A |"
        report.append(row + "\n")
    
    report.append("\n")
    
    # Hallucination examples
    examples = results_dict.get("hallucination_examples", [])
    if examples:
        report.append("## Hallucination Reduction Examples\n\n")
        for i, ex in enumerate(examples[:3], 1):
            report.append(f"### Example {i}\n")
            report.append(f"**Question:** {ex['question']}\n\n")
            for mode_resp in ex.get('mode_responses', []):
                status = "❌ Hallucinated" if mode_resp['hallucinated'] else "✅ Accurate"
                report.append(f"- **{mode_resp['mode']}:** {status}\n")
            report.append("\n")
    
    return {"report": "".join(report)}
