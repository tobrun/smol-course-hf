#!/usr/bin/env python3
"""
LightEval Model Comparison Script

This script compares the performance of two small language models (Qwen2.5-0.5B and
SmolLM2-360M-Instruct) on medical domain tasks using the LightEval framework.
"""

import os
import logging
from typing import Dict, List, Tuple, Any
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class to handle the evaluation of language models using LightEval."""
    
    def __init__(
        self, 
        output_dir: str = "~/tmp",
        cache_dir: str = "~/tmp",
        token: str = None,
        max_samples: int = 10,
        batch_size: int = 1,
        job_id: int = 1,
        num_fewshot: int = 5,
        save_details: bool = False,
    ):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
            cache_dir: Directory to cache models and data
            token: HuggingFace token for accessing models
            max_samples: Maximum number of samples to evaluate
            batch_size: Batch size for evaluation
            job_id: Job identifier
            num_fewshot: Number of few-shot examples to use
            save_details: Whether to save detailed evaluation results
        """
        self.output_dir = Path(output_dir).expanduser()
        self.cache_dir = Path(cache_dir).expanduser()
        self.token = token
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.job_id = job_id
        self.num_fewshot = num_fewshot
        self.save_details = save_details
        
        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration
        self._init_config()
        
    def _init_config(self) -> None:
        """Initialize LightEval configuration objects."""
        self.env_config = EnvConfig(
            token=self.token, 
            cache_dir=str(self.cache_dir)
        )
        
        self.evaluation_tracker = EvaluationTracker(
            output_dir=str(self.output_dir),
            save_details=self.save_details,
            push_to_hub=False,
            push_to_tensorboard=False,
            public=False,
            hub_results_org=False,
        )
        
        self.pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.ACCELERATE,
            env_config=self.env_config,
            job_id=self.job_id,
            override_batch_size=self.batch_size,
            num_fewshot_seeds=0,
            max_samples=self.max_samples,
            use_chat_template=False,
        )
    
    def create_domain_tasks(self, tasks: List[Tuple[str, str, int, bool]]) -> str:
        """
        Create LightEval domain tasks string from a list of task specifications.
        
        Args:
            tasks: List of (suite, task, num_fewshot, limit_fewshots) tuples
            
        Returns:
            Comma-separated string of task specifications
        """
        task_specs = []
        for suite, task, num_fewshot, limit_fewshots in tasks:
            limit_fewshots_int = 1 if limit_fewshots else 0
            task_specs.append(f"{suite}|{task}|{num_fewshot}|{limit_fewshots_int}")
        return ",".join(task_specs)
    
    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a model on the specified tasks.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Loading model: {model_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        logger.info(f"Evaluating model: {model_name}")
        pipeline = Pipeline(
            tasks=self.domain_tasks,
            pipeline_parameters=self.pipeline_params,
            evaluation_tracker=self.evaluation_tracker,
            model=model
        )
        
        try:
            pipeline.evaluate()
            results = pipeline.get_results()
            logger.info(f"Evaluation completed for {model_name}")
            return results
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
            raise
    
    def plot_results(
        self, 
        results: Dict[str, Dict[str, Any]], 
        output_path: str = "model_comparison.png"
    ) -> None:
        """
        Plot and save comparison results.
        
        Args:
            results: Dictionary mapping model names to their evaluation results
            output_path: Path to save the output plot
        """
        dataframes = []
        
        for model_name, result in results.items():
            model_df = pd.DataFrame.from_records(result["results"]).T["acc"].rename(model_name)
            dataframes.append(model_df)
        
        df = pd.concat(dataframes, axis=1)
        
        plt.figure(figsize=(10, 6))
        ax = df.plot(kind="barh")
        ax.set_xlabel("Accuracy")
        ax.set_title("Model Performance Comparison")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved as {output_path}")
        
        # Also save raw results as CSV
        csv_path = output_path.replace(".png", ".csv")
        df.to_csv(csv_path)
        logger.info(f"Raw results saved as {csv_path}")
        
        return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare language models on medical tasks")
    parser.add_argument("--output-dir", type=str, default="~/lighteval_results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--cache-dir", type=str, default="~/lighteval_cache", 
                        help="Directory to cache models and data")
    parser.add_argument("--max-samples", type=int, default=10, 
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="Batch size for evaluation")
    parser.add_argument("--num-fewshot", type=int, default=5, 
                        help="Number of few-shot examples")
    parser.add_argument("--save-details", action="store_true", 
                        help="Save detailed evaluation results")
    parser.add_argument("--output-plot", type=str, default="model_comparison.png", 
                        help="Path to save the output plot")
    return parser.parse_args()


def main():
    """Main function to run the evaluation."""
    args = parse_args()
    
    # Get token from environment variable
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN environment variable not set. Some models may not be accessible.")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        token=token,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        save_details=args.save_details,
    )
    
    # Define medical domain tasks
    medical_tasks = [
        ("leaderboard", "mmlu:anatomy", args.num_fewshot, False),
        ("leaderboard", "mmlu:professional_medicine", args.num_fewshot, False),
        ("leaderboard", "mmlu:high_school_biology", args.num_fewshot, False),
        ("leaderboard", "mmlu:high_school_chemistry", args.num_fewshot, False),
    ]
    
    evaluator.domain_tasks = evaluator.create_domain_tasks(medical_tasks)
    logger.info(f"Domain tasks: {evaluator.domain_tasks}")
    
    # Models to evaluate
    models = {
        "Qwen2-0.5B-DPO": "Qwen/Qwen2.5-0.5B",
        "SmolLM2-360M-Instruct": "HuggingFaceTB/SmolLM2-360M-Instruct",
    }
    
    # Evaluate models
    results = {}
    for model_name, model_id in models.items():
        try:
            results[model_name] = evaluator.evaluate_model(model_id)
        except Exception as e:
            logger.error(f"Skipping model {model_name} due to error: {e}")
    
    # Plot and save results
    if results:
        df = evaluator.plot_results(results, args.output_plot)
        print("\nEvaluation Results Summary:")
        print(df.to_string())
    else:
        logger.error("No results to plot. All evaluations failed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise