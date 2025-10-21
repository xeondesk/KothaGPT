#!/usr/bin/env python3
"""
KothaGPT Master Pipeline Orchestrator
====================================

End-to-end pipeline automation with progress tracking, error handling,
and automatic step execution for the complete KothaGPT training pipeline.

Usage:
    python scripts/master_pipeline.py [--resume] [--parallel] [--debug]
    python scripts/master_pipeline.py --help
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Import our progress tracker
from progress_tracker import get_progress_tracker, PipelineStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/master_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MasterPipeline:
    """Master orchestrator for the complete KothaGPT pipeline."""

    def __init__(self, resume: bool = False, parallel: bool = False, debug: bool = False):
        self.resume = resume
        self.parallel = parallel
        self.debug = debug
        self.tracker = get_progress_tracker()

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)

        # Define pipeline steps with their configurations
        self.pipeline_steps = self._define_pipeline_steps()

    def _define_pipeline_steps(self) -> Dict:
        """Define all pipeline steps and their configurations."""
        return {
            "data_ingestion": {
                "description": "Collect and ingest raw data",
                "script": "data_ingestion/run_pipeline.py",
                "dependencies": [],
                "optional": True,
                "check_function": self._check_data_ingestion
            },
            "data_preprocessing": {
                "description": "Clean and preprocess data for training",
                "script": "data_tools/prepare_data.py",
                "dependencies": [],
                "optional": False,
                "check_function": self._check_data_preprocessing
            },
            "tokenizer_training": {
                "description": "Train SentencePiece tokenizer",
                "script": "training/train_tokenizer.py",
                "dependencies": ["data_preprocessing"],
                "optional": False,
                "check_function": self._check_tokenizer_training
            },
            "lora_finetuning": {
                "description": "Apply LoRA fine-tuning to base model",
                "script": "training/scripts/train_lora.py",
                "dependencies": ["data_preprocessing", "tokenizer_training"],
                "optional": False,
                "check_function": self._check_lora_finetuning
            },
            "rlhf_preparation": {
                "description": "Prepare human feedback dataset for RLHF",
                "script": "training/rlhf/prepare_feedback_data.py",
                "dependencies": ["lora_finetuning"],
                "optional": True,
                "check_function": self._check_rlhf_preparation
            },
            "reward_model_training": {
                "description": "Train reward model for RLHF",
                "script": "training/rlhf/train_reward_model.py",
                "dependencies": ["rlhf_preparation"],
                "optional": True,
                "check_function": self._check_reward_model_training
            },
            "rlhf_finetuning": {
                "description": "Apply RLHF fine-tuning",
                "script": "training/scripts/train_rlhf.py",
                "dependencies": ["lora_finetuning", "reward_model_training"],
                "optional": True,
                "check_function": self._check_rlhf_finetuning
            },
            "model_evaluation": {
                "description": "Evaluate trained models",
                "script": "evaluation/evaluate_models.py",
                "dependencies": ["lora_finetuning", "rlhf_finetuning"],
                "optional": False,
                "check_function": self._check_model_evaluation
            },
            "deployment": {
                "description": "Deploy model and setup API",
                "script": "deploy/deploy.sh",
                "dependencies": ["model_evaluation"],
                "optional": False,
                "check_function": self._check_deployment
            }
        }

    def _check_data_ingestion(self) -> bool:
        """Check if data ingestion step is complete."""
        raw_data_dir = Path("data/raw")
        return raw_data_dir.exists() and any(raw_data_dir.iterdir())

    def _check_data_preprocessing(self) -> bool:
        """Check if data preprocessing step is complete."""
        processed_dir = Path("data/processed")
        final_dir = Path("data/final")
        return (processed_dir.exists() and any(processed_dir.iterdir()) and
                final_dir.exists() and (final_dir / "corpus.txt").exists())

    def _check_tokenizer_training(self) -> bool:
        """Check if tokenizer training step is complete."""
        tokenizer_dir = Path("models/tokenizer")
        return (tokenizer_dir.exists() and
                (tokenizer_dir / "kothagpt_tokenizer.model").exists() and
                (tokenizer_dir / "kothagpt_tokenizer.vocab").exists())

    def _check_lora_finetuning(self) -> bool:
        """Check if LoRA fine-tuning step is complete."""
        lora_dir = Path("models/banglagpt_lora")
        return lora_dir.exists() and any(lora_dir.iterdir())

    def _check_rlhf_preparation(self) -> bool:
        """Check if RLHF preparation step is complete."""
        rlhf_dir = Path("data/processed")
        return (rlhf_dir / "rlhf_feedback.jsonl").exists()

    def _check_reward_model_training(self) -> bool:
        """Check if reward model training step is complete."""
        reward_dir = Path("models/reward_model")
        return reward_dir.exists() and any(reward_dir.iterdir())

    def _check_rlhf_finetuning(self) -> bool:
        """Check if RLHF fine-tuning step is complete."""
        rlhf_dir = Path("models/banglagpt_rlhf")
        return rlhf_dir.exists() and any(rlhf_dir.iterdir())

    def _check_model_evaluation(self) -> bool:
        """Check if model evaluation step is complete."""
        eval_dir = Path("evaluation/results")
        return eval_dir.exists() and any(eval_dir.iterdir())

    def _check_deployment(self) -> bool:
        """Check if deployment step is complete."""
        deploy_dir = Path("deploy")
        api_file = Path("kotha_api.py")
        return (deploy_dir.exists() or api_file.exists())

    def _run_script(self, step_name: str, script_path: str) -> bool:
        """Run a pipeline script and return success status."""
        try:
            logger.info(f"🚀 Executing step: {step_name}")
            logger.info(f"📜 Script: {script_path}")

            # Start the step in progress tracker
            self.tracker.start_step(step_name)

            # Run the script
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info(f"✅ Step '{step_name}' completed successfully")
                self.tracker.complete_step(step_name)
                return True
            else:
                error_msg = f"Script failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr[-500:]}"  # Last 500 chars
                logger.error(f"❌ Step '{step_name}' failed: {error_msg}")
                self.tracker.fail_step(step_name, error_msg)
                return False

        except subprocess.TimeoutExpired:
            error_msg = f"Step '{step_name}' timed out after 1 hour"
            logger.error(f"⏰ {error_msg}")
            self.tracker.fail_step(step_name, error_msg)
            return False
        except Exception as e:
            error_msg = f"Unexpected error running step '{step_name}': {str(e)}"
            logger.error(f"💥 {error_msg}")
            self.tracker.fail_step(step_name, error_msg)
            return False

    def _check_step_completion(self, step_name: str, step_config: Dict) -> bool:
        """Check if a step is already completed."""
        # First check progress tracker
        step = self.tracker.steps.get(step_name)
        if step and step.status == PipelineStatus.COMPLETED:
            return True

        # Then check using step-specific check function
        if step_config["check_function"]:
            return step_config["check_function"]()

        return False

    def run_pipeline(self) -> bool:
        """Run the complete KothaGPT pipeline."""
        logger.info("🎯 Starting KothaGPT Master Pipeline")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Execute steps in dependency order
            completed_steps = set()

            while len(completed_steps) < len(self.pipeline_steps):
                # Find runnable steps
                runnable_steps = []

                for step_name, step_config in self.pipeline_steps.items():
                    if (step_name not in completed_steps and
                        all(dep in completed_steps for dep in step_config["dependencies"]) and
                        not self._check_step_completion(step_name, step_config)):
                        runnable_steps.append(step_name)

                if not runnable_steps:
                    # No more steps to run
                    break

                logger.info(f"📋 Next runnable steps: {', '.join(runnable_steps)}")

                # Execute runnable steps (in parallel if enabled)
                if self.parallel and len(runnable_steps) > 1:
                    logger.info("⚡ Running steps in parallel...")
                    # For now, run sequentially even in parallel mode
                    # TODO: Implement actual parallel execution

                for step_name in runnable_steps:
                    step_config = self.pipeline_steps[step_name]

                    if not self._check_step_completion(step_name, step_config):
                        success = self._run_script(step_name, step_config["script"])

                        if success:
                            completed_steps.add(step_name)
                        elif not step_config["optional"]:
                            logger.error(f"❌ Required step '{step_name}' failed. Stopping pipeline.")
                            return False
                    else:
                        logger.info(f"⏭️ Step '{step_name}' already completed, skipping")
                        completed_steps.add(step_name)

                # Small delay between batches
                time.sleep(1)

            # Final progress report
            self.tracker.print_progress_report()

            # Calculate total time
            total_time = time.time() - start_time
            logger.info(f"🎉 Pipeline completed in {total_time:.2f} seconds!")

            # Export final report
            report_file = self.tracker.export_report()
            logger.info(f"📋 Detailed report saved to: {report_file}")

            return True

        except KeyboardInterrupt:
            logger.info("⏹️ Pipeline interrupted by user")
            self.tracker.print_progress_report()
            return False
        except Exception as e:
            logger.error(f"💥 Unexpected error in pipeline: {e}")
            return False

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="KothaGPT Master Pipeline Orchestrator")
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel execution where possible')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--status', action='store_true',
                       help='Show current pipeline status and exit')
    parser.add_argument('--reset', action='store_true',
                       help='Reset pipeline progress and exit')

    args = parser.parse_args()

    # Handle special commands
    if args.status:
        tracker = get_progress_tracker()
        tracker.print_progress_report()
        return

    if args.reset:
        from scripts.progress_tracker import reset_progress
        reset_progress()
        logger.info("🔄 Pipeline progress reset")
        return

    # Create and run pipeline
    pipeline = MasterPipeline(
        resume=args.resume,
        parallel=args.parallel,
        debug=args.debug
    )

    success = pipeline.run_pipeline()

    if success:
        logger.info("🎉 KothaGPT pipeline completed successfully!")
        logger.info("🚀 Your model is ready for deployment!")
    else:
        logger.error("❌ Pipeline failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
