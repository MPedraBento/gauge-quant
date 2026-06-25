import argparse
import sys
import logging

# Configure basic logging to look professional and track cloud execution times
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("GaugeQuant")


def main():
    parser = argparse.ArgumentParser(
        description="GaugeQuant: Differentiable Gauge Co-Adaptation for LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Execution Mode")

    # ==========================================
    #             TRAIN ROUTER
    # ==========================================
    train_parser = subparsers.add_parser("train", help="Run the continuous gauge pre-training/fine-tuning loop.")
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., configs/llama_7b_lora.yaml)"
    )

    # ==========================================
    #             EVAL ROUTER
    # ==========================================
    eval_parser = subparsers.add_parser("eval", help="Evaluate Quantization Perplexity and Reasoning.")
    eval_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the architecture YAML config (to rebuild the exact model structure)"
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved artifact directory (e.g., checkpoints/run_005_llama_unified/)"
    )
    eval_parser.add_argument(
        "--tasks",
        type=str,
        default="wikitext,piqa,arc_easy",
        help="Comma-separated list of evaluation tasks (passed to EleutherAI LM-Eval)"
    )
    eval_parser.add_argument(
        "--w_bits",
        type=int,
        default=4,
        help="Target weight quantization bit-width"
    )
    eval_parser.add_argument(
        "--a_bits",
        type=int,
        default=4,
        help="Target activation quantization bit-width"
    )

    args = parser.parse_args()

    # ==========================================
    #             EXECUTION LOGIC
    # ==========================================
    try:
        if args.command == "train":
            logger.info(f"Initializing Training Pipeline via config: {args.config}")
            # Lazy import ensures we don't load massive PyTorch graphs unnecessarily
            from scripts import run_train
            run_train.main(args)

        elif args.command == "eval":
            logger.info(f"Initializing Evaluation Pipeline on checkpoint: {args.checkpoint}")
            logger.info(f"Tasks: {args.tasks} | Regime: W{args.w_bits}A{args.a_bits}")
            from scripts import run_eval
            run_eval.main(args)

    except ImportError as e:
        logger.error(f"Failed to load execution scripts. Are you in the virtual environment? Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Execution violently interrupted by user. Exiting gracefully.")
        sys.exit(0)


if __name__ == "__main__":
    main()