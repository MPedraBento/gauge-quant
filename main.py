import argparse
import sys
import logging

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
    #             TRAIN
    # ==========================================
    train_parser = subparsers.add_parser("train", help="Run the gauge co-adaptation training loop.")
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., configs/llama2_7b_unified.yaml)"
    )

    args = parser.parse_args()

    try:
        if args.command == "train":
            logger.info(f"Initializing Training Pipeline via config: {args.config}")
            from scripts import run_train
            run_train.main(args)

    except ImportError as e:
        logger.error(f"Failed to load execution scripts. Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
