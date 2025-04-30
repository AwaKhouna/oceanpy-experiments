#!/usr/bin/env python
import argparse
import logging
import sys


def main():
    """Run the experiment with the specified parameters."""
    parser = argparse.ArgumentParser(description="Run oceanpy experiments")
    parser.add_argument(
        "--model",
        choices=["cp", "mip", "both"],
        default="both",
        help="Model type to evaluate",
    )
    parser.add_argument("--dataset", required=True, help="Dataset to use")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.info(
        f"Running experiment with model={args.model} on dataset={args.dataset}"
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
