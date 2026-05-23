"""Top-level dispatcher for the PET reconstruction MVP.

Examples:
    # 1) Preprocess every volume (one-time, ~hours)
    python -m src.pet_reconstruction.main preprocess

    # 1b) Smoke variant: 50 patients, 128x128 resolution
    python -m src.pet_reconstruction.main preprocess --smoke --limit 50

    # 2) Train Pipeline A
    python -m src.pet_reconstruction.main train --pipeline supervised

    # 2b) Train Pipeline B
    python -m src.pet_reconstruction.main train --pipeline unconditional

    # 3) Reconstruct test volumes with a trained checkpoint
    python -m src.pet_reconstruction.main reconstruct --pipeline supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-029

    # 4) Evaluate a checkpoint on the held-out test split (metrics + figures)
    python -m src.pet_reconstruction.main evaluate --pipeline supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-029 \
        --output-dir evaluations/supervised/

The flags --smoke for preprocess and train shrink the run to a ~1-hour
end-to-end validation of the pipeline.
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="PET reconstruction MVP dispatcher.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_pre = subparsers.add_parser("preprocess", help="Build the cached slice dataset.")
    p_pre.add_argument("--limit", type=int, default=None)
    p_pre.add_argument("--smoke", action="store_true")

    p_train = subparsers.add_parser("train", help="Train one of the two pipelines.")
    p_train.add_argument("--pipeline", choices=["supervised", "unconditional"], required=True)
    p_train.add_argument("--smoke", action="store_true")

    p_recon = subparsers.add_parser(
        "reconstruct", help="Reconstruct test volumes from a checkpoint."
    )
    p_recon.add_argument("--pipeline", choices=["supervised", "unconditional"], required=True)
    # The remaining flags are passed through to the per-pipeline reconstruct script.
    p_recon.add_argument("recon_args", nargs=argparse.REMAINDER)

    p_eval = subparsers.add_parser(
        "evaluate", help="Run inference + metrics on a held-out split."
    )
    p_eval.add_argument("--pipeline", choices=["supervised", "unconditional"], required=True)
    p_eval.add_argument("eval_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.command == "preprocess":
        from .config import DataConfig
        from .preprocess import preprocess_all

        cfg = DataConfig()
        if args.smoke:
            cfg.image_size = 128
        preprocess_all(cfg, limit=args.limit)
        return

    if args.command == "train":
        if args.pipeline == "supervised":
            from .config import SupervisedConfig, apply_smoke_overrides
            from .train_supervised import train as train_fn

            cfg = SupervisedConfig()
        else:
            from .config import UnconditionalConfig, apply_smoke_overrides
            from .train_unconditional import train as train_fn

            cfg = UnconditionalConfig()
        if args.smoke:
            apply_smoke_overrides(cfg)
        train_fn(cfg)
        return

    if args.command == "reconstruct":
        # Re-route to the per-pipeline reconstruct main(). We rewrite sys.argv so the
        # nested argparser sees just its own flags.
        if args.pipeline == "supervised":
            from .reconstruct_supervised import main as recon_main
        else:
            from .reconstruct_unconditional import main as recon_main
        sys.argv = [sys.argv[0], *(args.recon_args or [])]
        recon_main()
        return

    if args.command == "evaluate":
        from .evaluate import main as eval_main

        # The pipeline flag is consumed here AND re-injected for the inner parser,
        # so the user can write the same flag set as if calling evaluate.py directly.
        passthrough = ["--pipeline", args.pipeline, *(args.eval_args or [])]
        sys.argv = [sys.argv[0], *passthrough]
        eval_main()
        return


if __name__ == "__main__":
    main()
