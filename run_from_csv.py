#!/usr/bin/env python3
"""
Run Holodeck scene generation for all prompts in a CSV file.

Usage:
    xvfb-run -a uv run python run_from_csv.py
"""

import argparse
import csv
import os
import traceback
from pathlib import Path

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR
from ai2holodeck.generation.holodeck import Holodeck

# Default paths
CSV_FILE = "/home/ubuntu/hsm/prompts.csv"
RESULTS_DIR = "./data/scenes_batch"


def main():
    parser = argparse.ArgumentParser(
        description="Run Holodeck scene generation from CSV prompts"
    )
    parser.add_argument("--csv_file", type=str, default=CSV_FILE)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument(
        "--start_id", type=int, default=None, help="Start from this ID (inclusive)"
    )
    parser.add_argument(
        "--end_id", type=int, default=None, help="End at this ID (inclusive)"
    )
    parser.add_argument(
        "--single_room",
        action="store_true",
        default=True,
        help="Generate single room scenes (default: True)",
    )
    parser.add_argument(
        "--generate_image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate top-down PNG image (default: True)",
    )
    parser.add_argument(
        "--generate_video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate walkthrough video (default: False)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"CSV file: {args.csv_file}")
    print(f"Results dir: {results_dir}")
    print(f"Single room: {args.single_room}")
    print(f"Generate image: {args.generate_image}")
    print(f"Generate video: {args.generate_video}")

    # Initialize Holodeck model once
    print("\nInitializing Holodeck model...")
    model = Holodeck(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_org=os.environ.get("OPENAI_ORG"),
        objaverse_asset_dir=OBJATHOR_ASSETS_DIR,
        single_room=args.single_room,
    )

    # Read prompts from CSV
    with open(args.csv_file, "r") as f:
        prompts = list(csv.DictReader(f))

    total = len(prompts)
    print(f"Loaded {total} prompts from CSV")

    for i, row in enumerate(prompts):
        prompt_id = int(row["ID"])

        # Filter by ID range
        if args.start_id is not None and prompt_id < args.start_id:
            continue
        if args.end_id is not None and prompt_id > args.end_id:
            continue

        description = row["Description"]
        save_dir = results_dir / f"scene_{prompt_id:03d}"

        print(f"\n{'='*60}")
        print(f"Scene {prompt_id} ({i+1}/{total}): {description[:50]}...")
        print(f"{'='*60}\n")

        try:
            scene = model.get_empty_scene()
            _, scene_dir = model.generate_scene(
                scene=scene,
                query=description,
                save_dir=str(save_dir),
                generate_image=args.generate_image,
                generate_video=args.generate_video,
                add_ceiling=False,
                add_time=False,  # Use numbered dirs instead
                use_constraint=True,
                use_milp=False,
                random_selection=False,
            )
            print(f"Scene {prompt_id} saved to: {scene_dir}")
        except Exception as e:
            print(f"Error generating scene {prompt_id}: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Batch complete! Results in: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
