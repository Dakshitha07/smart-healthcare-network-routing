"""
Main Training & Evaluation Script
===================================
Run this to: train MAPPO, train MADDPG, evaluate all algorithms, generate plots.

Usage examples:
  python train_and_evaluate.py               # default: 200k timesteps, 500 episodes
  python train_and_evaluate.py --quick       # smoke-test: 50k timesteps, 200 episodes
  python train_and_evaluate.py --full        # research: 2M timesteps, 3000 episodes
  python train_and_evaluate.py --eval-only   # skip training, just evaluate + plot
  python train_and_evaluate.py --plot-only   # only regenerate plots
  python train_and_evaluate.py --full --seed 123   # multi-seed run

FIXES
-----
1. Default run (no --quick / --full flag) now uses sensible defaults
   (200k timesteps, 500 MADDPG episodes) instead of falling through
   with no timestep assignment, which caused MAPPO to inherit whatever
   total_timesteps was in config — potentially 2M on a first test run.

2. args.skip_maddpg attribute name: argparse converts "--skip-maddpg"
   to args.skip_maddpg (hyphen → underscore) automatically — verified OK.

3. Explicit check that num_agents in config matches NUM_NODES expectation
   so a stale config never silently trains with the wrong agent count.

4. Graceful handling when --eval-only is used without trained checkpoints
   — already handled inside evaluate_all.py (prints warning, continues).
"""

import argparse
import os
import time

from config import MAPPO_CONFIG, MADDPG_CONFIG, NUM_NODES
from network_env import HealthcareSDNEnv
from mappo_agent import MAPPOAgent
from maddpg_agent import MADDPGAgent
from evaluate_all import run_full_evaluation
from plot_results import generate_all_plots

# Default timesteps/episodes used when neither --quick nor --full is passed
_DEFAULT_MAPPO_TIMESTEPS = 200_000
_DEFAULT_MADDPG_EPISODES = 500


def main():
    parser = argparse.ArgumentParser(
        description="Healthcare SDN Routing — Train & Evaluate"
    )
    parser.add_argument("--quick",       action="store_true",
                        help="Smoke-test: 50k timesteps, 200 MADDPG episodes")
    parser.add_argument("--full",        action="store_true",
                        help="Full research run: 2M timesteps, 3000 MADDPG episodes")
    parser.add_argument("--skip-maddpg", action="store_true",
                        help="Skip MADDPG training")
    parser.add_argument("--eval-only",   action="store_true",
                        help="Skip training — evaluate existing checkpoints only")
    parser.add_argument("--plot-only",   action="store_true",
                        help="Only regenerate plots from existing results")
    parser.add_argument("--timesteps",   type=int, default=None,
                        help="Override MAPPO total timesteps")
    parser.add_argument("--episodes",    type=int, default=None,
                        help="Override MADDPG total episodes")
    parser.add_argument("--seed",        type=int, default=None,
                        help="Random seed override (use with --full for multi-seed)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------
    seed_tag        = f"_seed{args.seed}" if args.seed is not None else ""
    results_dir     = f"results/seed_{args.seed}" if args.seed is not None else "results"
    checkpoints_dir = f"checkpoints{seed_tag}"
    images_dir      = "project_images"

    os.makedirs(results_dir,     exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(images_dir,      exist_ok=True)

    # ------------------------------------------------------------------
    # Plot-only shortcut
    # ------------------------------------------------------------------
    if args.plot_only:
        generate_all_plots(results_dir, images_dir)
        return

    # ------------------------------------------------------------------
    # Resolve timesteps / episodes
    # ------------------------------------------------------------------
    if args.quick:
        mappo_timesteps  = 50_000
        maddpg_episodes  = 200
        log_interval     = 10
        save_interval    = 100
    elif args.full:
        mappo_timesteps  = 2_000_000
        maddpg_episodes  = 3_000
        log_interval     = 50
        save_interval    = 500
    else:
        # FIX 1: sensible defaults so a bare run doesn't train for 2M steps
        mappo_timesteps  = _DEFAULT_MAPPO_TIMESTEPS
        maddpg_episodes  = _DEFAULT_MADDPG_EPISODES
        log_interval     = 20
        save_interval    = 200

    # Manual overrides always win
    if args.timesteps:
        mappo_timesteps = args.timesteps
    if args.episodes:
        maddpg_episodes = args.episodes

    # ------------------------------------------------------------------
    # STEP 1: Train MAPPO
    # ------------------------------------------------------------------
    if not args.eval_only:
        print("\n" + "=" * 70)
        print("STEP 1: TRAINING MAPPO (Your Improvement)")
        print("=" * 70)

        mappo_config = MAPPO_CONFIG.copy()
        mappo_config["total_timesteps"] = mappo_timesteps
        mappo_config["log_interval"]    = log_interval
        mappo_config["save_interval"]   = save_interval
        if args.seed is not None:
            mappo_config["seed"] = args.seed

        # FIX 3: guard against stale config with wrong num_agents
        assert mappo_config["num_agents"] == 3, (
            f"MAPPO config num_agents={mappo_config['num_agents']} "
            f"but environment expects 3. Fix config.py."
        )

        env   = HealthcareSDNEnv(
            num_nodes=NUM_NODES,
            num_agents=mappo_config["num_agents"],
            seed=mappo_config["seed"]
        )
        agent = MAPPOAgent(env, mappo_config)

        t0 = time.time()
        agent.train(
            total_timesteps=mappo_timesteps,
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
        )
        elapsed = time.time() - t0
        print(f"MAPPO training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ------------------------------------------------------------------
    # STEP 2: Train MADDPG (baseline)
    # ------------------------------------------------------------------
    if not args.eval_only and not args.skip_maddpg:
        print("\n" + "=" * 70)
        print("STEP 2: TRAINING MADDPG (Baseline — Dake et al.)")
        print("=" * 70)

        maddpg_config = MADDPG_CONFIG.copy()
        maddpg_config["total_episodes"] = maddpg_episodes
        if args.seed is not None:
            maddpg_config["seed"] = args.seed

        # FIX 3: same guard
        assert maddpg_config["num_agents"] == 3, (
            f"MADDPG config num_agents={maddpg_config['num_agents']} "
            f"but environment expects 3. Fix config.py."
        )

        env2   = HealthcareSDNEnv(
            num_nodes=NUM_NODES,
            num_agents=maddpg_config["num_agents"],
            seed=maddpg_config["seed"]
        )
        agent2 = MADDPGAgent(env2, maddpg_config)

        t0 = time.time()
        agent2.train(
            total_episodes=maddpg_episodes,
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
        )
        elapsed = time.time() - t0
        print(f"MADDPG training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ------------------------------------------------------------------
    # STEP 3: Evaluate all algorithms
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: EVALUATING ALL ALGORITHMS")
    print("=" * 70)

    run_full_evaluation(results_dir, checkpoints_dir)

    # ------------------------------------------------------------------
    # STEP 4: Generate plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING PLOTS")
    print("=" * 70)

    generate_all_plots(results_dir, images_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
    print(f"\n  Training logs     : {results_dir}/")
    print(f"  Checkpoints       : {checkpoints_dir}/")
    print(f"  Comparison results: {results_dir}/comparison_results.csv")
    print(f"  All plots         : {images_dir}/")
    print(f"\nUseful next commands:")
    print(f"  Full training  : python train_and_evaluate.py --full")
    print(f"  Multi-seed run : python train_and_evaluate.py --full --seed 42")
    print(f"                   python train_and_evaluate.py --full --seed 123")
    print(f"                   python train_and_evaluate.py --full --seed 456")
    print(f"  Plots only     : python train_and_evaluate.py --plot-only")


if __name__ == "__main__":
    main()