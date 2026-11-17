"""Self-play training loop (AlphaZero-lite).

    python -m minichess.train --benchmark            # measure speed, suggest --games
    python -m minichess.train --time-budget 45       # train for ~45 minutes
    python -m minichess.train --resume ...           # continue from checkpoints/

Each iteration: self-play games with the current net -> replay buffer -> gradient steps ->
monitoring matches (vs a random mover, and vs best.pt every few iterations). The latest net
always generates the next data; best.pt is only a monitoring reference / the net to play against.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .arena import parallel_match
from .encode import encode_state, legal_mask
from .game import initial_state
from .model import PolicyValueNet, load_checkpoint, load_model, predict, save_checkpoint
from .replay import ReplayBuffer
from .selfplay import SELFPLAY_MAX_PLY, generate, run_games


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the mini chess network by self-play.")
    p.add_argument("--time-budget", type=float, default=45.0, help="minutes (checked between iterations)")
    p.add_argument("--iterations", type=int, default=1000, help="maximum number of iterations")
    p.add_argument("--games", type=int, default=48, help="self-play games per iteration")
    p.add_argument("--sims", type=int, default=30, help="MCTS simulations per move in self-play")
    p.add_argument("--steps", type=int, default=150, help="gradient steps per iteration")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--log-every", type=int, default=25, help="print loss every N gradient steps (0 = off)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decay", type=float, default=0.95, help="learning-rate factor per iteration")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--buffer", type=int, default=30_000, help="replay buffer capacity (positions)")
    p.add_argument("--max-ply", type=int, default=SELFPLAY_MAX_PLY)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--eval-games", type=int, default=16, help="games vs a random mover per iteration")
    p.add_argument("--eval-sims", type=int, default=20)
    p.add_argument("--compare-every", type=int, default=5, help="iterations between matches vs best.pt")
    p.add_argument("--compare-games", type=int, default=20)
    p.add_argument("--no-final-eval", action="store_true")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--benchmark", action="store_true", help="measure speed and exit")
    return p.parse_args(argv)


def train_steps(net: PolicyValueNet, opt, buf: ReplayBuffer, steps: int, batch: int,
                log_every: int = 0, tag: str = ""):
    """Run gradient steps on random batches. Returns mean (policy loss, value loss).

    If ``log_every`` > 0, prints the mean loss over each window of that many steps.
    """
    net.train()
    pol = val = 0.0
    win_pol = win_val = 0.0
    for i in range(1, steps + 1):
        obs, pi, z = buf.sample(batch)
        logits, v = net(torch.from_numpy(obs))
        policy_loss = -(torch.from_numpy(pi) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        value_loss = F.mse_loss(v, torch.from_numpy(z))
        opt.zero_grad()
        (policy_loss + value_loss).backward()
        opt.step()
        p_item, v_item = policy_loss.item(), value_loss.item()
        pol += p_item
        val += v_item
        if log_every > 0:
            win_pol += p_item
            win_val += v_item
            if i % log_every == 0 or i == steps:
                n = log_every if i % log_every == 0 else i % log_every
                print(f"{tag}  step {i:4d}/{steps} | pol {win_pol / n:.3f} val {win_val / n:.3f}",
                      flush=True)
                win_pol = win_val = 0.0
    return pol / max(1, steps), val / max(1, steps)


def latest_checkpoint(directory: Path):
    best = None
    for path in directory.glob("iter_*.pt"):
        m = re.fullmatch(r"iter_(\d+)\.pt", path.name)
        if m and (best is None or int(m.group(1)) > best[1]):
            best = (path, int(m.group(1)))
    return best


def benchmark(args) -> None:
    torch.set_num_threads(1)
    net = PolicyValueNet(args.channels, args.blocks)
    s = initial_state()
    obs, mask = encode_state(s), legal_mask(s)
    t = time.perf_counter()
    for _ in range(300):
        predict(net, obs, mask)
    forward_ms = (time.perf_counter() - t) / 300 * 1000

    t = time.perf_counter()
    _, _, _, stats = run_games(net, 3, args.sims, np.random.SeedSequence(0), args.max_ply)
    elapsed = time.perf_counter() - t
    plies_per_game = stats["plies"] / stats["games"]
    sec_per_game = elapsed / stats["games"]

    torch.set_num_threads(args.workers)
    buf = ReplayBuffer(512, rng=np.random.default_rng(0))
    rng = np.random.default_rng(0)
    pi = rng.random((512, 625)).astype(np.float32)
    buf.add(rng.integers(0, 2, (512, 12, 5, 5)).astype(np.uint8), pi / pi.sum(1, keepdims=True),
            np.zeros(512, np.float32))
    net2 = PolicyValueNet(args.channels, args.blocks)
    opt = torch.optim.Adam(net2.parameters())
    t = time.perf_counter()
    train_steps(net2, opt, buf, 20, args.batch)
    step_ms = (time.perf_counter() - t) / 20 * 1000

    selfplay_target = 90.0
    suggested = max(args.workers, int(selfplay_target * args.workers / sec_per_game))
    print(f"net forward (1 position, 1 thread): {forward_ms:.2f} ms")
    print(f"self-play: {sec_per_game:.1f} s/game/worker at {args.sims} sims "
          f"({plies_per_game:.0f} plies/game, {stats['plies'] * args.sims / elapsed:.0f} sims/s/worker)")
    print(f"training step (batch {args.batch}, {args.workers} threads): {step_ms:.1f} ms "
          f"-> {args.steps} steps = {args.steps * step_ms / 1000:.1f} s")
    print(f"suggested: --games {suggested}  (~{selfplay_target:.0f} s of self-play per iteration "
          f"on {args.workers} workers)")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.benchmark:
        benchmark(args)
        return

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.workers))
    net = PolicyValueNet(args.channels, args.blocks)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buf = ReplayBuffer(args.buffer, rng=np.random.default_rng(args.seed))
    start_iter = 0

    found = latest_checkpoint(ckpt_dir) if args.resume else None
    if found:
        ckpt = load_checkpoint(found[0])
        net = PolicyValueNet(ckpt["channels"], ckpt["blocks"])
        net.load_state_dict(ckpt["model"])
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if ckpt["optimizer"] is not None:
            opt.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iteration"]
        if (ckpt_dir / "buffer.npz").exists():
            buf.load(ckpt_dir / "buffer.npz")
        print(f"resumed from {found[0].name} (buffer: {len(buf)} positions)")
    else:
        save_checkpoint(ckpt_dir / "iter_000.pt", net, opt, 0)   # untrained reference
    if not (ckpt_dir / "best.pt").exists():
        save_checkpoint(ckpt_dir / "best.pt", net, None, start_iter)

    pool = None
    if args.workers > 1:
        pool = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"))
    t_start = time.time()
    last_duration = 0.0
    budget = args.time_budget * 60
    it = start_iter
    try:
        for it in range(start_iter + 1, args.iterations + 1):
            if it > start_iter + 1 and time.time() - t_start + last_duration > budget:
                print("time budget reached")
                it -= 1
                break
            t_iter = time.time()
            for group in opt.param_groups:
                group["lr"] = args.lr * args.lr_decay ** (it - 1)

            obs, pi, z, stats = generate(net, args.games, args.sims,
                                         np.random.SeedSequence([args.seed, it]), pool,
                                         args.workers, args.max_ply)
            buf.add(obs, pi, z)
            t_sp = time.time() - t_iter
            print(f"iter {it:3d}: self-play done in {t_sp:.0f}s, training ...", flush=True)
            ploss, vloss = (train_steps(net, opt, buf, args.steps, args.batch,
                                        args.log_every, f"iter {it:3d}")
                            if len(buf) >= args.batch else (float("nan"), float("nan")))

            vs_random = parallel_match(net, None, args.eval_games, args.eval_sims,
                                       np.random.SeedSequence([args.seed, it, 1]), pool,
                                       args.workers, args.max_ply)
            vs_best = None
            if args.compare_every and it % args.compare_every == 0:
                vs_best = parallel_match(net, load_model(ckpt_dir / "best.pt"), args.compare_games,
                                         args.eval_sims, np.random.SeedSequence([args.seed, it, 2]),
                                         pool, args.workers, args.max_ply)
                if vs_best["score"] >= 0.5:
                    save_checkpoint(ckpt_dir / "best.pt", net, None, it)

            save_checkpoint(ckpt_dir / f"iter_{it:03d}.pt", net, opt, it)
            if it % 5 == 0:
                buf.save(ckpt_dir / "buffer.npz")
            last_duration = time.time() - t_iter

            record = {
                "iter": it, "policy_loss": ploss, "value_loss": vloss, "buffer": len(buf),
                "games": stats["games"], "white": stats["white"], "black": stats["black"],
                "draws": stats["draws"], "avg_plies": stats["plies"] / max(1, stats["games"]),
                "vs_random": vs_random["score"], "vs_best": vs_best["score"] if vs_best else None,
                "selfplay_s": round(t_sp, 1), "iter_s": round(last_duration, 1),
                "elapsed_min": round((time.time() - t_start) / 60, 1),
            }
            with open(ckpt_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"iter {it:3d} | self-play W{stats['white']} B{stats['black']} D{stats['draws']} "
                  f"avg {record['avg_plies']:.0f} plies | buf {len(buf)} | "
                  f"loss pol {ploss:.3f} val {vloss:.3f} | vs random {vs_random['score']:.2f}"
                  + (f" | vs best {vs_best['score']:.2f}" if vs_best else "")
                  + f" | {last_duration:.0f}s ({record['elapsed_min']} min total)", flush=True)

        buf.save(ckpt_dir / "buffer.npz")
        if not args.no_final_eval and it > 0:
            print("final evaluation ...")
            final_sims = max(args.eval_sims, args.sims)
            r = parallel_match(net, None, 40, final_sims, np.random.SeedSequence([args.seed, 99]),
                               pool, args.workers, args.max_ply)
            print(f"vs random mover: {r['wins']}W {r['draws']}D {r['losses']}L "
                  f"(score {r['score']:.2f}; target >= 0.90)")
            r0 = parallel_match(net, load_model(ckpt_dir / "iter_000.pt"), 20, final_sims,
                                np.random.SeedSequence([args.seed, 98]), pool, args.workers,
                                args.max_ply)
            print(f"vs untrained net: {r0['wins']}W {r0['draws']}D {r0['losses']}L "
                  f"(score {r0['score']:.2f}; target >= 0.50)")
    finally:
        if pool is not None:
            pool.shutdown()


if __name__ == "__main__":
    main()
