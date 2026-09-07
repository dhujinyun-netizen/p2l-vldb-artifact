from pathlib import Path

repo = Path(__file__).resolve().parents[3]
ckpt_dir = repo / "checkpoint/rq_clip_large/Large/Instruct/InBatch"
ckpts = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)

if not ckpts:
    print(f"No checkpoint found in {ckpt_dir}")
    raise SystemExit(0)

latest = ckpts[-1]
for p in ckpts[:-1]:
    p.unlink()
    print(f"Deleted old checkpoint: {p}")

print(f"Kept final/latest checkpoint: {latest}")
