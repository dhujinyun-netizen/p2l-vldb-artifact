from pathlib import Path
import shutil

repo = Path(__file__).resolve().parents[3]
ckpt_dir = repo / "checkpoint/rq_clip_large/Large/Instruct/InBatch"
cands = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
if not cands:
    raise SystemExit(f"No .pth checkpoint found in {ckpt_dir}")
src = cands[-1]
dst = repo / "checkpoint/rq_clip_large.pth"
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
print(f"Copied {src} -> {dst}")
