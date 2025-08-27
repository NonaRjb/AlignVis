from pathlib import Path
import webdataset as wds
from tqdm import tqdm

ROOT = Path("/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_meg")          # <-- change me
OUT  = Path("/proj/rep-learning-robotics/users/x_nonra/data/alignvis/things_meg")            # <-- change me
OUT.mkdir(parents=True, exist_ok=True)

# Tune shard limits
MAX_BYTES = 1_000_000_000   # ~1 GB per shard
MAX_COUNT = 10000           # or cap by sample count (whichever hits first)

# file extension -> WebDataset field name
EXTMAP = {
    ".jpg": "jpg", ".jpeg": "jpg", ".png": "png",
    ".bmp": "bmp", ".tiff": "tiff",
    ".npy": "npy", ".npz": "npz",
    ".json": "json", ".jsonl": "jsonl", ".txt": "txt", ".csv": "csv",
    ".pkl": "pkl", ".pt": "pt", ".pth": "pth",
    ".edf": "edf", ".fif": "fif", ".mat": "mat", ".h5": "h5",
}
def field_for(path: Path) -> str:
    return EXTMAP.get(path.suffix.lower(), path.suffix.lower().lstrip("."))

# Pick which top-level dirs to shard (auto-detect by default)
top_dirs = [p.name for p in ROOT.iterdir() if p.is_dir()]
# or explicitly:
# top_dirs = ["images", "image_embeddings", "Preprocessed_data_250Hz",
#             "preprocessed_eeg", "raw_eeg"]

for top in top_dirs:
    src = ROOT / top
    pattern = str(OUT / f"{ROOT.name}-{top}-%06d.tar")
    wrote = 0
    with wds.ShardWriter(pattern, maxsize=MAX_BYTES, maxcount=MAX_COUNT) as sink:
        it = (p for p in src.rglob("*") if p.is_file())
        for p in tqdm(it, desc=f"Sharding {top}", unit="file"):
            field = field_for(p)
            rel   = p.relative_to(src)          # keep hierarchy under top folder
            key   = str(rel.with_suffix(""))    # e.g. "s01/run3/eeg_0001"
            sample = {"__key__": key, field: p.read_bytes()}
            sink.write(sample)
            wrote += 1
    print(f"[{top}] wrote {wrote} files into shards at pattern {pattern}")

print("Done. Shards in:", OUT)
