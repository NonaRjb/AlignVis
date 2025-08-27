import io, numpy as np
from PIL import Image
import webdataset as wds

def verify_shard(globpat):
    n = 0
    ds = wds.WebDataset(globpat, handler=wds.handlers.warn_and_continue)
    for sample in ds:
        assert "__key__" in sample  # key should exist
        # try decoding common types (adapt to your content)
        if "jpg" in sample or "png" in sample:
            img = Image.open(io.BytesIO(sample.get("jpg", sample.get("png"))))
            img.verify()  # raises if corrupted
        if "npy" in sample:
            try:
                np.load(io.BytesIO(sample["npy"]), allow_pickle=False)
            except ValueError as e:
                if "Object arrays" in str(e):
                    # fallback only if you trust the source
                    np.load(io.BytesIO(sample["npy"]), allow_pickle=True)
                else:
                    raise
        n += 1
    print(globpat, "OK, samples:", n)

verify_shard("/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg2_wds/things_eeg_2-image_embeddings-{000000..000010}.tar")
verify_shard("/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg2_wds/things_eeg_2-raw_eeg-{000000..000010}.tar")

