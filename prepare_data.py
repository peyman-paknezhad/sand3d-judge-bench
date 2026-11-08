"""Step 1: download the Sand3D-Objaverse validation split (2,245 programs) to data/val.pkl."""
import os
import pandas as pd
from huggingface_hub import hf_hub_download

# ---- knobs ----
REPO = "Sandflow/sand3d-objaverse"
FILE = "data/objaverse/validation-00000-of-00001.parquet"
OUT = "data/val.pkl"
# ---------------
os.makedirs(os.path.dirname(OUT), exist_ok=True)
d = pd.read_parquet(hf_hub_download(REPO, FILE, repo_type="dataset"))
d.to_pickle(OUT)
print(f"{len(d)} programs, columns {list(d.columns)} -> {OUT}")
