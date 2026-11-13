import sys, os
from modelscope import snapshot_download

ROOT = r"G:\AZ\stt\models"
jobs = [
    ("xukaituo/FireRedASR2-AED", "FireRedASR2-AED"),
    ("xukaituo/FireRedVAD", "FireRedVAD"),
    ("xukaituo/FireRedPunc", "FireRedPunc"),
]
for repo, name in jobs:
    print(f"=== downloading {repo} -> {name}", flush=True)
    p = snapshot_download(repo, local_dir=os.path.join(ROOT, name))
    print("done:", p, flush=True)

# drop unused TF weights
tf = os.path.join(ROOT, "FireRedPunc", "chinese-lert-base", "tf_model.h5")
if os.path.exists(tf):
    os.remove(tf)
    print("removed tf_model.h5", flush=True)
print("ALL DONE", flush=True)
