# -*- coding: utf-8 -*-
"""把 FireRedASR2-AED 的 fp32 权重转成 fp16, 磁盘 4.4GB -> 2.2GB.

用法:
    G:\\AZ\\stt\\venv\\Scripts\\python.exe make_fp16.py

生成 models\\FireRedASR2-AED\\model.fp16.pth.tar 后, 程序会自动优先加载它
(见 voice_typing.py 的 _patch_aed_fast_load), 加载时用 mmap, 不再在内存里
复制两份 4.4GB 权重, 峰值内存从 ~9GB 降到 ~2GB.

确认识别正常后, 原来的 model.pth.tar 可以删掉(省 4.4GB);
想回退就删掉 fp16 文件, 程序会自动用回 fp32.
"""
import os
import time

import torch

BASE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE, "models", "FireRedASR2-AED")
SRC = os.path.join(DIR, "model.pth.tar")
DST = os.path.join(DIR, "model.fp16.pth.tar")


def main():
    if not os.path.exists(SRC):
        print("找不到", SRC)
        return 1
    t0 = time.time()
    print("读取 fp32 权重 (mmap, 不占额外内存) ...", flush=True)
    ck = torch.load(SRC, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model_state_dict"]
    out = {}
    for k, v in sd.items():
        out[k] = v.half().clone() if torch.is_floating_point(v) else v.clone()
    print("转换完成, 正在写出 ...", flush=True)
    torch.save({"model_state_dict": out, "args": ck["args"]}, DST)
    print("完成: %s (%.2f GB, 用时 %.0fs)"
          % (DST, os.path.getsize(DST) / 1024 ** 3, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
