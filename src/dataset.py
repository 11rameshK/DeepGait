from pathlib import Path
import pickle
import random
import numpy as np
import torch

VIEWS = ["000","018","036","054","072","090","108","126","144","162","180"]

def load_pkl(path):
    with open(path, "rb") as f:
        arr = pickle.load(f)
    if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[1:] != (64, 44):
        raise ValueError(f"Invalid PKL shape at {path}: {getattr(arr, 'shape', None)}")
    return arr

def sample_frames(arr, frames=30, train=True):
    n = len(arr)
    if n == 0:
        raise ValueError("Empty sequence")
    if n >= frames:
        if train:
            idx = sorted(random.sample(range(n), frames))
        else:
            idx = np.linspace(0, n - 1, frames).round().astype(int).tolist()
    else:
        idx = np.linspace(0, n - 1, frames).round().astype(int).tolist()
    x = arr[idx].astype(np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(1)  # [T,1,64,44]

class PairedConditionDataset(torch.utils.data.Dataset):
    """
    Returns one identity with aligned NM/BG/CL sequences from the same view.
    This is a practical training dataset for the current DeepGait reproduction.
    """
    def __init__(self, root, subjects, frames=30, train=True, preferred_view="090"):
        self.root = Path(root)
        self.frames = frames
        self.train = train
        self.items = []
        views = [preferred_view] + [v for v in VIEWS if v != preferred_view]
        for sid in subjects:
            s = f"{int(sid):03d}"
            found = None
            for view in views:
                nm = self.root/s/"nm-01"/view/(view+".pkl")
                bg = self.root/s/"bg-01"/view/(view+".pkl")
                cl = self.root/s/"cl-01"/view/(view+".pkl")
                if nm.exists() and bg.exists() and cl.exists():
                    found = (s, view, nm, bg, cl)
                    break
            if found is not None:
                self.items.append(found)
        if not self.items:
            raise RuntimeError("No subjects with NM/BG/CL paired sequences were found")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sid, view, nm_p, bg_p, cl_p = self.items[index]
        nm = sample_frames(load_pkl(nm_p), self.frames, self.train)
        bg = sample_frames(load_pkl(bg_p), self.frames, self.train)
        cl = sample_frames(load_pkl(cl_p), self.frames, self.train)
        return {
            "subject": int(sid),
            "view": view,
            "nm": nm,
            "bg": bg,
            "cl": cl,
        }

def collate_paired(batch):
    return {
        "subject": torch.tensor([x["subject"] for x in batch], dtype=torch.long),
        "view": [x["view"] for x in batch],
        "nm": torch.stack([x["nm"] for x in batch]),
        "bg": torch.stack([x["bg"] for x in batch]),
        "cl": torch.stack([x["cl"] for x in batch]),
    }
