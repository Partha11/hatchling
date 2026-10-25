import time

import torch


def save_checkpoint(path, model, meta):
    torch.save({"model": model.state_dict(), "meta": dict(meta)}, path)


def load_checkpoint(path, device, retries=5):
    for attempt in range(retries):
        try:
            data = torch.load(path, map_location=device)
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.0)

    if isinstance(data, dict) and "model" in data and "meta" in data:
        return data["model"], data["meta"]
    return data, {"order": "relative"}