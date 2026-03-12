import os
import urllib.request
from pathlib import Path

def ensure_checkpoint(checkpoint_path, model_type="vit_b"):
    """Downloads the SAM checkpoint if it doesn't exist."""
    path = Path(checkpoint_path)
    if path.exists():
        return str(path)

    # URLs for SAM checkpoints
    urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }

    if model_type not in urls:
        raise ValueError(f"Unknown model type: {model_type}")

    url = urls[model_type]
    print(f"Checkpoint not found at {checkpoint_path}. Downloading from {url}...")
    
    os.makedirs(path.parent, exist_ok=True)
    
    # Progress callback
    def report(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 1e2 / total_size
            s = f"\rDownloading: {percent:5.1f}% ({read_so_far / 1e6:.1f} / {total_size / 1e6:.1f} MB)"
            print(s, end="")
        else:
            print(f"\rRead {read_so_far / 1e6:.1f} MB", end="")

    urllib.request.urlretrieve(url, path, reporthook=report)
    print("\nDownload complete.")
    return str(path)
