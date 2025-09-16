import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
json_file = BASE_DIR / "Configs" / "training.json"

def count_images_per_pose(json_file, only_downloaded=True):

    with open(json_file, "r") as f:
        data = json.load(f)

    pose_counts = {}
    for img_name, info in data.items():
        if info.get("status") == "downloaded":
            pose = info.get("pose")
            pose_counts[pose] = pose_counts.get(pose, 0) + 1

    return pose_counts

counts = count_images_per_pose(json_file)
print("Images per pose:", counts)
