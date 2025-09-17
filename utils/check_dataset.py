import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
json_file = BASE_DIR / "Configs" / "training.json"

def count_images_per_pose(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    pose_counts = {}

    for _, info in data.items():
        pose = info["pose"]
        status = info["status"]  # only "downloaded" or "trained"

        if pose not in pose_counts:
            pose_counts[pose] = {"downloaded": 0, "trained": 0}

        pose_counts[pose][status] += 1

    return pose_counts


counts = count_images_per_pose(json_file)

print("Images per pose:")
for pose, stats in counts.items():
    print(f"{pose}: Downloaded={stats['downloaded']}, Trained={stats['trained']}")

