import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
json_file = BASE_DIR / "Configs" / "training.json"

with open(json_file, "r") as f:
    data = json.load(f)

print(f"Total images in JSON before balancing: {len(data)}")

pose_counts = {}
for img_name, info in data.items():
    if info.get("status") == "downloaded":
        pose = info["pose"]
        pose_counts[pose] = pose_counts.get(pose, 0) + 1

print("Pose counts before:", pose_counts)

min_count = min(pose_counts.values())
print(f"Target = {min_count} images per pose")

for pose, count in pose_counts.items():
    if count > min_count:

        candidates = [img for img, info in data.items() if info.get("pose") == pose]
        to_delete = random.sample(candidates, count - min_count)

        for victim in to_delete:
            del data[victim]

        print(f"Deleted {len(to_delete)} from pose '{pose}'")

with open(json_file, "w") as f:
    json.dump(data, f, indent=4)
