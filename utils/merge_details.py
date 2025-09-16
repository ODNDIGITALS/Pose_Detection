import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
json_path = BASE_DIR / "Configs" / "training.json"

if json_path.exists():
    with open(json_path, "r") as f:
        data = json.load(f)

    for img_name, info in data.items():
        if "pose" in info and info["pose"] in ["detail_front", "detail_top"]:
            info["pose"] = "detail_top_front"

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
