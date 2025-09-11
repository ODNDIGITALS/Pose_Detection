import os
from dotenv import load_dotenv
import boto3
import json

load_dotenv()


def get_aws_credentials():
    return {
        "aws_access_key": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "bucket_name": os.getenv("POSE_S3_BUCKET"),
    }


def create_s3_client():
    """Initialize and return a boto3 S3 client."""
    creds = get_aws_credentials()
    return boto3.client(
        "s3",
        aws_access_key_id=creds["aws_access_key"],
        aws_secret_access_key=creds["aws_secret_key"],
    )

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   
TRAINING_DIR = BASE_DIR

def download_all_images_from_pose_bucket(download_dir=TRAINING_DIR / "training_images", log_file=TRAINING_DIR /"Configs"/ "training.json"):
    creds = get_aws_credentials()
    s3 = create_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    skip_folder = {"outfit_front","tag","swatch","delete"}
    
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            downloaded_data = json.load(f)
    else:
        downloaded_data = {}

    for page in paginator.paginate(Bucket=creds["bucket_name"]):
        for obj in page.get("Contents", []):
            key = obj["Key"]  # "front/image1.jpg"
            if key.lower().endswith(".jpg"):
                image_name = os.path.basename(key)
                pose = key.split("/")[0] if "/" in key else "unknown"
                if pose in skip_folder:
                    continue
                # Skip if already present in log
                if image_name in downloaded_data:
                    print(f"Skipping {key} (already logged)")
                    continue

                local_path = os.path.normpath(os.path.join(download_dir, key))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                try:
                    s3.download_file(creds["bucket_name"], key, local_path)
                    status = "downloaded"
                except Exception as e:
                    status = f"error: {e}"


                # Store in dictionary
                downloaded_data[image_name] = {
                    "path": local_path,
                    "status": status,
                    "pose": pose,
                }

                print(f"{key} → {local_path} [{status}]")

                with open(log_file, "w") as f:
                    json.dump(downloaded_data, f, indent=4)
                    f.flush()
                    os.fsync(f.fileno()) 

    return downloaded_data


# if __name__ == "__main__":
#     metadata = download_all_images_from_pose_bucket()
#     print("Total logged:", len(metadata))


