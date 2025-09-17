import argparse
from utils.train_model import train
from utils.merge_details import merge_detail_poses
from utils.aws_config import download_all_images_from_pose_bucket

def main():
    parser = argparse.ArgumentParser(description="Custom Script Commands")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Choose a command to run")

    parser_download = subparsers.add_parser("download_images", help="Download images and merge details")

    parser_train = subparsers.add_parser("train_model", help="Train ResNet50 on existing dataset")
    parser_train.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser_train.add_argument("--batch_size", type=int, default=16, help="Batch size for training")

    parser_download_train = subparsers.add_parser(
        "download_and_train", help="Download images, merge details, and train"
    )
    parser_download_train.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser_download_train.add_argument("--batch_size", type=int, default=16, help="Batch size for training")

    args = parser.parse_args()


    if args.command == "download_images":
        metadata = download_all_images_from_pose_bucket()
        print("Total downloaded:", len(metadata))
        merge_detail_poses()
    elif args.command == "train_model":
        train(epochs=args.epochs, batch_size=args.batch_size)
    elif args.command == "download_and_train":
        metadata = download_all_images_from_pose_bucket()
        print("Total downloaded:", len(metadata))
        merge_detail_poses()
        train(epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
