import argparse
from utils.train_model import train

def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 on Custom Dataset")

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)"
    )

    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
