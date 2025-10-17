# examples/train.py

import argparse
import time

def train(smoke_test=False):
    if smoke_test:
        print("Running smoke test...")
        time.sleep(1)  # Simulate quick training
        print("Smoke test passed ✅")
    else:
        print("Running full training...")
        time.sleep(5)  # Simulate long training
        print("Training complete ✅")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help="Run quick training check")
    args = parser.parse_args()

    train(smoke_test=args.smoke_test)
