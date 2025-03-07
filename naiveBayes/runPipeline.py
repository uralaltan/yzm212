#!/usr/bin/env python3
# RunPipeline.py
# Script to run the entire analysis pipeline

import os
import subprocess
import time


def run_script(script_name):
    """Run a Python script and print its output"""
    print(f"\n{'=' * 50}")
    print(f"Running {script_name}...")
    print(f"{'=' * 50}\n")

    start_time = time.time()

    # Run the script and capture its output
    process = subprocess.Popen(['python', script_name],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)

    # Print output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    # Get any errors
    stderr = process.stderr.read()
    if stderr:
        print("\nErrors:")
        print(stderr)

    end_time = time.time()
    print(f"\nCompleted in {end_time - start_time:.2f} seconds")

    return process.returncode


def main():
    """Run the entire pipeline"""
    print("Starting Naive Bayes classification pipeline...")

    # Create directories if they don't exist
    os.makedirs('images', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Run scikit-learn implementation
    run_script('NaiveBayesScikitLearn.py')

    # Run custom implementation
    run_script('NaiveBayes.py')

    # Run comparison
    run_script('CompareModels.py')

    print("\nPipeline completed!")
    print("Check the generated visualizations and the modelComparison.csv file for results.")


if __name__ == "__main__":
    main()