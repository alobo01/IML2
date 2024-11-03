import argparse
import os
import sys
from pathlib import Path
import importlib.util


def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'sklearn',
        'pydantic',
        'pandas',
        'numpy',
        'joblib',
        'matplotlib',
        'sklearn_relief',
        'scipy',
        'seaborn',
        'scikit_posthocs',
        'tabulate'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Error: Missing required packages:", ', '.join(missing_packages))
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements using:")
        print("pip install -r requirements.txt")
        sys.exit(1)


def import_script(script_path):
    """Dynamically import a Python script from path."""
    spec = importlib.util.spec_from_file_location("module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_dataset_folder(dataset):
    """Validate that the dataset folder exists."""
    if not os.path.exists(dataset):
        print(f"Error: Dataset folder '{dataset}' not found.")
        sys.exit(1)


def run_knn_base(dataset_path, recalculate):
    """Run KNN base configuration study."""
    if recalculate:
        print("\nRunning KNN base training and testing...")
        import_script(dataset_path / "KNN_base_results.py")

    print("\nGenerating KNN base plots...")
    import_script(dataset_path / "KNN_base_plots.py")

    print("\nPerforming KNN base models analysis...")
    import_script(dataset_path / "KNN_base_analysis.py")

    print("\nPerforming KNN hyperparameter analysis...")
    import_script(dataset_path / "KNN_hyperparameter_analysis.py")


def run_svm_base(dataset_path, recalculate):
    """Run SVM base configuration study."""
    svm_script = "svm_algorithm_hepatitis.py" if "Hepatitis" in str(dataset_path) else "svm_algorithm_mushroom.py"

    print("\nPerforming SVM base analysis...")
    import_script(dataset_path / svm_script)


def run_knn_vs_svm(dataset_path):
    """Run comparison between top KNN and SVM models."""
    print("\nPerforming Wilcoxon test between KNN and SVM...")
    import_script(dataset_path / "wilcoxon.py")


def run_knn_reduction(dataset_path, recalculate):
    """Run KNN reduction study."""
    if recalculate:
        print("\nRunning KNN reduction training and testing...")
        import_script(dataset_path / "KNN_reduced_results.py")

    print("\nGenerating KNN reduction plots...")
    import_script(dataset_path / "KNN_reduction_plots.py")

    print("\nPerforming KNN reduction analysis...")
    import_script(dataset_path / "KNN_reduction_analysis.py")


def run_svm_reduction(dataset_path, recalculate):
    """Run SVM reduction study."""
    print("\nPerforming SVM reduction analysis...")
    analysis_script = "svm_reduction_analysis.py"
    import_script(dataset_path / analysis_script)


def run_preprocessing(dataset_path):
    """Run dataset preprocessing."""
    print("\nRunning dataset preprocessing...")
    preprocessing_script = dataset_path / "preprocessing.py"
    if not preprocessing_script.exists():
        print(f"Error: Preprocessing script not found at {preprocessing_script}")
        sys.exit(1)
    import_script(preprocessing_script)


def run_reduction(dataset_path):
    """Run dataset reduction."""
    print("\nRunning dataset reduction...")
    reduction_script = dataset_path / "reduction.py"
    if not reduction_script.exists():
        print(f"Error: Reduction script not found at {reduction_script}")
        sys.exit(1)
    import_script(reduction_script)


def confirm_long_computation():
    """Ask for user confirmation before long computation."""
    print("\nWarning: This computation may take a significant amount of time.")
    response = input("Do you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        sys.exit(0)


def main():
    # Check dependencies before proceeding
    check_dependencies()

    parser = argparse.ArgumentParser(description='Run classification analysis studies.')

    # Dataset selection
    parser.add_argument('--dataset', type=str, choices=['Hepatitis', 'Mushroom'],
                        required=True, help='Dataset to study')

    # Study type selection
    parser.add_argument('--study', type=str, required=True,
                        choices=['preprocess-dataset', 'apply-reduction',
                                 'knn-base', 'svm-base', 'knn-vs-svm',
                                'knn-reduction', 'svm-reduction'],
                        help='Type of study to perform')

    # Recalculation flag
    parser.add_argument('--recalculate', action='store_true',
                        help='Recalculate training and testing data')

    args = parser.parse_args()

    # Validate dataset folder
    dataset_path = Path(args.dataset)
    validate_dataset_folder(dataset_path)

    # Warning about recalculation or long computation
    if args.recalculate or args.study in ['preprocess-dataset', 'apply-reduction']:
        confirm_long_computation()

    # Run appropriate study
    try:
        if args.study == 'knn-base':
            run_knn_base(dataset_path, args.recalculate)
        elif args.study == 'svm-base':
            run_svm_base(dataset_path, args.recalculate)
        elif args.study == 'knn-vs-svm':
            run_knn_vs_svm(dataset_path)
        elif args.study == 'knn-reduction':
            run_knn_reduction(dataset_path, args.recalculate)
        elif args.study == 'svm-reduction':
            run_svm_reduction(dataset_path, args.recalculate)
        elif args.study == 'preprocess-dataset':
            run_preprocessing(dataset_path)
        elif args.study == 'apply-reduction':
            run_reduction(dataset_path)

        print("\nStudy completed successfully!")

    except Exception as e:
        print(f"\nError occurred during execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()