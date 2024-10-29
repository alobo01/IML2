from classes.Reader import DataPreprocessor

for i in range(10):
    # Format the fold index to match the filename
    fold_str = f"{i:06d}"

    # File paths for the current fold
    train_file = f"../datasets/mushroom/mushroom.fold.{fold_str}.train.arff"
    test_file = f"../datasets/mushroom/mushroom.fold.{fold_str}.test.arff"

    # Initialize and fit the preprocessor on the training data
    reader = DataPreprocessor(DataPreprocessor.load_arff(train_file))
    train_data_preprocessed = reader.fit_transform(cat_encoding="target")

    # Save the preprocessor for this fold
    preprocessor_save_path = f"preprocessor_instances/mushroom.fold.{fold_str}.preprocessor.joblib"
    reader.save(preprocessor_save_path)

    # Save train fold as csv
    train_data_preprocessed.to_csv(f"preprocessed_csvs/mushroom.fold.{fold_str}.train.csv")

    # Preprocess and save test fold as csv
    test_data = DataPreprocessor.load_arff(test_file)
    test_data_preprocessed = reader.transform(test_data)
    test_data_preprocessed.to_csv(f"preprocessed_csvs/mushroom.fold.{fold_str}.test.csv")
