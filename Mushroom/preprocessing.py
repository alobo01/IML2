from classes.Reader import DataPreprocessor

# Load the complete dataset and identify columns with excessive missing values
complete_df = DataPreprocessor.get_whole_dataset_as_df(
    "../datasets/mushroom/mushroom.fold.000000.train.arff",
    "../datasets/mushroom/mushroom.fold.000000.test.arff",
)

# Binary features are considered ordinal for the preprocesser
ordinal_features = [
    "gill-spacing",
    "ring-number",
    "population",
    "bruises?",
    "gill-size",
    "stalk-shape"
]

# Remove columns with missing values above a specified threshold
removed_features = DataPreprocessor.get_columns_with_missing_values_over_threshold(complete_df)

for i in range(10):
    # Format the fold index to match the filename
    fold_str = f"{i:06d}"

    # File paths for the current fold
    train_file = f"../datasets/mushroom/mushroom.fold.{fold_str}.train.arff"
    test_file = f"../datasets/mushroom/mushroom.fold.{fold_str}.test.arff"

    # Load training and test dataframes and drop columns with high missing values
    train_data = DataPreprocessor.load_arff(train_file).drop(columns=removed_features)
    test_data = DataPreprocessor.load_arff(test_file).drop(columns=removed_features)

    # Initialize and fit the preprocessor on the training data and transform
    reader = DataPreprocessor(train_data, class_column="class")
    train_data_preprocessed = reader.fit_transform(ordinal_features=ordinal_features)

    # Save the preprocessor for this fold
    preprocessor_save_path = f"preprocessor_instances/mushroom.fold.{fold_str}.preprocessor.joblib"
    reader.save(preprocessor_save_path)

    # Save train fold as csv
    train_data_preprocessed.to_csv(f"preprocessed_csvs/mushroom.fold.{fold_str}.train.csv")

    # Preprocess and save test fold as csv
    test_data_preprocessed = reader.transform(test_data)
    test_data_preprocessed.to_csv(f"preprocessed_csvs/mushroom.fold.{fold_str}.test.csv")
