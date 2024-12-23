import os.path
from classes.Reader import DataPreprocessor
from test_reducing.dyingyang import dataset

if __name__ == "__main__":
    dataset_path = '..\\Hepatitis'
else:
    dataset_path = 'Hepatitis'

original_folds_path = os.path.join(dataset_path, '../datasets/hepatitis')
path1 = os.path.join(original_folds_path, "hepatitis.fold.000000.train.arff")
path2 = os.path.join(original_folds_path, "hepatitis.fold.000000.test.arff")

complete_df = DataPreprocessor.get_whole_dataset_as_df(
    path1,
    path2
)

binary_features  = [
    "SEX",
    "STEROID",
    "ANTIVIRALS",
    "FATIGUE",
    "MALAISE",
    "ANOREXIA",
    "LIVER_BIG",
    "LIVER_FIRM",
    "SPLEEN_PALPABLE",
    "SPIDERS",
    "ASCITES",
    "VARICES",
    "HISTOLOGY"
]

removed_features = DataPreprocessor.get_columns_with_missing_values_over_threshold(complete_df)

for i in range(10):
    # Format the fold index to match the filename
    fold_str = f"{i:06d}"

    # File paths for the current fold
    train_file = os.path.join(original_folds_path, f"hepatitis.fold.{fold_str}.train.arff")
    test_file = os.path.join(original_folds_path, f"hepatitis.fold.{fold_str}.test.arff")

    # Load as Dataframes and remove columns with missing values
    train_data = DataPreprocessor.load_arff(train_file).drop(columns=removed_features)
    test_data = DataPreprocessor.load_arff(test_file).drop(columns=removed_features)

    # Initialize and fit the preprocessor on the training data and transform
    reader = DataPreprocessor(train_data, class_column="Class")
    train_data_preprocessed = reader.fit_transform(ordinal_features=binary_features)

    # Save the preprocessor for this fold
    preprocessor_save_path = os.path.join(dataset_path, f"preprocessor_instances/hepatitis.fold.{fold_str}.preprocessor.joblib")
    reader.save(preprocessor_save_path)

    # Save train fold as csv
    preprocessed_train_path = os.path.join(dataset_path, f"preprocessed_csvs/hepatitis.fold.{fold_str}.train.csv")
    train_data_preprocessed.to_csv(preprocessed_train_path)

    # Preprocess and save test fold as csv
    test_data_preprocessed = reader.transform(test_data)
    preprocessed_test_path = os.path.join(dataset_path, f"preprocessed_csvs/hepatitis.fold.{fold_str}.test.csv")
    test_data_preprocessed.to_csv(preprocessed_test_path)
