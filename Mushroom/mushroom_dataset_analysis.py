from classes.Reader import DataPreprocessor
import classes.analyzer as analyzer

complete_df = DataPreprocessor.get_whole_dataset_as_df(
    "../datasets/mushroom/mushroom.fold.000000.test.arff",
    "../datasets/mushroom/mushroom.fold.000000.train.arff"
)

# test preprocessor
reader = DataPreprocessor(complete_df)
preprocessed_df = reader.fit_transform(cat_encoding="label", num_scaler="minmax")

reader.save("mushroom_preprocessor.joblib")

analyzer.save_dataframe_description_analysis(complete_df)
analyzer.save_feature_distributions_by_class(complete_df)
