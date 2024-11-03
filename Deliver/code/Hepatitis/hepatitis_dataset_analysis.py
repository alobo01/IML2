from classes.Reader import DataPreprocessor
import classes.analyzer as analyzer

complete_df = DataPreprocessor.get_whole_dataset_as_df(
    "../datasets/hepatitis/hepatitis.fold.000000.test.arff",
    "../datasets/hepatitis/hepatitis.fold.000000.train.arff"
)

# save analysis
analyzer.save_dataframe_description_analysis(complete_df)
analyzer.save_feature_distributions_by_class(complete_df)
