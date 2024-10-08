from Reader import DataPreprocessor

reader = DataPreprocessor("test_reader/adult.fold.000000.train.arff")
pd = reader.fit_transform(config_path="test_reader/adult.fold.config.json")
print()