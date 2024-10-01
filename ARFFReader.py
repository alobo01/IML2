import scipy.io.arff as arff
import pandas as pd

class ARFFReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.meta = None

    def read_arff(self):
        try:
            # Load the ARFF file
            data, meta = arff.loadarff(self.file_path)
            self.data = pd.DataFrame(data)
            self.meta = meta
            return self.data
        except Exception as e:
            print(f"Error reading ARFF file: {e}")
            return None

    def get_metadata(self):
        if self.meta:
            return self.meta
        else:
            print("ARFF file not loaded yet.")
            return None

# Example usage
reader = ARFFReader('datasets/iris.arff')
df = reader.read_arff()
print(df.head())
metadata = reader.get_metadata()
print(metadata)
