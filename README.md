### Team

- Antonio Lobo Santos
- Pedro Agúndez Fernandez
- Bruno Sánchez Gómez
- María del Carmen Ramírez Trujillo


### Running script for the first time
These sections show how to create virtual environment for
our script and how to install dependencies
1. Open folder in terminal
```bash
cd <root_folder_of_project>/
```
2. Create virtual env
```bash
python3 -m venv venv/
```
3. Open virtual env
```bash
source venv/bin/activate
```
4. Install required dependencies
```bash
pip install -r requirements.txt
```
you can check if dependencies were installed by running next
command,it should print list with installed dependencies
```bash
pip list
```
5. Close virtual env
```bash
deactivate
```

## Execute scripts

1. Open virtual env
```bash
source venv/bin/activate
```

2. Running the script
```bash
python3 main.py --dataset <dataset_name> --study <study_type> [--recalculate]
```

Available options:
- `dataset`: Choose between `Hepatitis` or `Mushroom`
- `study`: Choose one of the following:
  - `preprocess-dataset`: Calculate the preprocessing of the original folds
  - `apply-reduction`: Calculate the reduced folds with the different reduction methods
  - `knn-base`: KNN model and hyperparameter analysis 
  - `svm-base`: SVM model and hyperparameter analysis
  - `knn-vs-svm`: Compare KNN and SVM models
  - `knn-reduction`: Compare reduction methods with KNN model
  - `svm-reduction`: Compare reduction methods with SVM model
- `recalculate`: Optional flag for recalculating training/testing results of KNN/SVM models.

**Warning:** The preprocessed datasets, reduced folds and training/testing results are already included with the code, and their re-calculation will take a significant amount of time (up to several hours). Running `preprocess-dataset`, `apply-reduction` or include the `recalculate` flag is **not** necessary in order to run the analyses.

Example commands:
```bash
# Preprocess Hepatitis dataset
python3 main.py --dataset Hepatitis --study preprocess-dataset

# Run KNN analysis on Mushroom dataset with recalculation
python3 main.py --dataset Mushroom --study knn-base --recalculate

# Compare KNN and SVM on Hepatitis dataset
python3 main.py --dataset Hepatitis --study knn-vs-svm
```

To display help menu:
```bash
python3 main.py --help
```

3. Close virtual env
```bash
deactivate
```