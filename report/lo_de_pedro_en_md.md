# Introduction

In recent years, machine learning (ML) techniques have become instrumental in the development of predictive models across various domains, including healthcare, environmental science, and business intelligence. Among the many available ML algorithms, K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) are widely recognized for their distinct approaches to classification problems. KNN, a simple yet powerful instance-based learning method, categorizes data points by considering their proximity to neighboring points, making it intuitive and versatile. Conversely, SVM, a margin-based classifier, aims to construct optimal hyperplanes to separate classes in a high-dimensional feature space, often yielding robust results, particularly in binary classification tasks.

This study aims to systematically compare the performance of KNN and SVM across two distinct datasets from the UCI Machine Learning Repository: the Mushroom dataset and the Hepatitis dataset. The Mushroom dataset comprises features of mushroom species, including cap shape, color, and odor, and presents a binary classification task of distinguishing edible from poisonous mushrooms. The Hepatitis dataset, on the other hand, contains medical records and aims to predict the survival status of patients diagnosed with hepatitis, thus introducing variability in the nature of features and data distribution.

Through this comparative analysis, we investigate the strengths and limitations of KNN and SVM in terms of classification accuracy, computational efficiency, and robustness when applied to datasets with different feature structures and class distributions. The findings are expected to shed light on the suitability of each algorithm under diverse data conditions, ultimately contributing to a more informed selection of algorithms for specific types of classification problems.

# Mushroom Poison Classification

## Introduction

Mushroom foraging can be risky due to the challenge of distinguishing between edible and poisonous species, especially 
as physical characteristics alone may not always make toxicity obvious. To address this, machine learning models have 
been applied to mushroom classification tasks, leveraging attributes such as odor, cap color, and gill characteristics 
to predict toxicity more reliably. Recent research has shown that models trained on these specific characteristics can 
significantly enhance classification accuracy, making mushroom identification safer and more accessible 
for both researchers and foragers ([Tutuncu et al., 2022](https://doi.org/10.1109/MECO55406.2022.9797212)).

## Dataset Features
Below is a breakdown of each feature in the dataset and the values it may take. These attributes capture physical 
characteristics or environmental indicators that can correlate with mushroom toxicity.

1. **cap-shape**: Refers to the shape of the mushroom's cap.
   - Values: bell (`b`), conical (`c`), convex (`x`), flat (`f`), knobbed (`k`), sunken (`s`)

2. **cap-surface**: Describes the surface texture of the cap.
   - Values: fibrous (`f`), grooves (`g`), scaly (`y`), smooth (`s`)

3. **cap-color**: Indicates the cap color, which may be associated with toxicity.
   - Values: brown (`n`), buff (`b`), cinnamon (`c`), gray (`g`), green (`r`), pink (`p`), purple (`u`), red (`e`), white (`w`), yellow (`y`)

4. **bruises?**: Shows if the mushroom cap bruises when damaged.
   - Values: bruises (`t`), no (`f`)

5. **odor**: Odor can often indicate toxicity in mushrooms.
   - Values: almond (`a`), anise (`l`), creosote (`c`), fishy (`y`), foul (`f`), musty (`m`), none (`n`), pungent (`p`), spicy (`s`)

6. **gill-attachment**: Describes the attachment of gills to the mushroom stem.
   - Values: attached (`a`), descending (`d`), free (`f`), notched (`n`)

7. **gill-spacing**: Describes the spacing of the mushroom’s gills.
   - Values: close (`c`), crowded (`w`), distant (`d`)

8. **gill-size**: Refers to the size of the gills.
   - Values: broad (`b`), narrow (`n`)

9. **gill-color**: Indicates the gill color, which may relate to toxicity.
   - Values: black (`k`), brown (`n`), buff (`b`), chocolate (`h`), gray (`g`), green (`r`), orange (`o`), pink (`p`), purple (`u`), red (`e`), white (`w`), yellow (`y`)

10. **stalk-shape**: Describes the shape of the mushroom stalk.
    - Values: enlarging (`e`), tapering (`t`)

11. **stalk-root**: Describes the root type of the stalk.
    - Values: bulbous (`b`), club (`c`), cup (`u`), equal (`e`), rhizomorphs (`z`), rooted (`r`), missing (`?`)

12. **stalk-surface-above-ring**: Surface texture above the ring on the stalk.
    - Values: fibrous (`f`), scaly (`y`), silky (`k`), smooth (`s`)

13. **stalk-surface-below-ring**: Surface texture below the ring on the stalk.
    - Values: fibrous (`f`), scaly (`y`), silky (`k`), smooth (`s`)

14. **stalk-color-above-ring**: Color of the stalk above the ring.
    - Values: brown (`n`), buff (`b`), cinnamon (`c`), gray (`g`), orange (`o`), pink (`p`), red (`e`), white (`w`), yellow (`y`)

15. **stalk-color-below-ring**: Color of the stalk below the ring.
    - Values: brown (`n`), buff (`b`), cinnamon (`c`), gray (`g`), orange (`o`), pink (`p`), red (`e`), white (`w`), yellow (`y`)

16. **veil-type**: Type of veil covering the mushroom.
    - Values: partial (`p`), universal (`u`)

17. **veil-color**: Color of the veil.
    - Values: brown (`n`), orange (`o`), white (`w`), yellow (`y`)

18. **ring-number**: Number of rings on the mushroom stalk.
    - Values: none (`n`), one (`o`), two (`t`)

19. **ring-type**: Type of ring on the mushroom stalk.
    - Values: cobwebby (`c`), evanescent (`e`), flaring (`f`), large (`l`), none (`n`), pendant (`p`), sheathing (`s`), zone (`z`)

20. **spore-print-color**: Color of the spore print.
    - Values: black (`k`), brown (`n`), buff (`b`), chocolate (`h`), green (`r`), orange (`o`), purple (`u`), white (`w`), yellow (`y`)

21. **population**: Population density where the mushroom grows.
    - Values: abundant (`a`), clustered (`c`), numerous (`n`), scattered (`s`), several (`v`), solitary (`y`)

22. **habitat**: Type of habitat where the mushroom is found.
    - Values: grasses (`g`), leaves (`l`), meadows (`m`), paths (`p`), urban (`u`), waste (`w`), woods (`d`)

23. **class**: Classification of the mushroom as edible or poisonous.
    - Values: edible (`e`), poisonous (`p`)

## Descriptive Statistics of the dataset

# Class Distribution

For this dataset, both classes have a similar number of elements, making it balanced:

| class   |   Count |
|---------|---------|
| e       |    4208 |
| p       |    3916 |

**Types of features**:

| Ordinal         | Nominal                  | Binary      |
|-------------------------|--------------------------|---------------------|
| gill-spacing            | cap-shape                | bruises?           |
| ring-number             | cap-surface              | gill-size          |
| population              | cap-color                | stalk-shape        |
|                         | odor                     |                     |
|                         | gill-attachment          |                     |
|                         | gill-color               |                     |
|                         | stalk-root               |                     |
|                         | stalk-surface-above-ring |                     |
|                         | stalk-surface-below-ring |                     |
|                         | stalk-color-above-ring   |                     |
|                         | stalk-color-below-ring   |                     |
|                         | veil-type                |                     |
|                         | veil-color               |                     |
|                         | ring-type                |                     |
|                         | spore-print-color        |                     |
|                         | habitat                  |                     |

Most of the features are nominal, however there are also ordinal and binary features, 
this information is useful to decide which encoder to use (see in preprocessing section).

The dataset contains 8,124 samples, here is the pandas description of the numerical columns of the dataset, 
and below are the missing values count:

|                | count | unique | top | freq |
|----------------|-------|--------|-----|------|
| cap-shape      | 8124  | 6      | x   | 3656 |
| cap-surface    | 8124  | 4      | y   | 3244 |
| cap-color      | 8124  | 10     | n   | 2284 |
| bruises?       | 8124  | 2      | f   | 4748 |
| odor           | 8124  | 9      | n   | 3528 |
| gill-attachment | 8124  | 2      | f   | 7914 |
| gill-spacing   | 8124  | 2      | c   | 6812 |
| gill-size      | 8124  | 2      | b   | 5612 |
| gill-color     | 8124  | 12     | b   | 1728 |
| stalk-shape    | 8124  | 2      | t   | 4608 |
| stalk-root     | 5644  | 4      | b   | 3776 |
| stalk-surface-above-ring | 8124 | 4  | s   | 5176 |
| stalk-surface-below-ring  | 8124 | 4  | s   | 4936 |
| stalk-color-above-ring | 8124 | 9   | w   | 4464 |
| stalk-color-below-ring | 8124 | 9   | w   | 4384 |
| veil-type      | 8124  | 1      | p   | 8124 |
| veil-color     | 8124  | 4      | w   | 7924 |
| ring-number    | 8124  | 3      | o   | 7488 |
| ring-type      | 8124  | 5      | p   | 3968 |
| spore-print-color | 8124 | 9    | w   | 2388 |
| population      | 8124 | 6      | v   | 4040 |
| habitat        | 8124  | 7      | d   | 3148 |
| class          | 8124  | 2      | e   | 4208 |


### Missing Values Count

This dataset contains only one column with missing values, "stalk-root", since 
the proportion of missing values represents roughly a 30% of the total, we have decided to
keep it, as the remaining 70% of samples are still numerous.

|            |   Missing Values Count |
|------------|------------------------|
| stalk-root |                   2480 |


## Feature distribution by class

We generate plots to analyse how the classes (e and p) are distributed for all the features 
in the dataset, each figure (saved in the plots_and_tables/feature_distributions folder) represents 
the distribution normalized taking into account the class imbalance (left plot) and with the absolute
values (right plot).

Analyzing the figures we can see there are certain features that have a clear separation in the 
classes, some examples are:

### Odor

Certain odors are specially significant, in particular a,c,m,p,s and y values, are completely class-separated:

![odor_distribution.png](plots_and_tables/feature_distributions/odor_distribution.png)


### Spore print color

Values b,o,r,u and y are completely class-separated, with the rest of the values being almost completely separated:

![spore-print-color_distribution.png](plots_and_tables/feature_distributions/spore-print-color_distribution.png)

Although we are not performing feature selection, these observations help us understand how the 
weighting applied in the KNN algorithm should be distributed throughout the features.

## Preprocessing methods

To handle missing values we have used sklearn.SimpleImputer from sklearn, replacing with the most occurring value.
To encode the features we have used sklearn.ordinalImputer for the ordinal features (gill-spacing, 
ring-number and population) and for the binary features (bruises, gill-size, stalk-shape).

The remaining 16 features are nominal, with each of them having at least 3 unique values, and the most numerous one
having 13 distinct values. Using one-hot encoder would result in too many dimensions, and using ordinal encoding
would give an unwanted order to the features values. To tackle this problem we have used targetEncoder, which encodes
the features "based on a shrunk estimate of the average target values for observations belonging to the category" (from 
sklearn documentation), with this method the values are still encoded with an order but in a meaningful way.


### Introduction to Hepatitis and Dataset Overview

Hepatitis is an inflammatory condition of the liver, often caused by viral infections, autoimmune conditions, or toxins such as alcohol and drugs. Inflammation due to hepatitis can impair liver function, potentially leading to severe complications, including liver failure, cirrhosis, or liver cancer if untreated. Hepatitis types A, B, C, D, and E vary in their transmission routes and severity, but all impact liver health (Source: [CDC](https://www.cdc.gov/hepatitis/index.htm)).

The provided dataset focuses on characteristics relevant to hepatitis prognosis (the predicted outcome of a disease given the state and characteristics of a patient), classifying patients into those who survived ("LIVE") or succumbed to the disease ("DIE"). Below is a breakdown of the dataset attributes, grouped by type to highlight their roles in assessing hepatitis:

#### Demographic and Basic Information
- **AGE**: Patient's age, ranging from 10 to 80 years.
- **SEX**: Gender, which may influence disease susceptibility and progression.

#### Symptom-Related Features
- **FATIGUE**, **MALAISE (feeling unwell)**, **ANOREXIA**: Common symptoms associated with hepatitis, which can signal disease progression and impact on general health.

#### Physical Examination Attributes
- **LIVER_BIG**, **LIVER_FIRM**: Physical indications of liver enlargement and firmness, often found in patients with advanced hepatitis.
- **SPLEEN_PALPABLE**: Detectable spleen enlargement, which may indicate advanced liver disease.
- **SPIDERS**: Presence of spider angiomas, small dilated blood vessels often associated with liver disease.
- **ASCITES**: Accumulation of fluid in the abdomen, a sign of severe liver impairment.
- **VARICES**: Enlarged veins in the digestive tract, which can develop in advanced liver disease due to increased pressure in the liver.

#### Laboratory Test Results
- **BILIRUBIN**: A liver function marker, where elevated levels indicate impaired liver processing of bilirubin.
- **ALK_PHOSPHATE**: Elevated levels of this enzyme can signal liver damage or blockage of bile flow.
- **SGOT**: Also known as AST, this enzyme is released with liver damage.
- **ALBUMIN**: Low levels may indicate poor liver function or malnutrition.
- **PROTIME**: Measures blood clotting time; impaired liver function often prolongs clotting.

#### Treatment and History Indicators
- **STEROID**, **ANTIVIRALS**: Treatment options that may impact patient outcomes.
- **HISTOLOGY**: Indicates whether a histological examination (microscopic examination of liver tissue) was performed, providing a more direct assessment of liver damage.

#### Target Class
- **Class**: Patient outcome, with two categories: "LIVE" or "DIE," representing survival status.

### Descriptive Statistics of the dataset

# Class Distribution

We can observe that the dataset is very unbalanced towards the LIVE class

| Class   |   Count |
|---------|---------|
| LIVE    |     123 |
| DIE     |      32 |

**Numerical Columns**:

AGE, BILIRUBIN, ALK_PHOSPHATE, SGOT, ALBUMIN, PROTIME

**Categorical Columns**:

SEX, STEROID, ANTIVIRALS, FATIGUE, MALAISE, ANOREXIA, LIVER_BIG, LIVER_FIRM, SPLEEN_PALPABLE, SPIDERS, ASCITES, VARICES, HISTOLOGY, Class

The dataset contains 155 samples, here is the pandas description of the numerical columns of the dataset, 
and below are the missing values count:

|       |      AGE |   BILIRUBIN |   ALK_PHOSPHATE |     SGOT |    ALBUMIN |   PROTIME |
|-------|----------|-------------|-----------------|----------|------------|-----------|
| count | 155      |   149       |        126      | 151      | 139        |   88      |
| mean  |  41.2    |     1.42752 |        105.325  |  85.894  |   3.81727  |   61.8523 |
| std   |  12.5659 |     1.21215 |         51.5081 |  89.6509 |   0.651523 |   22.8752 |
| min   |   7      |     0.3     |         26      |  14      |   2.1      |    0      |
| 25%   |  32      |     0.7     |         74.25   |  31.5    |   3.4      |   46      |
| 50%   |  39      |     1       |         85      |  58      |   4        |   61      |
| 75%   |  50      |     1.5     |        132.25   | 100.5    |   4.2      |   76.25   |
| max   |  78      |     8       |        295      | 648      |   6.4      |  100      |

### Missing Values Count

There are a few features with a big portion of the data missing, in the case of the protime
feature, almost half of the values are missing, we have set a threshold of 30% for the missing values,
for this dataset only the protime feature has been removed.
For the rest of features, we have replaced with the median value for numerical features and the mode 
for categorical features.

|                 |   Missing Values Count |
|-----------------|------------------------|
| STEROID         |                      1 |
| FATIGUE         |                      1 |
| MALAISE         |                      1 |
| ANOREXIA        |                      1 |
| SGOT            |                      4 |
| SPIDERS         |                      5 |
| ASCITES         |                      5 |
| SPLEEN_PALPABLE |                      5 |
| VARICES         |                      5 |
| BILIRUBIN       |                      6 |
| LIVER_BIG       |                     10 |
| LIVER_FIRM      |                     11 |
| ALBUMIN         |                     16 |
| ALK_PHOSPHATE   |                     29 |
| PROTIME         |                     67 |

## Feature distribution by class

We generate plots to analyse how the classes (LIVE and DIE) are distributed for all the features 
in the dataset, each figure (saved in the plots_and_tables/feature_distributions folder) represents 
the distribution scaled taking into account the class imbalance (left plot) and with the absolute
values (right plot).

Analyzing the figures we can see there are certain features that have a clear separation in the 
classes, some examples are:

### Sex distribution

This plot shows that in this dataset all the patients that died were females:

![SEX_distribution.png](plots_and_tables/feature_distributions/SEX_distribution.png)


### Albumin distribution

This plot shows that all patients that died had an albumin value lower than 0.4:

![ALBUMIN_distribution.png](plots_and_tables/feature_distributions/ALBUMIN_distribution.png)


## Preprocessing methods

### For categorical features

Since all categorical features only have two types, they are encoded with ones and zeros using ordinalEncoder

### For numerical features

Numerical features are scaled in the range 0,1 using minMaxScaler