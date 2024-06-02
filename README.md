# ML Classification of Certified Deposit Bank Customers
The purpose of this project is to compare performance of different machine learning classifiers and build an efficient model to classify bank customers into prospects who would accept opening a CD and those who would not. With this model, the bank can predict the outcome of their marketting campaigns, and focus their resources on customers who are more likely to open a CD.

The data for this project comes from [UCI Machine Learning repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) for a  Portugese banking institution and is a collection of the results of multiple marketing campaigns. 

#Jupyter Notebooks
The accompanied [Jupyter Notebook](data/prompt_III.ipynb) provides the calculations, methodologies, and life-cycle of applying CRISP-DM process to this problen.

# Exploratory Data Analysis and Data Cleaning
The data from this dataset is very clean with no missing values and only 12 duplicates which were dropped accordingly. It has 41188 rows and 21 columns (11 numerical and 10 categorical). The outcome column is labeled "y" and contained "yes" and "no" values which were converted to 0s and 1s. Other categorical columns have been encoded with ***get_dummies()*** function.

![Imbalance Data](images/p3_outcome_heatmap.png)

Data have been split into train and test set by a 70/30. The data is highly imbalanced with 88.7% rejection and 11.3% acceptance. There is also scaling issue with the columns that was taken care of with ***StandardScaler*** class.

Performing a correlation heatmap reveals linear correlations between the outcome variable "y" that is positively depended on ***duration*** and ***previous***, and negatively is impacted by ***nr.employed***, ***pdays***, ***emp.var.rate***, ***euribor3m***, and ***cons.price.idx***.

# Baseline Model
A ***DummyClassifier*** returned 89% accuracy on train and test data, and it is the baseline for this imbalanced dataset. Any classifier has to do better than 89% which is a random classifier.

# Basic Models (first iteration)
Basic ***LogisticRegression*** model returns 91% accuracy on test and train data. 

Comparing it to basic non-hyperparameterized KNN, Decision Tree, SVM, and Random Forest model reveals the following result:

![Classifiers Comparison](images/p3_classifier_compare.png)

High accuracy is only one part of the story, and looking at the confusion matrix reveals that our basic models have low ***Recall*** scores. In this practical business—just like the classic malignant tumor cancer classification problem—our goal is to maximize marketting campaign efforts by signing up more customers for termed CD. If we have a low ***Recall*** score with high ***False Negative*** numbers, it means we are misclassifying some of the prospective customers as insignificant, and no matter how high our accuracy is, we are losing these businesses. So on the next iteration, ***GridSearchCV*** is used to find an optimal model to increase the ***Recal*** score.

![Basic Models Confusion Matrix](images/p3_basic_cm.png)

## Feature Engineering
Basic ***Logistic Regression*** model revealed a good set of linear features that can be used to build the second iteration models. The pictures below show the raw features and their correlations:

![Linear Features](images/p3_linear_features.png)

