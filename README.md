# ML Classification of Certified Deposit Bank Customers
The purpose of this project is to compare performance of different machine learning classifiers and build an efficient model to classify bank customers into prospects who would accept opening a CD and those who would not. With this model, the bank can predict the outcome of their marketting campaigns, and focus their resources on customers who are more likely to open a CD.

The data for this project comes from [UCI Machine Learning repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) for a  Portugese banking institution and is a collection of the results of multiple marketing campaigns. 

#Jupyter Notebooks
The accompanied [Jupyter Notebook](data/prompt_III.ipynb) provides the calculations, methodologies, and life-cycle of applying CRISP-DM process to this problen.

# Exploratory Data Analysis and Data Cleaning
The data from this dataset is very clean with no missing values and only 12 duplicates which were dropped accordingly. It has 41188 rows and 21 columns (11 numerical and 10 categorical). The outcome column is labeled "y" and contained "yes" and "no" values which were converted to 0s and 1s. Other categorical columns have been encoded with ***get_dummies()*** function.

[Imbalance Data](images/p3_outcome.png)

Data have been split into train and test set by a 70/30. The data is highly imbalanced with 88.7% rejection and 11.3% acceptance. There is also scaling issue with the columns that was taken care of with ***StandardScaler*** class.

[Correlation Heatmap](images/p3_heatmap.png)
