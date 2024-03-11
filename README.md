# Team-Sirius-Project-04

## Team:  
### Luz Aguilar, Scott McNiff, Melanie Runser, Owen Sull, & Jerry Youngblood

## Objective and Requirements
The objective of this project was to apply machine learning (ML) by employing a predictive model to a dataset, arriving at a targeted variable within that dataset at a high degree of accuracy.

Requirements included cleaning, normalizing, and standardizing the data before modeling, then utilizing Spark to retrieve the data and a Python script to initiate, train, and evaluate a model. The model was to demonstrate at least 75% classification accuracy or 0.80 R-squared, with model optimization performed as needed to attain that level of accuracy.

## Data Source and Characteristics
The particular dataset used was one on Lending Club (LC) loan data for all loans issued by LC from 2007 through 2018.

The Database Contents License for the dataset grants the user “a worldwide, royalty-free, non-exclusive, perpetual, irrevocable copyright license” as well as “permission for any information having copyright contained in the [dataset] Contents.” Accordingly, copyright or license considerations were fulfilled, and by virtue of the “id” and “member_id” columns in the dataset being empty and no other individual identifiable information being present, privacy considerations were satisfied as well.
The dataset appeared in csv format ([loan.csv](Data_Source/loan.csv)). Its shape was bounded by  2,260,668 rows across 145 columns, most of which contained float d types but also included integers and objects. Total memory usage was 2.4+ GB.

The dataset was accompanied by a three-tab [data dictionary](Data_Source/LCDataDictionary.xlsx) (xlsx format) in which each tab listed column names and their corresponding definitions.

## Data Processing and Analysis
Extensive review of the data dictionary was done to understand the columns in the dataset and the kind of information contained.
Then, an initial normalizing of the data was performed to see, relative to loan status/default rates, several areas for possible further inquiry. Those areas included grade and sub-grade distribution, loan defaults for 60-month loans vs 36-month ones, home ownership status, loan purpose, loan amounts and installment amounts, income and debt-to-income levels, and interest rates.

After initial modeling was performed, the dataset was further examined and considered for refinement. Analysis revealed a significant number of columns that were less-than-fully populated with values and a few columns that were even empty. Accordingly, those columns were eliminated, leaving only the ones that contained a full set of values. In addition, investigation of the data in terms of years determined  that more than 40% of the dataset was concentrated in the years 2017 and 2018 alone. Considering that as a significant representation for modeling purposes, those years were selected as the ones to draw from, and the dataset was further limited to them. Reshaping of the dataset, then, reduced it to a more manageable one containing 938,821 rows and 50 columns and a memory usage of 365.3+ MB.

The final dataset was accordingly saved to a new [csv file](Data_Source/LC_loans_2017-2018.csv).

## Data Modeling – Approach and Results
A [predictive model](neural_network.ipynb) was initiated using the original dataset and sub-grades as the target, but still with select columns (a dozen or less) as features, considering how their number would expand with use of get_dummies() and consequently exceed memory constraints. Rows with null values were also dropped. Because sub-grades totaled 35, however, they proved to be unwieldy as a target, and grades – which totaled, in contrast, 7 – were used as the target instead.

With the StandardScaler applied, and three neural network layers made using sigmoid and softmax activations, the model was fitted, and epochs run. Those soon revealed accuracy to be only in the 41-42% range, and consequently the dataset and columns selected as features were revisited.

Upon more analysis of the dataset, it was determined that it could be reshaped as described above, reducing its columns by nearly a third and cutting its number of rows by more than half. In addition, it was further observed that loan grade influences interest rate, but interest rate does not affect loan grade.

In light of those observations, another model was initiated with the reshaped dataset and a new target of interest rate instead of grade. A different selection of columns was made, considering the changes to the original dataset and including sub-grade given its granularity compared with grades. In addition, four layers versus three were used for the neural network, repeating usage of sigmoid but exchanging softmax for linear. With the model accordingly fitted, epochs were started and immediately began showing achievement of 98-99% accuracy.
Visualizations were then prepared to demonstrate relationships between features and the target of interest rate.

## Technological Resources
Several tools/libraries and applications were used in this project. Python Pandas, along with Numpy, Matplotlib, and Seaborn were used in Jupyter notebooks for data processing and early analysis. Sci-Kit Learn, TensorFlow, and Keras were used in Google Colab to initiate and execute the neural network models. And Tableau was used for visualizations.

## Usage & Contributing
This project is not open for contributions, as it is a homework assignment only. Please do not copy, modify, or distribute this code without permission. 

## Credits
We utilized multiple resources for clarification, to determine error sources, etc.
*
*
*

## Sources:
https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv
https://www.lendingclub.com/glossary
https://www.informatica.com/resources/articles/what-is-etl.html (ETL diagram image)
https://stackoverflow.com/questions/37532098/split-dataframe-into-two-on-the-basis-of-date
