# Prediction of Car Sales using Regression Algorithm
The Project aims at predicting the price of used cars by building a Machine Learning model using Linear Regression algorithm.

## About the Dataset:
The dataset contains around 300 entries made during the period 1992 - 2020.  The various attributes present in the dataset are:

1.  Car_Name
2.  Year
3.  Selling_Price
4.  Present_Price
5.  Kms_Driven
6.  Fuel_Type
7.  Seller_Type
8.  Transmission
9.  Owner

|Attribute| Description |
|--|--|
|Car_Name  | Contains the name/model of the car sold |
|Year| Contains the year in which the car was bought |
|Selling_Price| Contains the price the owner wished to sell the car |
|Present_Price| Contains the current ex-showroom price of the car |
|Kms_Driven| Contains the distance completed by the car in kilometers |
|Fuel_Type| Contains the Fuel type of the car |
|Seller_Type| Contains the type of seller such as a dealer or an individual |
|Transmission| Contains the type of transmission such as the car is manual or automatic |
|Owner|Contains the number of previous owners of the car |

  ## Project Implementation:

 - The required python libraries for data analysis and plots such as numpy, pandas, matplotlib are imported. 
 - The dataset is imported using the pandas library and stored into a dataframe.
 - All the features of the data are extracted and analysed.
 - This data can now be visualized using bar plots, pie charts etc. 
 - The next step is to perform data cleansing. It is done by converting any categorical variables of string type into int values.
 - The required libraries for linear regression are imported to train the model.
 - The dataset now needs to be split into training and testing set, so that once the model is trained, it's functionality can be verified using the test data. 
 - In the project, the linear regression is computed using two methods. They are:	
			 
	 - By computing the R-Squared Error Value
	 - Using the Ordinary Least Squares Method(OLS)
 - The project is implemented using the `conda` `virtualenv` 4.10.3 running on python 3.8.10
 
