# pylint: disable=no-member
# pylint: disable=unsubscriptable-object 

# Importing Libraries
"""Import required libraries"""
import pandas as pd
import matplotlib.pyplot as plt
import joblib
# Importing Libraries for Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# Linear Regression Prediction using Ordinary Least Squares Method(OLS)
import statsmodels.api as sm
def lr_model():
    """Function to train the linear regression model."""
    # Importing the dataset (.csv file)
    
    data_frame = pd.read_csv('../Dataset/car data.csv')
     
    # Features of the Dataset
    data_frame.head(10)
    data_frame.info()
    data_frame.shape
    data_frame.drop('Owner', inplace=True, axis=1)
    data_frame.head()
    data_frame['Fuel_Type'].value_counts()
    data_frame['Seller_Type'].value_counts()
    data_frame['Transmission'].value_counts()
    # Data Visualization
    data_frame.hist(bins = 50, figsize = (15, 10))
    # Handling Categorical Variables
    data_frame.replace({'Fuel_Type' : {'Petrol' : 0, 'Diesel' : 1, 'CNG': 2}}, inplace = True)
    data_frame.replace({'Seller_Type' : {'Dealer' : 0, 'Individual' : 1}}, inplace = True)
    data_frame.replace({'Transmission' : {'Manual' : 0, 'Automatic' : 1}}, inplace = True)
    data_frame.head()
    x_value = data_frame.drop(['Car_Name', 'Selling_Price'], axis = 1)
    y_value = data_frame['Selling_Price']
    # Splitting data into training and testing set
    x_train, x_valid, y_train, y_valid = train_test_split(x_value, y_value, test_size = 0.25, random_state = 99) # pylint: disable=line-too-long
    training_lr_model = LinearRegression()
    training_lr_model.fit(x_train, y_train)
    joblib.dump(training_lr_model, 'trained_reg_model.sav')
    # Training the regression model
    model_pred = training_lr_model.predict(x_valid)
    ## Computing the R-Squared Error
    error_score_valid = metrics.r2_score(y_valid, training_lr_model.predict(x_valid))
    error_score_train = metrics.r2_score(y_train, training_lr_model.predict(x_train))
    print(error_score_train, error_score_valid)
    # Plot of Real Price Vs Predicted Price of Car Sales
    plt.scatter(y_valid, model_pred)
    plt.xlabel("Real Price")
    plt.ylabel("Predicted Price")
    plt.title("Real Vs Predicted Car Price")
    plt.show()
    plt.scatter(y_valid, model_pred, label = 'Valid')
    plt.scatter(y_train, training_lr_model .predict(x_train), label = 'Train')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual Vs Predicted Car Price")
    plt.show()
    model = sm.OLS(y_valid, model_pred).fit()
    model.predict(model_pred)
    model_details = model.summary()
    print(model_details)
if __name__ == '__main__':
    lr_model()
    