# Importing Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def lr_model():
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
    
    histograms = data_frame.hist(bins = 50, figsize = (15, 10))
    
    # Handling Categorical Variables
    
    data_frame.replace({'Fuel_Type' : {'Petrol' : 0, 'Diesel' : 1, 'CNG': 2}}, inplace = True)
    data_frame.replace({'Seller_Type' : {'Dealer' : 0, 'Individual' : 1}}, inplace = True)
    data_frame.replace({'Transmission' : {'Manual' : 0, 'Automatic' : 1}}, inplace = True)
    
    data_frame.head()
    
    # Importing Libraries for Linear Regression
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    
    X = data_frame.drop(['Car_Name', 'Selling_Price'], axis = 1) 
    Y = data_frame['Selling_Price']
    
    # Splitting data into training and testing set
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.25, random_state = 99)
    Training_Lr_model = LinearRegression()
    Training_Lr_model.fit(X_train, Y_train)
    
    joblib.dump(Training_Lr_model, 'trained_reg_model.sav')
    
    # Training the regression model
    
    model_pred = Training_Lr_model.predict(X_valid)
    
    ## Computing the R-Squared Error
    
    error_score_valid = metrics.r2_score(Y_valid, Training_Lr_model.predict(X_valid))
    error_score_train = metrics.r2_score(Y_train, Training_Lr_model.predict(X_train))
    print(error_score_train, error_score_valid)
    
    
    # Plot of Real Price Vs Predicted Price of Car Sales
    
    plt.scatter(Y_valid, model_pred)
    plt.xlabel("Real Price")
    plt.ylabel("Predicted Price")
    plt.title("Real Vs Predicted Car Price")
    plt.show()
    
    plt.scatter(Y_valid, model_pred, label = 'Valid')
    plt.scatter(Y_train, Training_Lr_model .predict(X_train), label = 'Train')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual Vs Predicted Car Price")
    plt.show()
    
    # Linear Regression Prediction using Ordinary Least Squares Method(OLS)
    
    import statsmodels.api as sm
    model = sm.OLS(Y_valid, model_pred).fit()
    model_prediction = model.predict(model_pred) 
    
    model_details = model.summary()
    print(model_details)

    # from sklearn.metrics import r2_score
    # selling_price_values = data_frame['Selling_Price'].tolist()
    # present_price_values = data_frame['Present_Price'].tolist()
    # r2_error = r2_score(selling_price_values, present_price_values)
    # print(r2_error)


if __name__ == '__main__':
    lr_model()

