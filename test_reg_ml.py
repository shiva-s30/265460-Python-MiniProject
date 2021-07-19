import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score


data_frame = pd.read_csv('/Dataset/car data.csv')
test_frame = data_frame

def test_instance_of_dataframe():
    assert isinstance(test_frame, pd.DataFrame)
    
def test_class_type():
    assert(type(test_frame).__name__ == 'DataFrame')
    
def test_data():
    pd.testing.assert_frame_equal(data_frame, test_frame)

def test_value():
    # val1 = data_frame.at[5, 'Kms_Driven']
    # assert val1 == 2071
    test_col_values = test_frame['Kms_Driven'].tolist()
    assert  min(test_col_values) == 500
    assert  max(test_col_values) == 500000
    
def test_error_value():
    selling_price_values = test_frame['Selling_Price'].tolist()
    present_price_values = test_frame['Present_Price'].tolist()
    r2_error = r2_score(selling_price_values, present_price_values)
    assert r2_error <= 0
    


