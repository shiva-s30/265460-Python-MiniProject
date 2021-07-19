 - This directory contains the pycache and python test files.
 - The tests performed on the `regression_ml_pyt.py` is perfomed using pytest.
 - The pytest version used is `pytest 6.2.4`. 


The required tests are written in `test_reg_ml.py` and it contains the following test methods:

|Test Method | Description  |
|--|--|
| `test_instance_of_dataframe()` | It checks if the object passed is a pandas dataframe.  |
| `test_class_type()`| It checks for the type of the test dataframe.|
| `test_data()`| It checks for the equality condition of the original dataframe and test data frame.|
|  `test_value()`| It used to check the minimum and maximum values of a column in the dataframe|
| `test_error_value()`| It checks for R-Squared Error value to ensure it is very minimum. |

Therefore, there are five test cases defined for the python src file and all five cases have passed successfully.

![Screenshot (222)](https://user-images.githubusercontent.com/65439506/126147308-8d47d506-2b93-4860-9c7b-56f5a15d2ccb.png)

