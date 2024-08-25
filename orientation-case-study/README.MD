## Setup Your Environment

To ensure all package files are stored separately for the project, create a virtual environment. This helps keep dependencies isolated and avoids conflicts between different projects.

### Creating a Virtual Environment

To create a virtual environment, use the following command (on Windows, you may need to omit the dot "." from the name):

```bash
python -m venv .venv
```

## Activating the Virtual Environment

After creating the virtual environment, activate it using the source command. On Windows, use venv\Scripts\activate.bat. Note that VS Code may automatically detect and activate the virtual environment for you.

For Unix or MacOS:

```bash
source .venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate.bat
```

## Installing Dependencies
Once the virtual environment is activated, install the necessary dependencies:

```bash
pip install pandas  # Required for the group approach
pip install scikit-learn  # Required for the regression approach
pip install matplotlib seaborn # for example 3 - new customer 
``` 

## Running the Project
## 1. Testing the Group Approach
If you have a robust dataset, significant domain knowledge, or limited computational resources, the group approach might be the best fit.

"A great dataset with a simple approach is always better than any sophisticated algorithms with poor data."


To test the group approach, run the following command:

```bash
python approach_groups.py
```

```yaml
Group default: 6445 (21.48%)
Group delayed: 5880 (19.60%)
Group duly: 1608 (5.36%)
```

## 2. Testing the Regression Approach
To evaluate the regression approach, use the following command:
```bash
python approach_regression.py
```

```yaml
              precision    recall  f1-score   support

         0.0       0.84      0.94      0.89      4518
         1.0       0.67      0.39      0.49      1310

    accuracy                           0.82      5828
   macro avg       0.75      0.67      0.69      5828
weighted avg       0.80      0.82      0.80      5828
```

This model performs well in predicting non-defaulters (class 0.0) with high precision and recall. However, it struggles to identify defaulters (class 1.0), as evidenced by the low recall (0.39) and F1-score (0.49). Although the overall accuracy is 82%, improvements may be needed, such as class balancing or hyperparameter tuning, to better detect defaulters.

## 3. New customer risk score per variable: gender, age and marital stage
```bash
python approach_regression.py
```

