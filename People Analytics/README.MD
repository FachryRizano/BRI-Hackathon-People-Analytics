# People Analytics

Machine learning allows decision makers in the HR department to predict employee performance in the next 1 year using historical KPI data owned by the company. In the People Analytics category, participants are asked to build a machine learning model that can accurately predict employee performance. 

## Task
Using existing data, participants are asked to predict which employees will be included in the best performance category and who will not.

## Evaluation Metric
The evaluation metric that will be used is AUC, where the True Positive Rate value will be compared with the False Positive Rate value at different thresholds.

\ [True Positive Rate = \ frac {TP} {TP + FN} \]

\ [False Positive Rate = \ frac {FP} {FP + TN} \]

##  Submission
Submission of this task contains 6000 rows with 2 index columns and Best Performance, where the index will be adjusted to the index data in test.csv, and Best Performance is the prediction value in the float.
The following is an illustration of the requested submission (an example of a submission can be seen in the sample_submission.csv file)
