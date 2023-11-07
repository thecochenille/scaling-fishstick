# scaling-fishstick

# Project Background
Sparkify (a fictive start-up providing music subscription). The current business model of Sparkify is to provide two levels of subscriptions to their customers (paid and free). Paid subscribers benefit from ad-free music. 


Over the past few months, Sparkify has noticed customer churn for their subscription service with almost 23% of churn. To solve that problem, Sparkify hired a team of data scientists to examine the behavior/usage they collected over time to help the company make marketing decisions and prevent even more customer churn.
In this case, Sparkify leaders have defined that churning customers are users deciding to cancel their paid subscriptions.

## Problem statement
Using data collected on Sparkify customers behavior and usage, which users will likely churn?
Once the users are identified, stakeholders can decide to build a marketing strategy plan to prevent them from churning.

## Technical problem
Which machine learning model will best identify users likely to churn?

## Approach and metrics
To answer this problem, we are going to look at the data collected, and build a machine learning model to predict which user is going to churn. 


### Metrics
Based on the observed churn rate, the dataset is considered as imbalanced. So, we decide to use:
- Precision: the ratio of true positive predictions to all positive predictions made by your model. It focuses on the accuracy of positive predictions. A high precision indicates that the model is good at correctly identifying churned customers without many false positives.

- Recall or sensitivity: the ratio of true positive predictions to all actual positive instances in the dataset. It measures the model's ability to identify all actual churned customers. A high recall indicates that the model effectively captures most of the churned customers.

- F1 score: the harmonic mean of precision and recall. It provides a balance between precision and recall, helping you make trade-offs between false positives and false negatives. This is particularly important in churn prediction, as you want to minimize both missed churn cases and incorrect churn predictions.


### Deployment
Before deploying the full model and train it on the full dataset (12GB), we will explore a subset of XXMB and build our pipeline using Spark.
We will then run our script on AWS EC2 with the full dataset.

## Result Summary


## Decision



## Libraries used 

## Files in the repository:




## Blog write-up 

## Acknowledgements 

