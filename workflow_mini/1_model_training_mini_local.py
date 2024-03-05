"""
File: model_training_mini.py
Author: Isabelle Vea
Date: 12/14/20123 (last edits)

Description: This scripts uses the csv file output from data_preparation_mini.py,
split and performs last transformations, scaling to train three main models. 
This scripts should provide evaluation reports for each model trained and 
save the best model to be used.

"""

# import libraries
import pyspark
from pyspark import SparkConf
from pyspark import SparkFiles

from pyspark.sql import SparkSession

from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import TimestampType


from pyspark.sql.functions import isnan, count, when, col, desc, udf, col, sort_array, asc, desc, countDistinct, first
from pyspark.sql.functions import udf, trim,expr
from pyspark.sql.functions import to_date, year, month, dayofmonth, dayofweek, hour, date_format, substring,datediff
from pyspark.sql.functions import sum as spark_sum, avg as spark_avg

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler, StandardScaler
from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


#creating spark session
spark = SparkSession \
    .builder \
    .appName("Training models") \
    .getOrCreate()


# FUNCTIONS
def create_encoders():
    gender_encoder = OneHotEncoder(inputCol="gender_index",
                                   outputCol="gender_encoded")
    state_encoder = OneHotEncoder(inputCol="state_index",
                                  outputCol="state_encoded")
    return [gender_encoder, state_encoder]


#feature assembler function
def create_feature_assembler():
    cols_for_assembler = [
        'unique_artist_count', 'total_session_length', 'avg_session_length',
        'unique_song_count', 'total_items', 'avg_items',
        'avg_session_length_per_month', 'avg_items_per_month',
        'avg_count_session_per_month', 'avg_session_length_per_day',
        'avg_items_per_day', 'avg_count_session_per_day', 'About', 'Add_Friend',
        'Add_to_Playlist', 'Downgrade', 'Error', 'Help', 'Home', 'Logout',
        'NextSong', 'Roll_Advert', 'Save_Settings', 'Settings',
        'Submit_Downgrade', 'Submit_Upgrade', 'Thumbs_Down', 'Thumbs_Up',
        'Upgrade', 'state_encoded', 'gender_encoded'
    ]

    assembler = VectorAssembler(inputCols=cols_for_assembler,
                                outputCol="features")
    return assembler


#scaler function
def create_scaler():
    scaler = StandardScaler(inputCol='features',
                            outputCol='scaled_features',
                            withStd=True,
                            withMean=True)
    return scaler


#label indexer function
def create_label_indexer():
    label_indexer = StringIndexer(inputCol='churn_label', outputCol='label')
    return label_indexer



#function that trains and evaluate a model
def train_and_evaluate_model(model, train_data, validation_data):
    
    trained_model = model.fit(train_data)
    predictions = trained_model.transform(validation_data)
    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.evaluate(predictions)
    
    return trained_model, auc, predictions


#function that trains and evaluate a model
def train_and_evaluate_model(model, train_data, validation_data):

    trained_model = model.fit(train_data)
    predictions = trained_model.transform(validation_data)

    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.evaluate(predictions)

    evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = evaluator_acc.evaluate(predictions)

    return trained_model, auc, accuracy, predictions


def evaluate_models(models, model_names, validation_data):
    metrics_list = []

    best_model_name = None
    best_f1_score = 0.0

    # Loop through each model
    for i, model in enumerate(models):
        # Make predictions on the validation set
        predictions = model.transform(validation_data)

        # Instantiate MulticlassClassificationEvaluator
        evaluator_multiclass = MulticlassClassificationEvaluator()

        # Evaluate metrics
        f1_score = evaluator_multiclass.evaluate(predictions, {evaluator_multiclass.metricName: "f1"})
        accuracy = evaluator_multiclass.evaluate(predictions, {evaluator_multiclass.metricName: "accuracy"})
        precision = evaluator_multiclass.evaluate(predictions, {evaluator_multiclass.metricName: "weightedPrecision"})
        recall = evaluator_multiclass.evaluate(predictions, {evaluator_multiclass.metricName: "weightedRecall"})

        # Adding area under ROC
        evaluator = BinaryClassificationEvaluator()
        auc = evaluator.evaluate(predictions)

        # Appending metrics to list
        metrics_list.append(
            (f"Model {model_names[i]}", f1_score, auc, accuracy, precision, recall))

        # Check if the current model has a better F1 score than the best one so far
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_model_name = model_names[i]
            best_model = models[i]

    # Create a DataFrame from the list of metrics
    metrics_df = spark.createDataFrame(metrics_list, [
        "Model",
        "F1 Score",
        "Area under ROC",
        "Accuracy",
        "Precision",
        "Recall",
    ])

    print(f"The best model based on F1 score is: {best_model_name} with F1 score {best_f1_score}")

    return metrics_df, best_model_name, best_model




#opening the prepared dataset exported as csv
path = "user_data.csv" #prepared dataset to use for training
df = spark.read.option("header", "true").option("inferSchema", "true").csv(path)


for column_name in df.columns: #replaces space in column names into _
    new_column_name = column_name.replace(" ", "_")
    df = df.withColumnRenamed(column_name, new_column_name)


#dataframe dimension
row = df.count() # extracting number of rows 
col = len(df.columns) # extracting number of columns 
 
# printing
print(f'Dimension of the Dataframe is: {(row,col)}')
print(f'Number of Rows are: {row}')
print(f'Number of Columns are: {col}')

print(df.printSchema()) #printing our all column names

#indexing categorical values
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")
state_indexer = StringIndexer(inputCol="state", outputCol="state_index")

print('indexing categorical data')
indexed_data = gender_indexer.fit(df).transform(df)
indexed_data = state_indexer.fit(indexed_data).transform(indexed_data)

print('creating train, validation and test sets')

train_df, remaining_df = indexed_data.randomSplit([0.7, 0.3], seed=42)
# The second split: 50% of the remaining data for validation, 50% for testing
validation_df, test_df = remaining_df.randomSplit([0.5, 0.5], seed=42)



# creating a pipeline including the preprocessing steps
encoders = create_encoders()
assembler = create_feature_assembler()
scaler = create_scaler()
label_indexer = create_label_indexer()
transform_pipeline = Pipeline(
    stages=[*encoders, assembler, scaler, label_indexer])

#fitting transform pipeline
fitted_transform_pipeline = transform_pipeline.fit(train_df)

#preprocessing the train and validation sets separately to avoid leak
train_transformed_data = fitted_transform_pipeline.transform(train_df)
validation_transformed_data = fitted_transform_pipeline.transform(validation_df)
test_transformed_data = fitted_transform_pipeline.transform(test_df)

#save transformed data
train_transformed_data.write.json(path = "mini_train_transformed.json", mode='overwrite')
validation_transformed_data.write.json(path = "mini_validation_transformed.json", mode='overwrite')
test_transformed_data.write.json(path = "mini_test_transformed.json", mode='overwrite')

#defining my models
rf_model = RandomForestClassifier(featuresCol='scaled_features',labelCol='label')
lr_model = LogisticRegression(featuresCol='scaled_features', labelCol='label')
gbt_model = GBTClassifier(featuresCol='scaled_features', labelCol='label')



#training and evaluating based on the function created above
rf_trained_model, rf_auc, rf_accuracy, rf_predictions = train_and_evaluate_model(
    rf_model, train_transformed_data, validation_transformed_data)
lr_trained_model, lr_auc, lr_accuracy, lr_predictions = train_and_evaluate_model(
    lr_model, train_transformed_data, validation_transformed_data)
gbt_trained_model, gbt_auc, gbt_accuracy, gbt_predictions = train_and_evaluate_model(
    gbt_model, train_transformed_data, validation_transformed_data)

#evaluating the three trained models
models = [rf_trained_model, lr_trained_model, gbt_trained_model]
model_names = ["Random Forest", "Logistic Regression", "Gradient Boosted Trees"]

metrics_df, best_model_name, best_model = evaluate_models(models, model_names, validation_transformed_data)

print("This is a table comparing all model metrics")
metrics_df.show()


#saving the best model
best_model.write().overwrite().save("best_model_to_tune")

