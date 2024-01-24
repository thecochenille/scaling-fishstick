"""
File: model_training_mini.py
Author: Isabelle Vea
Date: 12/14/20123 (last edits)

Description: This scripts uses the csv file output from data_preparation_mini.py,
split and performs last transformations, scaling to train three main models. 
This scripts should provide evaluation reports for each model trained and 
save the best model to be used.

"""


# load saved prepared dataset (csv file)
# last data engineering steps
# building pipeline to train the training set
# training models
# create evaluation reports

# save the best model


#libraries
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

print('saving test set')
#saving test set to be used for the app
test_df.write.parquet("test_data", mode="overwrite")


#preprocessing functions
#encoders function
def create_encoders():
    gender_encoder = OneHotEncoder(inputCol="gender_index", outputCol="gender_encoded")
    state_encoder = OneHotEncoder(inputCol="state_index", outputCol="state_encoded")
    return [gender_encoder, state_encoder]


#feature assembler function
def create_feature_assembler():
    cols_for_assembler = [ 'unique_artist_count', 
       'total_session_length', 'avg_session_length',
       'unique_song_count', 'total_items', 'avg_items',
       'avg_session_length_per_month', 'avg_items_per_month',
       'avg_count_session_per_month', 'avg_session_length_per_day',
       'avg_items_per_day', 'avg_count_session_per_day', 'About', 'Add_Friend',
       'Add_to_Playlist', 'Cancel', 'Cancellation_Confirmation', 'Downgrade',
       'Error', 'Help', 'Home', 'Logout', 'NextSong', 'Roll_Advert',
       'Save_Settings', 'Settings', 'Submit_Downgrade', 'Submit_Upgrade',
       'Thumbs_Down', 'Thumbs_Up', 'Upgrade',
       'state_encoded', 'gender_encoded']

    assembler = VectorAssembler(inputCols=cols_for_assembler, outputCol="features")
    return assembler


#scaler function
def create_scaler():
    scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=True)
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

# preprocessing steps
encoders = create_encoders()
assembler = create_feature_assembler()
scaler = create_scaler()
label_indexer = create_label_indexer()



#defining my models
rf_model = RandomForestClassifier(featuresCol='scaled_features', labelCol='label')
lr_model = LogisticRegression(featuresCol='scaled_features', labelCol='label')
gbt_model = GBTClassifier(featuresCol='scaled_features', labelCol='label')

#pipeline for each model
rf_pipeline = Pipeline(stages=[*encoders, assembler, scaler, label_indexer, rf_model])
lr_pipeline = Pipeline(stages=[*encoders, assembler, scaler, label_indexer, lr_model])
gbt_pipeline = Pipeline(stages=[*encoders, assembler, scaler, label_indexer, gbt_model])

#train and evaluate each model 
rf_trained_model, rf_auc, rf_predictions = train_and_evaluate_model(rf_pipeline, train_df, validation_df)
lr_trained_model, lr_auc, lr_predictions = train_and_evaluate_model(lr_pipeline, train_df, validation_df)
gbt_trained_model, gbt_auc, gbt_predictions = train_and_evaluate_model(gbt_pipeline, train_df, validation_df)

print("Random Forest AUC:", rf_auc)
print("Logistic Regression AUC:", lr_auc)
print("GBT Classifier AUC:", gbt_auc)

models = [(rf_auc, rf_trained_model), (lr_auc, lr_trained_model), (gbt_auc, gbt_trained_model)]
best_auc, best_model = max(models, key=lambda x: x[0])

if all(auc == best_auc for auc, _ in models):
    print("All models have the same AUC. Saving the Random Forest model.")
    best_model = rf_trained_model

best_model.save("best_model_mini")