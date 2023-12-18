# load saved prepared dataset
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


path = "user_data.csv"

df = spark.read.option("header", "true").option("inferSchema", "true").csv(path)

#dataframe dimension
# extracting number of rows from the Dataframe
row = df.count()
   
# extracting number of columns from the Dataframe
col = len(df.columns)
 
# printing
print(f'Dimension of the Dataframe is: {(row,col)}')
print(f'Number of Rows are: {row}')
print(f'Number of Columns are: {col}')

print(df.printSchema())

#indexing categorical values
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")
state_indexer = StringIndexer(inputCol="state", outputCol="state_index")

print('indexing categorical data')
indexed_data = gender_indexer.fit(df).transform(df)
indexed_data = state_indexer.fit(indexed_data).transform(indexed_data)

print('creating train and test sets')
train_df, test_df = indexed_data.randomSplit(weights=[0.8,0.2], seed=200)

print(train_df)
print('oversampling train set')
train_oversampled = train_df.union(train_df.filter(col("churn_label") == 1))


#encoders
print('creating encoders')
gender_encoder = OneHotEncoder(inputCol="gender_index", outputCol="gender_encoded")
state_encoder = OneHotEncoder(inputCol="state_index", outputCol="state_encoded")


#assembler
print('creating feature assembler')
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

#scaler
print('scaling features')
scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=True)

#label indexer
print('creating label')
label_indexer = StringIndexer(inputCol='churn_label', outputCol='label')

# List of models to evaluate
print('creating the three models')
models = [
    LogisticRegression(featuresCol='scaled_features', labelCol='label'),
    RandomForestClassifier(featuresCol='scaled_features', labelCol='label'),
    GBTClassifier(featuresCol='scaled_features', labelCol='label')
]

# Dictionary to store the best model and its metric value
best_model_info = {
    'model': None,
    'metric_value': float('-inf')  # Initialize with a very small value for AUC
}

num_rows = train_oversampled.count()
if num_rows < 1000:
    num_folds = 5
elif 1000 <= num_rows < 10000:
    num_folds = 8
else:
    num_folds = 10
    
for model in models:
    # Assign names to each model
    if isinstance(model, RandomForestClassifier):
        model_name = "Random Forest"
    elif isinstance(model, LogisticRegression):
        model_name = "Logistic Regression"
    elif isinstance(model, GradientBoostedTreeClassifier):
        model_name = "Gradient Boosted Tree"
    else:
        # Handle other model types if needed
        model_name = "UnknownModel"
        
    # Assuming 'model_name' is a string representing the model name (e.g., "RandomForest")
    print(f"Evaluating {model_name}...")

    # Assuming 'param_grid' is a ParamGrid for model hyperparameter tuning
    param_grid = ParamGridBuilder().build()

    # Assuming 'pipeline' is your feature engineering and model pipeline
    pipeline = Pipeline(stages=[gender_encoder, state_encoder, assembler, scaler, label_indexer, model])

    # Assuming 'evaluator' is your BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')

    # Assuming 'num_folds' is the number of cross-validation folds
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=num_folds)

    # Fit the cross-validator
    cv_model = cv.fit(train_oversampled)

    #making predictions
    predictions = cv_model.transform(test)

    # evaluate the model
    metric_value = evaluator.evaluate(predictions)
    print(f"{evaluator_metric} on test set: {metric_value}")

    # compare and update the best model information
    if metric_value > best_model_info['metric_value']:
        best_model_info['model'] = cv_model
        best_model_info['metric_value'] = metric_value

# save the best model to a pickle file
best_model = best_model_info['model']
best_model_path = "best_model.pkl"
with open(best_model_path, 'wb') as file:
    pickle.dump(best_model, file)

print(f"Best model is {cv_model} and was saved to: {best_model_path}")