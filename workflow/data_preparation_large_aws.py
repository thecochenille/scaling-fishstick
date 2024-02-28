"""
File: data_preparation.py
Author: Isabelle Vea
Date: 02/28/20123 (last edits)

Description: This scripts takes a json dataset sparkify_event_data.json (a large dataset) that 
contains usage records of a fictional music platform Sparkify and prepares a new dataset to 
perform churn prediction analysis. The output file is saved in a json file 
to be processed in the model_training_large_aws.py for model training.

NB: The dataset used in this script needs to be processed in a cluster and uses spark to do so.
"""



# LIBRARIES
import pyspark
from pyspark import SparkConf
from pyspark import SparkFiles
from pyspark.sql import SparkSession

from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import TimestampType


from pyspark.sql.functions import isnan, count, when, col, desc, udf, col, sort_array, asc, desc, countDistinct, first
from pyspark.sql.functions import udf
from pyspark.sql.functions import to_date, year, month, dayofmonth, dayofweek, hour, date_format, substring,datediff
from pyspark.sql.functions import sum as spark_sum, avg as spark_avg

import datetime



#FUNCTIONS
def remove_missing_userid(df):
    ''' This function takes a dataframe and filters out all rows where column UserId is empty
    Input = Spark dataframe with column UserId included
    Output = Spark dataframe
    '''
    condition = df.userId != ''
    df= df.filter(condition)
    
    return df


def create_churn_label(df):
    ''' This function takes a dataframe and creates churn labels based on the 
    churn definition: Cancellation Confirmation value in 'pages' is considered churn.
    
    Input = Spark dataframe
    
    Output = Spark dataframe with new column 'churn_lab' for each user who churned
    '''
    #labeling all 'pages' with Cancellation Confirmation
    create_churn = udf(lambda x: 1 if x =="Cancellation Confirmation" else 0, IntegerType())
    df = df.withColumn('churn_lab_temp', create_churn('page'))
    
    #extract a list of churn users from the new df
    churn_userid = df.select('userId').where(col('churn_lab_temp')==1).groupby('userId').count()
    churn_userid_list = [row['userId'] for row in churn_userid.collect()]
    
    #new column based on the churn list
    df = df.withColumn('churn_lab', when((df.userId).isin(churn_userid_list), 1).otherwise(0))
    
    #dropping column
    df = df.drop('churn_lab_temp')
    
    return df   


def create_features(df):
    '''This function takes a clean spark dataframe of sparkify records and 
    creates new features for each userId based on different aggregations, then combine all the new feature 
    into a final df called user_df. This user_df is saved into a csv file
  
    Input: df - spark dataframe
    Output: user_df - spark dataframe including one row per userId and new features'''
    
    
    #new features and aggregation by feature
   #allows to aggregate many features in one step
    agg_df1 = df.groupBy('userId').agg(
                             first('gender').alias('gender'), #takes the first occurence of gender for each user
                             countDistinct('artist').alias('unique_artist_count'), #counting the unique artist per userId
                             first('state1').alias('state'), #retrieving state information for each userId
                             spark_sum('length').alias('total_session_length'),
                             spark_avg('length').alias('avg_session_length'),
                             first('churn_lab').alias('churn_label'),
                             countDistinct('song').alias('unique_song_count'),
                             spark_sum('itemInSession').alias('total_items'),
                             spark_avg('itemInSession').alias('avg_items')
       
       
       
   )
   #aggregate by month
    agg_df2 = df.groupBy('userId','month').agg(
                              spark_sum("length").alias("session_length_per_month"),
                              spark_sum('itemInSession').alias('items_per_month'),
                              count('*').alias('count_sessions_per_month')
    )
    
    agg_df2 = agg_df2.groupBy('userId').agg(
                              spark_avg('session_length_per_month').alias('avg_session_length_per_month'),
                              spark_avg('items_per_month').alias('avg_items_per_month'),
                              spark_avg('count_sessions_per_month').alias('avg_count_session_per_month')
    )
    #aggregate by day
    agg_df3 = df.groupBy('userId','ts_todate').agg(
                              spark_sum("length").alias("session_length_per_day"),
                              spark_sum('itemInSession').alias('items_per_day'),
                              count('*').alias('count_sessions_per_day')
    )
    agg_df3 = agg_df3.groupBy('userId').agg(
                              spark_avg('session_length_per_day').alias('avg_session_length_per_day'),
                              spark_avg('items_per_day').alias('avg_items_per_day'),
                              spark_avg('count_sessions_per_day').alias('avg_count_session_per_day')
    )
   
    # aggregate counts per page per user
    page_counts_per_user = df.groupBy('userId', 'page').count()
    agg_df4 = page_counts_per_user.groupBy('userId').pivot('page').sum('count')
    agg_df4 = agg_df4.na.fill(0)



    
    #join all agg_df
    
    user_df = agg_df1.join(agg_df2,['userid']) 
    user_df = user_df.join(agg_df3,['userid'])
    user_df = user_df.join(agg_df4,['userid'])

    return user_df



if __name__ == "__main__":
    # create a Spark session
    spark = SparkSession \
        .builder \
        .appName("Sparkify - data preparation 12GB dataset") \
        .getOrCreate()
    
    print("Retrieving the 12 GB dataset from S3 bucket and loading them in the spark session")
    event_data = "s3://sparkifylargedataset/sparkify_event_data.json"
    df = spark.read.json(event_data)
    
    print('Here is the dataset schema')
    print(df.printSchema())

    print('Removing rows with missing UserId...')
    df = remove_missing_userid(df)
    
    #feature drop
    print('Dropping unnecessary columns...')
    cols = ('firstname','lastname','method','auth','level','sessionId','status','userAgent')
    df = df.drop(*cols)


    print('Reformating dates and location data...')
    #change format dates
    cols = ("ts","registration","ts_ts","registration_ts")
    
    df = df.withColumn('ts_ts', (col('ts') / 1000.0).cast(TimestampType())) \
      .withColumn('ts_todate', to_date('ts_ts')) \
      .withColumn('registration_ts', (col('registration') / 1000.0).cast(TimestampType())) \
      .withColumn('registration_todate', to_date('registration_ts')) \
      .drop(*cols)
    df = df.withColumn('month', month('ts_todate'))

    
    #prepare location data
    split_col_location = pyspark.sql.functions.split(df['location'], ', ')
    df = df.withColumn('state', split_col_location.getItem(1))\
    
    split_col_state = pyspark.sql.functions.split(df['state'], '-')
    df = df.withColumn('state1', split_col_state.getItem(0))

    print('Creating churn labels based on Cancellation confirmation for each user...')
    df = create_churn_label(df)
    
    
    #create new features
    print('Creating new features ...')
    user_df = create_features(df)
    
    
    print('Saving our new dataset ...')
    path = "s3://sparkifylargedataset/user_data_12GB.jso"
    user_df.write.json(path = path, mode='overwrite')

    print(f'Large User Dataset prepared sucessfully and saved to the following S3 URI: {path}.')