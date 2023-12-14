# import libraries
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    # function to create the spark session
    # create a Spark session
    spark = SparkSession \
        .builder \
        .appName("Sparkify - data preparation") \
        .getOrCreate()

    spark.sparkContext.addFile("https://udacity-dsnd.s3.amazonaws.com/sparkify/mini_sparkify_event_data.json")

    file_path = SparkFiles.get("mini_sparkify_event_data.json")
    df = spark.read.json(file_path)
    
    print('Here is the data schema')
    print(df.printSchema())
    print('Here is the first row of the dataset')
    print(df.take(1))

