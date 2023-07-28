import os
import findspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pandas as pd    
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/"
os.environ["SPARK_HOME"] = "/workspace/tripx/MCS/big_data/spark-3.1.1-bin-hadoop3.2"
findspark.init()
import time 
import logging

logging.basicConfig(filename='logg.log', filemode='w', level=logging.INFO)
logger = logging.getLogger()

def read_h5_data(spark_session, data_path):
    start_time = time.time()
    data = pd.read_hdf(data_path) 
    print(data.head())
    data=spark_session.createDataFrame(data) 
    logger.info(f"Execution time h5 by pandas: {time.time() - start_time}")
    return data

def read_json(spark_session, data_path):
    start_time = time.time()
    data=spark_session.read.json(data_path)
    logger.info(f"Execution time of read json: {time.time() - start_time}")
    return data 

def overview_data(df_data):
    logger.info(f'Data overview: {df_data.printSchema()}')
    print(df_data.take(3))

def main():
    # Stop spark if it existed.
    try:
        sc.stop()
    except:
        print('sc have not yet created!')
    sc = SparkContext(master = "local", appName = "Multimodal Single Cell")
    # sc = SparkContext.getOrCreate()
    # Init session
    logger.info("Test logger")
    print("Init session")
    my_spark = SparkSession.builder.getOrCreate()
    print(my_spark.catalog.listTables())

    # data_path='/dataset/NeurIPS2022/train_cite_inputs.h5'
    json_path='/workspace/tripx/MCS/big_data/train_cite_targets.json'
    h5_path='/dataset/NeurIPS2022/train_cite_targets.h5'
    df_data = read_h5_data(my_spark, h5_path)
    overview_data(df_data)

    df_data = read_json(my_spark, json_path)
    overview_data(df_data)
    # # sparkDF.show()
    # # Create a temporary table on catalog of local data frame flights as new temporary table flights_temp on catalog
    # df_data.createOrReplaceTempView('single_cell_temp')
    # # check list all table available on catalog
    # print(my_spark.catalog.listTables())
if __name__ == '__main__':
    main()

########################## Analysis ################################

# Number record 

# Number feature

# Max cell is non-zero value

# Average cell contains non-zero value

# The average non-zero value 

# The standard deviation of our features

# The average standard deviation 

########################## Analysis ################################

