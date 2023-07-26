import os
import findspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/"
os.environ["SPARK_HOME"] = "/workspace/tripx/MCS/big_data/spark-3.1.1-bin-hadoop3.2"
findspark.init()
import pandas as pd    


def read_h5_data(spark_session, data_path):
    data = pd.read_hdf(data_path) 
    print(data.head())
    data=spark_session.createDataFrame(data) 
    # print(data.take(5))
    # print(data.printSchema())
    # flights.printSchema()
    return data

def overview_data(df_data):
    print('Data overview')
    df_data.printSchema()
    print('Columns overview')
    pd.DataFrame(df_data.dtypes, columns = ['Column Name','Data type'])
    
def main():
    # Stop spark if it existed.
    try:
        sc.stop()
    except:
        print('sc have not yet created!')
    sc = SparkContext(master = "local", appName = "Multimodal Single Cell")
    print(sc.version) 
    # sc = SparkContext.getOrCreate()
    # Init session
    print("Init session")
    my_spark = SparkSession.builder.getOrCreate()
    print(my_spark.catalog.listTables())

    data_path='/dataset/NeurIPS2022/train_cite_inputs.h5'
    df_data = read_h5_data(my_spark, data_path)
    overview_data(df_data)
    # sparkDF.show()
    # Create a temporary table on catalog of local data frame flights as new temporary table flights_temp on catalog
    df_data.createOrReplaceTempView('single_cell_temp')
    # check list all table available on catalog
    print(my_spark.catalog.listTables())
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

