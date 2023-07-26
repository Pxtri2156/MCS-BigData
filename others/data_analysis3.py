import os
import findspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/"
os.environ["SPARK_HOME"] = "/workspace/tripx/MCS/big_data/spark-3.1.1-bin-hadoop3.2"
findspark.init()
import pandas as pd    


# Stop spark if it existed.
try:
    sc.stop()
except:
    print('sc have not yet created!')
    
sc = SparkContext(master = "local", appName = "First app")
# Check spark context version
print(sc.version) 
sc = SparkContext.getOrCreate()
# print(sc)
# Init session
print("Init session")
my_spark = SparkSession.builder \
    .appName("Read HDF5 file with Spark") \
    .getOrCreate()

h5_df = my_spark.read.format("binaryFile").option("pathGlobFilter", "*.png").load("/dataset/NeurIPS2022/1st-solution/test_cite_inputs_values.sparse.npz")

h5_df.show()
h5_df.printSchema()
my_spark.stop()

########################## Analysis ################################

# Number record 

# Number feature

# Max cell is non-zero value

# Average cell contains non-zero value

# The average non-zero value 

# The standard deviation of our features

# The average standard deviation 

########################## Analysis ################################

