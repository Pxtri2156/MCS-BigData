import os
import findspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/"
os.environ["SPARK_HOME"] = "./spark-3.1.1-bin-hadoop3.2"
findspark.init()
# Stop spark if it existed.
try:
    sc.stop()
except:
    print('sc have not yet created!')
    
sc = SparkContext(master = "local", appName = "First app")
# Check spark context
print(sc)
# Check spark context version
print(sc.version) 
sc = SparkContext.getOrCreate()
print(sc)

print("Init session")
# my_spark = SparkSession.builder.getOrCreate()
# # Print my_spark session
# print(my_spark)