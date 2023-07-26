import os
import findspark
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pandas as pd    
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/"
os.environ["SPARK_HOME"] = "/workspace/tripx/MCS/big_data/spark-3.1.1-bin-hadoop3.2"
findspark.init()
import time 
import logging
import seaborn as sns
from tqdm import tqdm
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
import pyspark.sql.functions as f
from pyspark.sql.functions import lit
from pyspark.sql.functions import col, count, explode, sum as sum_
from pyspark.ml.regression import LinearRegression


# Stop spark if it existed
SparkContext.setSystemProperty('spark.executor.memory', '200g')

try:
    sc.stop()
except:
    print('sc have not yet created!')
sc = SparkContext(master = "local[*]", appName = "Multimodal Single Cell")
# sc = SparkContext.getOrCreate()
# Init session
print("Init session")
my_spark = SparkSession.builder.getOrCreate()
print(my_spark.catalog.listTables())

print(sc._conf.getAll())

train_cite_target_path = "/workspace/tripx/MCS/big_data/data/train_cite_targets.csv"
train_cite_target_data = my_spark.read.csv(train_cite_target_path,  header=True)

train_cite_target_data.createOrReplaceTempView('train_cite_target_data')
train_output = train_cite_target_data
new_cols=(column.replace('.', '_') for column in train_output.columns)
train_output = train_output.toDF(*new_cols)

for col_name in tqdm(train_output.columns[1:]):
    # print(col_name)
    train_output = train_output.withColumn(col_name,  col(col_name).cast('float'))
    

# Load extracted features
feature_path='/dataset/NeurIPS2022/2nd-solution/kaggle/src_top2/senkin13/features/'
train_df = pd.read_feather(feature_path+'train_cite_inputs_id.feather')
test_df = pd.read_feather(feature_path+'test_cite_inputs_id.feather')

def zscore(x):
    x_zscore = []
    for i in range(x.shape[0]):
        x_row = x[i]
        x_row = (x_row - np.mean(x_row)) / np.std(x_row)
        x_zscore.append(x_row)
    x_std = np.array(x_zscore)    
    return x_std

cite_inputs_svd_clr = np.load(feature_path+'cite_inputs_svd_clr_200.npy')
train_cite_svd_clr = cite_inputs_svd_clr[:len(train_df)]
test_cite_svd_clr = cite_inputs_svd_clr[len(train_df):]
train_cite_svd_clr = zscore(train_cite_svd_clr)
test_cite_svd_clr = zscore(test_cite_svd_clr)

print(train_cite_svd_clr.shape)
print(test_cite_svd_clr.shape)

# Train 

train_feature_df = map(lambda x: (int(1), Vectors.dense(x)), train_cite_svd_clr)
train_feature_df = my_spark.createDataFrame(train_feature_df,schema=["label", "features"])

train_data = train_feature_df.join(train_output)
train_data = train_data.drop('label')

output_columns = train_output.columns[1:]

def train(lr_model, train_data, column_name):
    data_vector = train_data.select("features", column_name)
    data_vector = data_vector.withColumnRenamed(column_name, "label")
    # data_vector.show(5)
    # input()
    lr_model = lr.fit(data_vector)
    save_path = "/workspace/tripx/MCS/big_data/models/" + column_name
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    lr_model.save(save_path)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))
    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
    

lr = LinearRegression(featuresCol = 'features', labelCol='label', \
    maxIter=10, regParam=0.3, elasticNetParam=0.8)

for column_name in tqdm(output_columns):
    train(lr, train_data, column_name)