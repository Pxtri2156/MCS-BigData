import os
import findspark
import numpy as np
import pandas as pd    
import time 
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
import pyspark.sql.functions as f
from pyspark.sql.functions import lit
from pyspark.sql.functions import col, count, explode, sum as sum_
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, FMRegressor

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/"
os.environ["SPARK_HOME"] = "/workspace/tripx/MCS/big_data/spark-3.1.1-bin-hadoop3.2"
findspark.init()


def train(train_data, column_name):
    lr_model = LinearRegression(featuresCol = 'features', labelCol='label', \
    maxIter=50, regParam=0.3, elasticNetParam=0.8)
    data_vector = train_data.select("features", column_name)
    data_vector = data_vector.withColumnRenamed(column_name, "label")
    data_vector.show(5)
    # input()
    lr_model = lr_model.fit(data_vector)
    save_path = "/workspace/tripx/MCS/big_data/models/" + column_name
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    lr_model.write().overwrite().save(save_path)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))
    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
    
def train_GBTs(train_data, test_data, column_name):
    data_vector = train_data.select("features", column_name)
    data_vector = data_vector.withColumnRenamed(column_name, "label")
    data_vector.show(5)
    
    test_vector =  test_data.select("features", column_name)
    test_vector = test_vector.withColumnRenamed(column_name, "label")
    test_vector.show(5)
    
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data_vector)
    # (trainingData, testData) = data_vector.randomSplit([0.7, 0.3])
    gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)
    pipeline = Pipeline(stages=[featureIndexer, gbt])
    model = pipeline.fit(data_vector)
    predictions = model.transform(test_vector)
    predictions.select("prediction", "label", "features").show(5)
    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    gbtModel = model.stages[1]
    save_path = "/workspace/tripx/MCS/big_data/models/decision_tree/" + column_name
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    gbtModel.write().overwrite().save(save_path)
    print(gbtModel)
    
    # result = predictions.select("prediction")
    # result = result.withColumnRenamed('prediction', column_name)
    # result = result.select("*").withColumn("id", monotonically_increasing_id())
    return rmse


def train_general(train_data, test_data, column_name, model_pyspark, model_name):
    print('model name: ', model_name)
    data_vector = train_data.select("features", column_name)
    data_vector = data_vector.withColumnRenamed(column_name, "label")
    data_vector.show(5)
    
    test_vector =  test_data.select("features", column_name)
    test_vector = test_vector.withColumnRenamed(column_name, "label")
    test_vector.show(5)
    
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data_vector)
    # gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)
    pipeline = Pipeline(stages=[featureIndexer, model_pyspark])
    model = pipeline.fit(data_vector)
    predictions = model.transform(test_vector)
    predictions.select("prediction", "label", "features").show(5)
    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    gbtModel = model.stages[1]
    origin_model =  f"/workspace/tripx/MCS/big_data/models/{model_name}/"
    if not os.path.isdir(origin_model):
        os.mkdir(origin_model)
        
    save_path = f"/workspace/tripx/MCS/big_data/models/{model_name}/" + column_name
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    gbtModel.write().overwrite().save(save_path)
    print(gbtModel)
    return rmse

def zscore(x):
    x_zscore = []
    for i in range(x.shape[0]):
        x_row = x[i]
        x_row = (x_row - np.mean(x_row)) / np.std(x_row)
        x_zscore.append(x_row)
    x_std = np.array(x_zscore)    
    return x_std

def load_output(my_spark, train_cite_target_path):
    train_cite_target_data = my_spark.read.csv(train_cite_target_path,  header=True)

    train_cite_target_data.createOrReplaceTempView('train_cite_target_data')
    train_output = train_cite_target_data
    new_cols=(column.replace('.', '_') for column in train_output.columns)
    train_output = train_output.toDF(*new_cols)
    for col_name in tqdm(train_output.columns[1:]):
    # print(col_name)
        train_output = train_output.withColumn(col_name,  col(col_name).cast('float'))
    train_output =  train_output.select("*").withColumn("index", monotonically_increasing_id())
    return train_output

def load_extracted_features(my_spark, feature_path):
    train_df = pd.read_feather(feature_path+'train_cite_inputs_id.feather')
    test_df = pd.read_feather(feature_path+'test_cite_inputs_id.feather')


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
    train_feature_df = train_feature_df.drop("label")
    train_feature_df = train_feature_df.select("*").withColumn("index", monotonically_increasing_id())
    return train_feature_df


def load_extracted_test_features(my_spark, feature_path):
    train_df = pd.read_feather(feature_path+'train_cite_inputs_id.feather')
    test_df = pd.read_feather(feature_path+'test_cite_inputs_id.feather')
    cite_inputs_svd_clr = np.load(feature_path+'cite_inputs_svd_clr_200.npy')
    test_cite_svd_clr = cite_inputs_svd_clr[len(train_df):]
    test_cite_svd_clr = zscore(test_cite_svd_clr)

    # print(test_cite_svd_clr.shape)
    # Test
    test_feature_df = map(lambda x: (int(1), Vectors.dense(x)), test_cite_svd_clr)
    test_feature_df = my_spark.createDataFrame(test_feature_df,schema=["label", "features"])
    test_feature_df = test_feature_df.drop("label")
    test_feature_df = test_feature_df.select("*").withColumn("index", monotonically_increasing_id())
    return test_feature_df

def choose_model(model_name):
    print("huhu")
    if model_name == "GradientBoostedTree ":
        model = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)
    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor(featuresCol="indexedFeatures")
    elif model_name == "LinearRegression":
        model = LinearRegression(featuresCol = 'indexedFeatures', labelCol='label', \
    maxIter=50, regParam=0.3, elasticNetParam=0.8)
    elif model_name == "FMRegressor":
        model =  FMRegressor(featuresCol="indexedFeatures", stepSize=0.001)
    else:
        raise("Model name don't match def")
    return model

def load_data(my_spark, feature_path, train_output, num=10000):
    train_feature_df = load_extracted_features(my_spark, feature_path)
    train_data = train_feature_df.join(train_output, train_feature_df.index == train_output.index, 'left')
    train_data = train_data.drop("index")
    train_data = train_data.select("*").withColumn("id", monotonically_increasing_id())
    train_data = train_data.where(train_data.id <= num)
    return train_data

def load_test_data(my_spark, feature_path, train_output, num=5000):
    test_feature_df = load_extracted_test_features(my_spark, feature_path)
    test_data = test_feature_df.join(train_output, test_feature_df.index == train_output.index, 'left')
    test_data = test_data.drop("index")
    test_data = test_data.select("*").withColumn("id", monotonically_increasing_id())
    test_data = test_data.where(test_data.id <= num)
    return test_data
    
def main(args):
    SparkContext.setSystemProperty('spark.executor.memory', '200g')
    # SparkContext.setSystemProperty('spark.plugins', 'com.nvidia.spark.SQLPlugin')
    # SparkContext.setSystemProperty('spark.rapids.sql.enabled','true')
    try:
        sc.stop()
    except:
        print('sc have not yet created!')
    sc = SparkContext(master = "local[*]", appName = "Multimodal Single Cell")
    # sc = SparkContext.getOrCreate()
    # Init session
    print("Init session")
    my_spark = SparkSession.builder.getOrCreate()
    my_spark.conf.set('spark.rapids.sql.enabled','true')
    print(my_spark.catalog.listTables())
    # print(sc._conf.getAll())

    # Load train output
    train_cite_target_path = "/workspace/tripx/MCS/big_data/data/train_cite_targets.csv"
    train_output = load_output(my_spark, train_cite_target_path)
    
    # Load extracted features
    feature_path='/dataset/NeurIPS2022/2nd-solution/kaggle/src_top2/senkin13/features/'
    train_data = load_data(my_spark, feature_path, train_output, 10000)
    test_data = load_test_data(my_spark, feature_path, train_output, 5000)
    # input("STOP")
    output_columns = train_output.columns[1:]
    avg_rmse = []

    model_name = args.model
    model = choose_model(model_name)
    for column_name in tqdm(output_columns):
        rmse = train_general(train_data, test_data, column_name, model, model_name)
        avg_rmse.append(rmse)
        # break
    print(avg_rmse)
    print(sum(avg_rmse)/len(avg_rmse))
        
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default='LinearRegression',
        type=str,
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    main(args)