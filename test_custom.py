# Add Spark Python Files to Python Path
import sys
import os
SPARK_HOME = "/lib/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path
from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from operator import add
def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = SparkContext(conf = conf) 
    return sc
sc = getSparkContext()

# Load and parse the data
data = sc.textFile("/dataset/data_banknote_authentication.txt")
def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """
    feats = line.strip().split(",")
    # labels must be at the beginning for LRSGD, it's in the end in our data, so
    # putting it in the right place
    label = feats[len(feats) - 1]
    feats = feats[: len(feats) - 1]
    features = [ float(feature) for feature in feats ] # need floats
    return LabeledPoint(label, features)

n_iter = 5
parsedData = data.map(mapper)
w_len = len(parsedData.collect()[0].features)
print(w_len)




def CustomSGD():
    w=np.zeros(shape=(1,w_len))
    cur_iter=1
    while(cur_iter<n_iter):
        temp = parsedData.sample(True, 0.1)
        gradient = temp.map( lambda point:(1/(1+ np.exp(-point.label*np.dot(w,point.features)))-1)*point.label*point.features).reduce(add)
        w = w - gradient
    return w
w = CustomSGD()

