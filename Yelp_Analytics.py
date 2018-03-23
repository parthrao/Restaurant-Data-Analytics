from pyspark import SparkContext, SparkConf, SparkSession
from pyspark.sql import SQLContext;
from pyspark.sql import *
from pyspark.sql import functions as F
from operator import add
import json
import requests
from pyspark.accumulators import AccumulatorParam
import numpy
from pyspark.sql.functions import lit
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def listAllCategories(row):
    # print row.categories
    if row.categories:
         return (row.categories, row.stars, row.review_count)
    else :
        return []
    
def listAllCityCategories(row):
    return row.categories

if __name__ == "__main__":
    
    conf = SparkConf().setAppName('Yelp_Analytics')
    sc = SparkContext(conf=conf)

    sqlContext = SQLContext(sc);

    spark = SparkSession \
        .builder \
        .appName("Yelp_Analytics") \
        .getOrCreate()

    # Load the data into dataFrames
    review = spark.read.json("/user/zeppelin/Yelp/review_sample2.json")

    business = spark.read.json("/user/zeppelin/Yelp/business_sample.json")

    # Drop Useless columns
    business = business.drop('latitude')
    business = business.drop('longitude')
    business = business.drop('address')
    business = business.drop('hours')
    business = business.drop('postal_code')
    business = business.drop('neighborhood')

    # Create Table
    business.createOrReplaceTempView('business')
    groupedByCity = spark.sql("select city,count(*) as count from business group by city order by count desc")

    # get First 10 cities with highest reviews
    first10 = groupedByCity.head(10)
    topCities = []
    for row in first10:
        topCities.append(row.city)
    print("These are the cities with maximum number of reviews")
    print(topCities)




    # Find only restaurants of One city and do analytics on that
    cityTable = business.where(business.city == topCities[0])
    cityRestaurantTable = cityTable.select ("*").where (F.array_contains (cityTable["categories"], "Restaurants") | F.array_contains (cityTable["categories"], "Desserts") | F.array_contains (cityTable["categories"], "Nightlife") )

    cityRestaurantTable = cityRestaurantTable.orderBy(business.review_count.desc())

    # Find average reviews and stars of the city restaurants
    averageReview = cityRestaurantTable.agg(F.avg(cityRestaurantTable.review_count)).first()['avg(review_count)']
    averageStars =  cityRestaurantTable.agg(F.avg(cityRestaurantTable.stars)).first()['avg(stars)']

    # Find restaurants that are above average both in number of reviews and number of stars
    aboveAverageRestaurants = cityRestaurantTable.where(cityRestaurantTable.review_count > averageReview )
    aboveAverageRestaurants = aboveAverageRestaurants.where(cityRestaurantTable.stars > averageStars )

    # Find all list of categories which are present in these restaurant list
    listOfCategories = []



    tempList = aboveAverageRestaurants.rdd.map(listAllCategories).collect()

    for x in tempList:
        for category in x[0]:
            listOfCategories.append((category, x[1]))

    for x in tempList:
        for category in x[0]:
            listOfReviews.append((category, x[2]))



    # Find number of times each category appears in the table
    categoryFrquencyRdd = sc.parallelize(listOfCategories)
    categoryFrquencyRdd.reduceByKey(add)

    reviewFrquencyRdd = sc.parallelize(listOfReviews)
    reviewFrquencyRdd.reduceByKey(add)

    # Show those categories and their frequencies in a table
    categoryFrequencyTable = sqlContext.createDataFrame(categoryFrquencyRdd.reduceByKey(add).collect(), ['category', 'frequency'])
    categoryFrequencyTable = categoryFrequencyTable.where((categoryFrequencyTable.category != 'Restaurants') & (categoryFrequencyTable.category != 'Food') )

    reviewFrequencyTable = sqlContext.createDataFrame(reviewFrquencyRdd.reduceByKey(add).collect(), ['category1', 'reviews'])
    reviewFrequencyTable = reviewFrequencyTable.where((reviewFrequencyTable.category1 != 'Restaurants') & (reviewFrequencyTable.category1 != 'Food') )

    # Do the join 
    finalTable = categoryFrequencyTable.join(reviewFrequencyTable,categoryFrequencyTable.category == reviewFrequencyTable.category1, 'inner' ).select(['category','frequency','reviews'])
    finalTable = finalTable.orderBy(F.desc('frequency'))
    finalTable.show(10)

    # Save it in HDFS as a csv file
    finalTable.collect()
    finalTable.write.json('/user/zeppelin/output/' + topCities[i], 'overwrite')

    # Label the data That means label restaurnts as successful if they have more that average reviews and star rating . This will be our target variable for prdictive
    # modeling
    labeledCityRestaurantTable = cityRestaurantTable.select(cityRestaurantTable.name,cityRestaurantTable.review_count, cityRestaurantTable.stars,
    cityRestaurantTable.categories,cityRestaurantTable.attributes, F.when((cityRestaurantTable.review_count > averageReview)&(cityRestaurantTable.stars > averageStars), 1).otherwise(0))

    labeledCityRestaurantTable = labeledCityRestaurantTable.withColumnRenamed(labeledCityRestaurantTable.schema.names[5], 'label')
    labeledCityRestaurantTable.show(4)

    # find List Of All different Categories present in all restaurants
    listOfAllCityCategories = []



    tempListAll = labeledCityRestaurantTable.rdd.map(listAllCityCategories).collect()

    for x in tempListAll:
        for category in x:
            listOfAllCityCategories.append(category)

    listOfAllCityCategory = list(set(listOfAllCityCategories))

    # creates category labels
    for category in listOfAllCityCategory:
       labeledCityRestaurantTable=labeledCityRestaurantTable.withColumn(category,( F.array_contains (labeledCityRestaurantTable.categories, category)).cast('integer'))


    # Remove Columns that we are not going to use in predicitve modeling
    labeledCityRestaurantTable1 = labeledCityRestaurantTable.drop('categories');
    labeledCityRestaurantTable1 = labeledCityRestaurantTable1.drop('name');
    labeledCityRestaurantTable1 = labeledCityRestaurantTable1.drop('stars');
    labeledCityRestaurantTable1 = labeledCityRestaurantTable1.drop('review_count');
    labeledCityRestaurantTable1 = labeledCityRestaurantTable1.drop('attributes');

    # Build pipeline for modeling
    stages = []

    assembler = VectorAssembler(inputCols=listOfAllCityCategory, outputCol="features")
    stages += [assembler]

    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)
    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.
    pipelineModel = pipeline.fit(labeledCityRestaurantTable1)
    dataset = pipelineModel.transform(labeledCityRestaurantTable1)

    # Keep relevant columns
    selectedcols = ["label", "features"]
    dataset = dataset.select(selectedcols)

    # Split in train and test data
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)


    # Making decision tree model
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=3)

    # Train model with Training Data
    dtModel = dt.fit(trainingData)
    print "numNodes = ", dtModel.numNodes
    print "depth = ", dtModel.depth

    # Run the model on the Test set
    predictions = dtModel.transform(testData)

    # Evaluate the model
    evaluator = BinaryClassificationEvaluator()
    print ("ROC AUS (Accuracy Measure)for decision tree", evaluator.evaluate(predictions))



    # Create initial LogisticRegression model
    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

    # Train model with Training Data
    lrModel = lr.fit(trainingData)

    # Predict using Test Data
    predictions = lrModel.transform(testData)

    # Evaluate the mode
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    print ("ROC AUS (Accuracy Measure) for logistic regression",evaluator.evaluate(predictions))




    # Conclusion: DecisionTree is performaing better than logistic regression But ingeneral on categorical features both performing bad
    print (" Conclusion: DecisionTree is performaing better than logistic regression But ingeneral on categorical features both performing bad")

    binaryAttributes = [ 'BikeParking','BusinessAcceptsBitcoin','BusinessAcceptsCreditCards','DogsAllowed',
    'WheelchairAccessible',
    'HasTV',
    'OutdoorSeating',
    'RestaurantsDelivery',
    'RestaurantsGoodForGroups',
    'RestaurantsReservations',
    'RestaurantsTableService',
    'RestaurantsTakeOut',
    'WiFi',
    'GoodForKids',
    ]

    allRestaurantTable = business.select ("*").where (F.array_contains (cityTable["categories"], "Restaurants") | F.array_contains (cityTable["categories"], "Desserts") | F.array_contains (cityTable["categories"], "Nightlife") )

    # allRestaurantTable = allRestaurantTable.orderBy(business.review_count.desc())
    labeledAllRestaurantTable = allRestaurantTable.select(allRestaurantTable.name,allRestaurantTable.review_count, allRestaurantTable.stars,
    allRestaurantTable.categories,allRestaurantTable.attributes, F.when((allRestaurantTable.review_count > averageReview)&(allRestaurantTable.stars > averageStars-0.5), 1).otherwise(0))

    labeledAllRestaurantTable = labeledAllRestaurantTable.withColumnRenamed(labeledAllRestaurantTable.schema.names[5], 'label')
    print("Labels All Restaurant data . Success means more than average stars and reviews")
    labeledAllRestaurantTable.show(4)


    #  creates attribute labels
    labeledCityRestaurantTableWithAttributes = labeledAllRestaurantTable

    # F.array_contains(arr, "BikeParking: False")
    for attribute in binaryAttributes:
      labeledCityRestaurantTableWithAttributes = labeledCityRestaurantTableWithAttributes.withColumn(attribute,( F.array_contains(labeledCityRestaurantTableWithAttributes.attributes, attribute + ": True")).cast('integer'))


    labeledCityRestaurantTableWithAttributes = labeledCityRestaurantTableWithAttributes.drop('categories');
    labeledCityRestaurantTableWithAttributes = labeledCityRestaurantTableWithAttributes.drop('name');
    labeledCityRestaurantTableWithAttributes = labeledCityRestaurantTableWithAttributes.drop('stars');
    labeledCityRestaurantTableWithAttributes = labeledCityRestaurantTableWithAttributes.drop('review_count');
    labeledCityRestaurantTableWithAttributes = labeledCityRestaurantTableWithAttributes.drop('attributes');
    # labeledCityRestaurantTableWithAttributes.schema.names

    # Drop rows with null values
    labeledCityRestaurantTableWithAttributes = labeledCityRestaurantTableWithAttributes.na.drop()

    # Mkae Predictive model using different features
    stages = []

    assembler = VectorAssembler(inputCols=binaryAttributes, outputCol="features")
    stages += [assembler]

    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)
    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.
    pipelineModel = pipeline.fit(labeledCityRestaurantTableWithAttributes)
    dataset = pipelineModel.transform(labeledCityRestaurantTableWithAttributes)
    # dataset.show(5)

    # Keep relevant columns
    selectedcols = ["label", "features"]
    dataset = dataset.select(selectedcols)

    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)

    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=5)

    # Train model with Training Data
    dtModel = dt.fit(trainingData)

    # Run the model on the Test set
    predictions = dtModel.transform(testData)
    selected = prediions.select("label", "prediction")
    selected.show(10)

    # Evaluation 
    evaluator = BinaryClassificationEvaluator()
    print("Accuraacy of the model considering other features", evaluator.evaluate(predictions))
    print("This is way better than what we got when we didnt consider all these features")
