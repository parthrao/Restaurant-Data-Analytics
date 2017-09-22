There are two files for the analytics

Main_Analytics which contains majority of the code

This file can be run using pyspark with command "spark-submit Main_Analytics.py "

Our merged data on which we ran analytics is on dumbo cluster under phr240/business.json and phr240/review.json

You may need to change the path in Main_Analytics to load the data correctly

Another issue that can come is with the external libraries that this file use. 
If you face any issue running this file please contact Parth Rao at phr240@nyu.edu (2019894134)


Another is Pig Program file which uses review.json
