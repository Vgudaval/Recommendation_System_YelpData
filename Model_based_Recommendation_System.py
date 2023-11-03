import xgboost as xgb
from xgboost import XGBRegressor
import collections
import itertools
import sys, os
import random
import json
import csv
import time
from itertools import combinations
from pyspark import SparkContext, SparkConf
from random import randint
import math

if __name__ == '__main__':
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    sc = SparkContext()
    sc.setLogLevel('WARN')

    user_keys = ['user_id', 'review_count', 'average_stars', ]
    business_keys = ['business_id', 'stars', 'latitude', 'longitude', 'review_count', 'is_open',
                     'attributes.BusinessAcceptsCreditCards', 'attributes.GoodForKids', 'attributes.HasTV',
                     'attributes.OutdoorSeating']


    def const_feat_set(feature_dict, feature_keys):
        result = []
        for key in feature_keys:
            levels = key.split(".")
            curr_dict = feature_dict
            for level in levels[:-1]:
                curr_dict = curr_dict[level] if level in curr_dict else {}
            if curr_dict is not None:
                curr_val = curr_dict.get(levels[-1], None)
                if isinstance(curr_val, str):
                    if curr_val == "True":
                        curr_val = 1
                    elif curr_val == "False":
                        curr_val = 0
            else:
                curr_val = None
            result.append(curr_val)
        return (result[0], result[1:])


    def combiner_(curr_set, curr_record):
        for i in range(len(curr_record[1])):
            if curr_record[i] is not None:
                curr_set[i] = (curr_set[i][0] + curr_record[1][i], curr_set[i][1] + 1)
            return curr_set


    def reducer_(record1, record2):
        return [(record1[i][0] + record2[i][0], record1[i][1] + record2[i][1]) for i in range(len(record1))]


    def replace_none_(record, means):
        result = [r if r is not None else means[i] for i, r in enumerate(record[1])]
        return record[0], result

    def rating_rdd_func(rating_path):
        r1_rdd = sc.textFile(rating_path).repartition(4)
        header = 'user_id,business_id,stars'
        r1_rdd = r1_rdd.filter(lambda x: x != header).map(lambda x: x.split(","))
        return r1_rdd


    folder_path = sys.argv[1]
    rating_path = f"{folder_path}/yelp_train.csv"
    test_path = sys.argv[2]
    business_path = f"{folder_path}/business.json"
    user_path = f"{folder_path}/user.json"

    output_path = sys.argv[3]

    start = time.time()

    #rating_rdd = sc.textFile(rating_path).repartition(4)
    #header = 'user_id,business_id,stars'
    #rating_rdd = rating_rdd.filter(lambda x: x != header).map(lambda x: x.split(","))
    rating_rdd = rating_rdd_func(rating_path)

    user_rdd = sc.textFile(user_path).repartition(4)
    user_rdd = user_rdd.map(lambda x: json.loads(x))
    user_rdd = user_rdd.map(lambda x: const_feat_set(x, user_keys))
    user_means = user_rdd.aggregate([(0, 0)] * (len(user_keys) - 1), combiner_, reducer_)

    user_means_ = [mean[0] / mean[1] if mean[1] != 0 else 0 for mean in user_means]
    user_rdd = user_rdd.map(lambda x: replace_none_(x, user_means_))
    user_map = user_rdd.collectAsMap()

    business_rdd = sc.textFile(business_path).repartition(4)
    business_rdd = business_rdd.map(lambda x: json.loads(x))
    business_rdd = business_rdd.map(lambda x: const_feat_set(x, business_keys))
    business_means = business_rdd.aggregate([(0, 0)] * (len(business_keys) - 1), combiner_, reducer_)

    business_means_ = [mean[0] / mean[1] if mean[1] != 0 else 0 for mean in business_means]
    business_rdd = business_rdd.map(lambda x: replace_none_(x, business_means_))
    business_map = business_rdd.collectAsMap()

    user_rating_rdd = rating_rdd.map(lambda x: (x[0], x[1:]))
    user_rating_rdd = user_rating_rdd.join(user_rdd)
    user_rating_rdd = user_rating_rdd.map(
        lambda x: (x[1][0][0], (x[0], float(x[1][0][1]), x[1][1])))

    feature_rating_rdd = user_rating_rdd.join(business_rdd)
    feature_rating_rdd = feature_rating_rdd.map(
        lambda x: ((x[1][0][0], x[0]), (x[1][0][2], x[1][1]), x[1][0][1])).sortBy(lambda x: x[0])

    train_features = feature_rating_rdd.map(lambda x: x[1][0] + x[1][1])
    train_features = train_features.collect()

    train_ratings = feature_rating_rdd.map(lambda x: x[2])
    train_ratings = train_ratings.collect()

    xgb_model = XGBRegressor(njobs=-1, verbosity=0, random_state=42)
    xgb_model.fit(train_features, train_ratings)

    test_rdd = sc.textFile(test_path).repartition(4)
    header_test = 'user_id,business_id,stars'
    test_rdd = test_rdd.filter(lambda x: x != header_test)
    test_rdd = test_rdd.map(lambda x: x.split(",")[:2])
    test_features_rdd = test_rdd.map(
        lambda x: (x, (user_map.get(x[0], user_means_), business_map.get(x[1], business_means_))))
    test_features_rdd = test_features_rdd.sortBy(
        lambda x: x[0])

    test_features = test_features_rdd.mapValues(lambda x: x[0] + x[1])
    test_features = test_features.values().collect()
    test_user_business = test_features_rdd.keys().collect()
    preds = xgb_model.predict(test_features)

    result = [(ub[0], ub[1], 2.5 if pred < 1.0 or pred > 5.0 else pred) for ub, pred in zip(test_user_business, preds)]

    end_time = time.time()
    print("Total time is :", end_time - start)

    with open(output_path, "w") as out_file:
        out_file.write("user_id, business_id, prediction\n")
        for res in result:
            out_file.write(f"{res[0]},{res[1]},{res[2]}\n")
        out_file.close()
