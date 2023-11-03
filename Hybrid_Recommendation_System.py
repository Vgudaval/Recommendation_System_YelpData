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

    VALID_BUSINESS_COUNT = 23
    NUM_NEIGHBORS = 65536
    ALPHA = 0.995

    user_keys = ['user_id', 'review_count', 'average_stars', ]
    business_keys = ['business_id', 'stars', 'latitude', 'longitude', 'review_count', 'is_open',
                     'attributes.BusinessAcceptsCreditCards', 'attributes.GoodForKids', 'attributes.HasTV',
                     'attributes.OutdoorSeating']


    def predict_rating(row, business_map, user_map, user_avg_rdd, business_avg_rdd):
        business, user = row
        if (business not in business_map) and (user not in user_map):
            return user, business, 2.5, ALPHA
        if (business not in business_map):
            return user, business, user_avg_rdd[user], ALPHA
        if (user not in user_map):
            return user, business, business_avg_rdd[business], ALPHA
        coeffs = []
        row1 = business_map[business]
        avg1 = business_avg_rdd[business]
        curr_user_map = user_map[user]
        for business2 in curr_user_map:
            if business2 not in business_map:
                continue
            avg2 = business_avg_rdd[business2]
            row2 = business_map[business2]
            common_users = set(row1.keys()) & set(row2.keys())
            if (len(common_users) < VALID_BUSINESS_COUNT):
                h = row2[user] - avg2
                s = business_avg_rdd[business] / business_avg_rdd[business2]
                coeffs.append((h, s))
            else:
                num = 0
                norm1 = 0
                norm2 = 0
                for key in common_users:
                    rating1 = row1[key]
                    rating2 = row2[key]
                    norm1 += (rating1 - avg1) ** 2
                    norm2 += (rating2 - avg2) ** 2
                    num += (rating1 - avg1) * (rating2 - avg2)
                denom = math.sqrt(norm1 * norm2)
                if denom == 0:
                    coeff = 0
                    c = row2[user] - avg2

                    coeffs.append((c, coeff))
                else:
                    f = num / denom
                    coeffs.append((row2[user] - avg2, f))
        coeffs.sort(key=lambda x: math.fabs(x[1]), reverse=True)
        coeffs = coeffs[:NUM_NEIGHBORS]
        numer = 0
        denom = 0
        curr_alpha = (ALPHA * len(coeffs)) / len(business_avg_rdd)
        for curr_rating, coeff in coeffs:
            numer += curr_rating * coeff
            denom += math.fabs(coeff)
        rating = avg1 + (curr_alpha * (numer / denom)) if denom != 0 else (user_avg_rdd[user] + business_avg_rdd[
            business]) / 2
        if rating <= 0 or rating > 5:
            e = (user_avg_rdd[user] + business_avg_rdd[business])
            rating = e / 2

        return user, business, rating, curr_alpha


    def item_based(train_path, test_path):
        input_rdd = sc.textFile(train_path)
        header = 'user_id,business_id,stars'
        input_rdd = input_rdd.filter(lambda x: x != header).map(lambda x: x.split(","))
        input_rdd = input_rdd.map(
            lambda x: (x[1], (x[0], float(x[2]))))

        business_rdd = input_rdd.groupByKey()
        business_rdd = business_rdd.mapValues(lambda x: {k: v for k, v in x})
        business_rdd = business_rdd.filter(
            lambda x: len(x[1]) >= VALID_BUSINESS_COUNT)
        business_map = business_rdd.collectAsMap()
        user_rdd = input_rdd.map(lambda x: (x[1][0], (x[0], x[1][1])))
        user_rdd = user_rdd.groupByKey()
        user_rdd = user_rdd.mapValues(
            lambda x: {k: v for k, v in x})
        user_map = user_rdd.collectAsMap()
        user_avg_rdd = user_rdd.mapValues(lambda x: sum(x.values()) / len(x))
        user_avg_rdd = user_avg_rdd.collectAsMap()
        business_avg_rdd = business_rdd.mapValues(lambda x: sum(x.values()) / len(x))
        business_avg_rdd = business_avg_rdd.collectAsMap()

        test_rdd = sc.textFile(test_path)
        header = 'user_id,business_id,stars'
        test_rdd = test_rdd.filter(lambda x: x != header)
        test_rdd = test_rdd.map(lambda x: x.split(","))
        test_rdd = test_rdd.map(lambda x: (x[1], x[0]))
        result = test_rdd.map(lambda x: predict_rating(x, business_map, user_map, user_avg_rdd, business_avg_rdd))
        r = result.toLocalIterator()
        return r

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


    def model_based(folder_path, test_path):

        rating_path = f"{folder_path}/yelp_train.csv"
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
            lambda x: ((x[1][0][0], x[0]), (x[1][0][2], x[1][1]), x[1][0][1]))
        feature_rating_rdd = feature_rating_rdd.sortBy(lambda x: x[0])

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

        result = [(ub[0], ub[1], 2.5 if pred < 1.0 or pred > 5.0 else pred) for ub, pred in
                  zip(test_user_business, preds)]

        return result


    train_path = sys.argv[1]
    rating_path = f"{train_path}/yelp_train.csv"
    test_path = sys.argv[2]
    business_path = f"{train_path}/business.json"
    user_path = f"{train_path}/user.json"
    output_path = sys.argv[3]

    start = time.time()

    item_ratings = item_based(f"{train_path}/yelp_train.csv", test_path)
    model_ratings = model_based(train_path, test_path)
    result = []
    for i, (item_rating, model_rating) in enumerate(zip(item_ratings, model_ratings)):
        user_id, business_id, model_rating = model_rating
        item_rating, alpha = item_rating[2:]
        new_rating = model_rating * ALPHA + (1 - ALPHA) * item_rating
        new_rating = 2.5 if new_rating < 1.0 or new_rating > 5.0 else new_rating
        result.append((user_id, business_id, new_rating))

    with open(output_path, "w") as out_file:
        out_file.write("user_id, business_id, prediction\n")
        for res in result:
            out_file.write(f"{res[0]},{res[1]},{res[2]}\n")
        out_file.close()

    print("Duration: ", time.time() - start)





















