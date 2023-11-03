import collections
import itertools
import sys, os
import random
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

    num_hashes = 30
    num_rows = 2

    def hash_func(num_hashes):

        valid_nums = set(range(1, 1024))
        hash_funct = []
        while len(hash_funct) < num_hashes:
            a = random.choice(list(valid_nums))
            b = random.choice(list(valid_nums))
            if (a, b) not in hash_funct:
                hash_funct.append((a, b))
        return hash_funct


    def create_sign(row, hash_functions, users, num_buckets):
        business_id = row[0]
        result = [num_buckets + 1] * len(hash_functions)
        for i, user in enumerate(users):
            if user in row[1]:
                for j, hash_function in enumerate(hash_functions):
                    hash_val = (i * hash_function[0] + hash_function[1]) % num_buckets
                    result[j] = min(result[j], hash_val)
        return (business_id, tuple(result))


    def create_bands(x, num_bands):
        business, signatures = x
        bands = []
        for band in range(num_bands):
            band_signature = signatures[band * num_rows:(band + 1) * num_rows]
            bands.append((tuple(band_signature), [business]))
        return bands


    def getPairs(business_ids):
        pairs = set()
        for i in range(len(business_ids)):
            for j in range(i + 1, len(business_ids)):
                pairs.add(tuple(sorted([business_ids[i], business_ids[j]])))
        return pairs


    def jacc(sets):
        set_1, set_2 = sets
        intersection = len(set_1 & set_2)
        union = len(set_1 | set_2)
        similarity = intersection / union
        return (similarity,)


    input_path = sys.argv[1]
    output_path = sys.argv[2]

    start = time.time()

    #input_rdd = sc.textFile(input_path)
    #header = 'user_id,business_id,stars'
    #input_rdd = input_rdd.filter(lambda x: x != header).map(lambda x: x.split(",")[:2]).map(lambda x: (x[1], x[0]))
    #users = input_rdd.map(lambda x: x[1]).distinct().collect()
    #num_buckets = len(users)
    #input_rdd = input_rdd.groupByKey().mapValues(lambda x: set(x))

    input_rdd_1 = sc.textFile(input_path)
    header = 'user_id,business_id,stars'
    input_rdd_2 = input_rdd_1.filter(lambda x: x != header).map(lambda x: x.split(",")[:2])
    input_rdd_3 = input_rdd_2.map(lambda x: (x[1], x[0]))
    users_1 = input_rdd_3.map(lambda x: x[1])
    users = users_1.distinct().collect()
    num_buckets = len(users)
    input_rdd_4 = input_rdd_3.groupByKey()
    input_rdd = input_rdd_4.mapValues(lambda x: set(x))




    hash_functions = hash_func(num_hashes)

    signatures = input_rdd.map(lambda x: create_sign(x, hash_functions, users, num_buckets))

    num_rows_ = num_rows
    num_bands = len(hash_functions) // num_rows_

    bands = signatures.flatMap(lambda x: create_bands(x, num_bands))

    candidate_pairs = bands.reduceByKey(lambda x, y: x + y).filter(
        lambda x: len(x[1]) > 1).values().flatMap(lambda x: getPairs(x)).distinct()

    vector_pairs = input_rdd.join(candidate_pairs)
    vector_pairs = vector_pairs.map(lambda x: (x[1][1], (x[0], x[1][0])))
    vector_pairs = vector_pairs.join(input_rdd)
    vector_pairs = vector_pairs.map(lambda x: ((x[1][0][0], x[0]), (x[1][0][1], x[1][1])))

    results = vector_pairs.map(lambda x: x[0] + jacc(x[1])).filter(lambda x: x[2] >= 0.5).sortBy(lambda x: (x[0], x[1]))

    res = results.toLocalIterator()

    end_time = time.time()
    print("Total time is :", end_time - start)

    with open(output_path, "w") as out_file:
        out_file.write("business_id_1, business_id_2, similarity\n")
        for result in res:
            out_file.write(f"{result[0]},{result[1]},{result[2]}\n")
        out_file.close()























