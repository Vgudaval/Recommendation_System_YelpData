import time
import math
from pyspark import SparkContext
import operator
import sys, os

if __name__ == '__main__':
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


    def pearson_sim(bus, tested_val, tested_average, val_tested):

        curr = []
        neighbors = []

        for review in val_tested:
            if review in train_data_business[bus]:
                neighbors.append(train_data_business[bus][review])
                curr.append(tested_val[review])

        if len(neighbors) > 0 and len(curr) > 0:
            numerator = 0
            curr_least = 0
            neighbor_least = 0

            for i in range(len(curr)):
                curr_1 = curr[i]
                neighbor_1 = neighbors[i]
                numerator += (curr_1 - tested_average) * (neighbor_1 - avg_bus[bus])
                curr_least += (curr_1 - tested_average) ** 2
                neighbor_least += (neighbor_1 - avg_bus[bus]) ** 2

            denominator = (curr_least ** 0.5) * (neighbor_least ** 0.5)

            x = num_denom(denominator, numerator)

            return x

        else:
            y = tested_average / avg_bus[bus]

            return y


    def num_denom(denominator, numerator):
        if denominator == 0:

            if numerator == 0:

                return 1
            else:
                return 0
        else:

            return numerator / denominator


    def item_based_cf(user, business):

        if business not in train_data_business and user not in train_data_user:
            return user, business, 2.5

        elif business not in train_data_business and user in train_data_user:
            return user, business, avg_user[user]

        # if both business and user exist then find using pearson correlation
        elif business in train_data_business and user in train_data_user:

            val_tested = list(train_data_business[business])
            business_tested_vals = list(train_data_user[user])

            tested_val = train_data_business[business]
            tested_average = avg_bus[business]

            subset = sim(business_tested_vals, tested_val, tested_average, val_tested, user)



            numerator_rating = 0
            denominator_rating = 0

            if len(subset) > 0:
                for review, weight in subset:
                    denominator_rating += math.fabs(weight)
                    numerator_rating += review * weight
                rating_given = min(max(numerator_rating / denominator_rating, 1.0), 5.0)
                return user, business, rating_given

            elif len(subset) < 0:
                return user, business, tested_average

        elif business in train_data_business and user not in train_data_business:
            return user, business, avg_bus[business]

    def sim(business_tested_vals, tested_val, tested_average, val_tested, user):
        similarities = []

        for bus in business_tested_vals:

            pearson_coeff = pearson_sim(bus, tested_val, tested_average, val_tested)
            bus_rating_by_user = train_data_business[bus][user]

            if pearson_coeff > 0:
                if pearson_coeff > 1:
                    pearson_coeff = 1 / pearson_coeff
                similarities.append((bus_rating_by_user, pearson_coeff))

        if len(similarities) > 60:
            subset = similarities[:60]

        else:
            subset = similarities

        return subset



    #input_file = "yelp_train.csv"
    #val_file = "yelp_val.csv"
    #output_file = "task2_1_modified_op.csv"

    input_file = sys.argv[1]
    val_file = sys.argv[2]
    output_file = sys.argv[3]

    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    start = time.time()

    train_file = sc.textFile(input_file)
    train_file_data = train_file.filter(lambda row: row != 'user_id,business_id,stars')
    train_file_data = train_file_data.map(lambda user: user.split(","))

    train_data_user = train_file_data.map(lambda user: (user[0], (user[1], float(user[2]))))
    train_data_user = train_data_user.groupByKey().sortByKey()
    train_data_user = train_data_user.mapValues(dict).collectAsMap()

    train_data_business = train_file_data.map(lambda pair: (pair[1], (pair[0], float(pair[2]))))
    train_data_business = train_data_business.groupByKey().sortByKey()
    train_data_business = train_data_business.mapValues(dict).collectAsMap()

    avg_user = train_file_data.map(lambda row: (row[0], float(row[2])))
    avg_user = avg_user.groupByKey()
    avg_user = avg_user.mapValues(lambda row: sum(row) / len(row))
    avg_user = avg_user.collectAsMap()

    avg_bus = train_file_data.map(lambda row: (row[1], float(row[2])))
    avg_bus = avg_bus.groupByKey()
    avg_bus = avg_bus.mapValues(lambda row: sum(row) / len(row))
    avg_bus = avg_bus.collectAsMap()

    validationRDD = sc.textFile(val_file)
    accuracyRDD = validationRDD.filter(lambda row: row != 'user_id,business_id,stars')
    accuracyRDD = accuracyRDD.map(lambda row: row.split(","))
    predictions = accuracyRDD.map(lambda row: item_based_cf(row[0], row[1]))
    predictions = predictions.collect()

    # writing to output file

    with open(output_file, 'w+') as outfile:

        outfile.write("user_id, business_id, prediction\n")

        for each in predictions:
            row = each[0] + "," + each[1] + "," + str(each[2]) + "\n"

            outfile.write(row)

    outfile.close()

    end = time.time()
    duration = end - start

    print("Duration: ", duration)



























