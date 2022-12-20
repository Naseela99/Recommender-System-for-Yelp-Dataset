"""
Model Description:

Used XGBoostRegressor. Did extensive feature engineering to get the best results possible.

Error Distribution:                                                                                                                                                                                                                                     
>=0 and <1: 101715                                                                                                                                                                                                                                     
>=1 and <2: 32260                                                                                                                                                                                                                                      
>=2 and <3: 6772                                                                                                                                                                                                                                       
>=3 and <4: 1269  

RMSE:
1.0096107719303637

Execution time:
Approx 600s

"""

from collections import Counter
from datetime import datetime
from pyspark import SparkContext
import time
import sys
from math import sqrt

from time import perf_counter
import os
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor
import pandas as pd
from scipy.stats import kurtosis, skew


def preprocess(data_rdd, business_rdd, user_rdd):

    # join and set ("user, business") as key
    business_features = data_rdd.map(lambda x:x[::-1]).leftOuterJoin(business_rdd).map(lambda x:((x[1][0],x[0]),x[1][1]))
    user_features = data_rdd.leftOuterJoin(user_rdd).map(lambda x:((x[0], x[1][0]),x[1][1]))

    ## combine rdd, with key ("user, business") as key and value as a list of features
    combined_features = business_features.join(user_features).map(lambda x:(x[0],x[1][0]+x[1][1]))
    return combined_features


def load_business_rdd(business_path, tips_path, photos_path, checkin_path):

    def preprocess_business(line):
        return_data = {}
        business = json.loads(line)
        attributes = business["attributes"]
        if attributes is None:
            attributes = {}

        business_parking = eval(attributes.get("BusinessParking", "{}"))
        if business_parking is None:
            business_parking = {}
        
        for col in attribute_cols:
            return_data[col] = attributes.get(col, None)

        longitude = business["longitude"]
        latitude = business["latitude"]
        return_data.update({"longitude": longitude, "latitude": latitude})
        stars = business["stars"]
        review_count = business["review_count"]
        return_data.update({"stars": stars, "review_count": review_count})

        likes = business2likes.get(business["business_id"], {"likes":-1, "total_tips":-1})
        return_data.update(likes)

        labels = business2labels.get(business["business_id"], {"food":0, "outside":0, "drink":0, "menu":0, "inside":0, "sum":0})
        return_data.update(labels)

        checkin_data = business2checkin.get(business["business_id"], {"checkin_total":0, "checkin_count":0})
        return_data.update(checkin_data)

        rating_stats = business2rating.get(business["business_id"], global_business_ratings)
        return_data.update(rating_stats)

        return_data.update({"business_id": business["business_id"]})
        return return_data



    business_rdd = sc.textFile(business_path)
    
    tips_rdd = sc.textFile(tips_path).map(json.loads)
    business2likes = tips_rdd.map(lambda x:(x["business_id"], [x["likes"], 1])).reduceByKey(lambda x,y:(x[0]+y[0], x[1]+y[1])).map(lambda x:(x[0], {"likes":x[1][0], "total_tips":x[1][1]})).collectAsMap()

    photos_rdd = sc.textFile(photos_path).map(json.loads)
    interested_labels = ['food', 'outside', 'drink', 'menu', 'inside']
    def cnt_interesting_labels(labels):
        cntr = Counter({i:0 for i in interested_labels})
        labels = [i for i in labels if i in interested_labels]
        cntr.update( Counter(labels))
        cntr["sum"] = sum(cntr.values())
        return dict(cntr)
    business2labels = photos_rdd.map(lambda x:(x["business_id"], x["label"])).groupByKey().map(lambda x:(x[0], cnt_interesting_labels(x[1]))).collectAsMap()

    checkin_rdd = sc.textFile(checkin_path).map(json.loads).map(lambda x:(x["business_id"], x["time"]))
    business2checkin = checkin_rdd.map(lambda x:(x[0], (sum(x[1].values()), len(x[1])))).map(lambda x:(x[0], {"checkin_total":x[1][0], "checkin_count":x[1][1]})).collectAsMap()
    
    
    business_rdd = business_rdd.map(preprocess_business)
    business_rdd_columns = list(business_rdd.first().keys())
    business_rdd_columns.remove("business_id")
    business_rdd = business_rdd.map(lambda x: (x["business_id"], [x[column] for column in business_rdd_columns]))
    return business_rdd, ["business_{}".format(i) for i in business_rdd_columns]


def load_user_rdd(user_path):

    def preprocess_user(line):
        user = json.loads(line)
        return_data = {}
        friends = user["friends"]
        if friends is None:
            friends = []

        elite = user["elite"]
        if elite is None:
            elite = []

        useful = user["useful"]
        funny = user["funny"]
        cool = user["cool"]

        friends = len(friends)
        elite = len(elite)
        fans = user["fans"]
        average_stars = user["average_stars"]
        review_count = user["review_count"]
        yelping_since = user["yelping_since"]
        yelping_since = datetime.strptime(yelping_since, "%Y-%m-%d")
        yelping_since = (datetime.now() - yelping_since)
        ## convert to months
        yelping_since = yelping_since.days / 30.0
        return_data.update({"friends": friends, "elite": elite, "useful": useful, "funny": funny, "cool": cool, "fans": fans, "average_stars": average_stars, "review_count": review_count, "yelping_since": yelping_since})

        compliment_hot = user["compliment_hot"]
        compliment_more = user["compliment_more"]
        compliment_profile = user["compliment_profile"]
        compliment_cute = user["compliment_cute"]
        compliment_list = user["compliment_list"]
        compliment_note = user["compliment_note"]
        compliment_plain = user["compliment_plain"]
        compliment_cool = user["compliment_cool"]
        compliment_funny = user["compliment_funny"]
        compliment_writer = user["compliment_writer"]
        compliment_photos = user["compliment_photos"]
        return_data.update({"compliment_hot": compliment_hot, "compliment_more": compliment_more, "compliment_profile": compliment_profile, "compliment_cute": compliment_cute, "compliment_list": compliment_list, "compliment_note": compliment_note, "compliment_plain": compliment_plain, "compliment_cool": compliment_cool, "compliment_funny": compliment_funny, "compliment_writer": compliment_writer, "compliment_photos": compliment_photos})


        rating_stats = user2rating.get(user["user_id"], global_user_ratings)
        return_data.update(rating_stats)

        return_data.update({"user_id": user["user_id"]})


        return return_data

    user_rdd = sc.textFile(user_path)
    user_rdd = user_rdd.map(preprocess_user)
    user_rdd_columns = list(user_rdd.first().keys())
    user_rdd_columns.remove("user_id")
    user_rdd = user_rdd.map(lambda x: (x["user_id"], [x[column] for column in user_rdd_columns]))
    return user_rdd, ["user_{}".format(i) for i in user_rdd_columns]

if __name__ == "__main__":
    
    st= perf_counter()
   
    sc = SparkContext().getOrCreate()
    
    sc.setLogLevel('WARN')

    folder_path,test_file,output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    
    train_file = os.path.join(folder_path, "yelp_train.csv")
    val_file = os.path.join(folder_path, "yelp_val.csv")
    train_data  = sc.textFile(train_file).filter(lambda x: x!= 'user_id,business_id,stars').map(lambda x: x.split(','))
    val_data = sc.textFile(val_file).filter(lambda x: x!= 'user_id,business_id,stars').map(lambda x: x.split(','))


    business2rating = train_data.map(lambda x: (x[1], float(x[2]))).groupByKey().map(lambda x: (x[0], {'mean': np.mean(list(x[1])), 'std':np.std(list(x[1])), 'count':len(list(x[1])), 'min':np.min(list(x[1])), 'max':np.max(list(x[1])), 'kur':kurtosis(list(x[1])), 'skew':skew(list(x[1]))})).collectAsMap()
    user2rating = train_data.map(lambda x: (x[0], float(x[2]))).groupByKey().map(lambda x: (x[0], {'mean': np.mean(list(x[1])), 'std':np.std(list(x[1])), 'count':len(list(x[1])), 'min':np.min(list(x[1])), 'max':np.max(list(x[1])), 'kur':kurtosis(list(x[1])), 'skew':skew(list(x[1]))})).collectAsMap()

    global_user_ratings = {'mean': np.mean([i['mean'] for i in user2rating.values()]), 
                            'std':np.mean([i['std'] for i in user2rating.values()]), 
                            'count':np.mean([i['count'] for i in user2rating.values()]), 
                            'min':np.mean([i['min'] for i in user2rating.values()]), 
                            'max':np.mean([i['max'] for i in user2rating.values()]),
                            'kur':np.mean([i['kur'] for i in user2rating.values()]),
                            'skew':np.mean([i['skew'] for i in user2rating.values()])}
    global_business_ratings = {'mean': np.mean([i['mean'] for i in business2rating.values()]), 
                                'std':np.mean([i['std'] for i in business2rating.values()]), 
                                'count':np.mean([i['count'] for i in business2rating.values()]), 
                                'min':np.mean([i['min'] for i in business2rating.values()]), 
                                'max':np.mean([i['max'] for i in business2rating.values()]),
                                'kur':np.mean([i['kur'] for i in business2rating.values()]),
                                'skew':np.mean([i['skew'] for i in business2rating.values()])}



    attribute_cols = ['BikeParking', 'BusinessAcceptsCreditCards', 'GoodForKids', 
                    'HasTV', 'NoiseLevel', 'OutdoorSeating', 'RestaurantsAttire', 'RestaurantsDelivery', 'RestaurantsGoodForGroups', 'RestaurantsPriceRange2', 'RestaurantsReservations', 'RestaurantsTakeOut', 'Alcohol', 'Caters', 'RestaurantsTableService', 'WheelchairAccessible', 'WiFi', 'ByAppointmentOnly']

    test_data = sc.textFile(test_file).filter(lambda x: x!= 'user_id,business_id,stars').map(lambda x: x.split(','))

    # train_y = train_data.map(lambda x: float(x[2]))
    train_y = train_data.map(lambda x: ((x[0],x[1]), float(x[2]))).collectAsMap()
    train_x = train_data.map(lambda x: (x[0],x[1]))

    val_y = val_data.map(lambda x: ((x[0],x[1]), float(x[2]))).collectAsMap()
    val_x = val_data.map(lambda x: (x[0],x[1]))


    test_x = test_data.map(lambda x: (x[0],x[1]))

    business_file = os.path.join(folder_path, "business.json")
    tips_file = os.path.join(folder_path, "tip.json")
    photos_file = os.path.join(folder_path, "photo.json")
    checkin_path = os.path.join(folder_path, "checkin.json")

    business_rdd, business_cols = load_business_rdd(business_file, tips_file, photos_file, checkin_path)

    user_file = os.path.join(folder_path, "user.json")

    user_rdd, user_cols = load_user_rdd(user_file)

    train_ftrs = preprocess(train_x, business_rdd, user_rdd)
    train_user_biz = train_ftrs.map(lambda x: x[0]).collect()
    train_ftrs = train_ftrs.map(lambda x: x[1]).collect()

    val_ftrs = preprocess(val_x, business_rdd, user_rdd)
    val_user_biz = val_ftrs.map(lambda x: x[0]).collect()
    val_ftrs = val_ftrs.map(lambda x: x[1]).collect()
    
    new_train_y = []
    for i in train_user_biz:
        new_train_y.append(train_y[i])
    train_y = new_train_y

    new_val_y = []
    for i in val_user_biz:
        new_val_y.append(val_y[i])
    val_y = new_val_y


    train_df = pd.DataFrame(train_ftrs, columns = business_cols + user_cols)
    train_df = pd.get_dummies(train_df, columns = ["business_{}".format(i) for i in attribute_cols], dummy_na=True, drop_first=True)
    df_cols = train_df.columns
    train_df = train_df.fillna(0)

    val_df = pd.DataFrame(val_ftrs, columns = business_cols + user_cols)
    val_df = pd.get_dummies(val_df, columns = ["business_{}".format(i) for i in attribute_cols], dummy_na=True, drop_first=True)
    for col in df_cols:
        if col not in val_df.columns:
            val_df[col] = 0
    val_df = val_df[df_cols]

    val_df = val_df.fillna(0)


    train_df = train_df.values
    train_y = np.array(train_y)
    
    val_df = val_df.values
    val_y = np.array(val_y)

    test_ftrs = preprocess(test_x, business_rdd, user_rdd)
    test_user_biz = test_ftrs.map(lambda x: x[0]).collect()
    test_ftrs = test_ftrs.map(lambda x: x[1]).collect()

    test_df = pd.DataFrame(test_ftrs, columns = business_cols + user_cols)

    test_df = pd.get_dummies(test_df, columns = ["business_{}".format(i) for i in attribute_cols], dummy_na=True, drop_first=True)
    for col in df_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    test_df = test_df[df_cols]

    test_df = test_df.fillna(0)

    test_df = test_df.values

    model = XGBRegressor(
        max_depth=5,
        min_child_weight=1,
        subsample=0.6,
        colsample_bytree=0.6,
        gamma=0,
        reg_alpha=1,
        reg_lambda=0,
        learning_rate=0.05,
        n_estimators=700,
        verbosity=3,
    )
    model.fit(train_df, train_y, early_stopping_rounds=50, eval_set=[(val_df, val_y)],verbose=50)
    print("TRAINED")
    
    
    pred_outs = model.predict(test_df)
    train_outs = model.predict(train_df)
    val_outs = model.predict(val_df)
    
    train_mse = mean_squared_error(train_y, train_outs)
    val_mse = mean_squared_error(val_y, val_outs)
    absolute_diff = np.abs(np.array(val_y) - np.array(val_outs))
    print("TRAIN MSE: {}".format(train_mse))
    print("VAL MSE: {}".format(val_mse))
    print("Error Distribution")
    print(">=0 and <1: {}".format(np.sum(absolute_diff < 1)))
    print(">=1 and <2: {}".format(np.sum((absolute_diff < 2) & (absolute_diff >= 1))))
    print(">=2 and <3: {}".format(np.sum((absolute_diff < 3) & (absolute_diff >= 2))))
    print(">=3 and <4: {}".format(np.sum((absolute_diff < 4) & (absolute_diff >= 3))))
    print(">=4: {}".format(np.sum(absolute_diff >= 4)))
    output_str = "user_id, business_id, prediction\n"
    for i in range(len(pred_outs)):
        output_str += "{},{},{}\n".format(test_user_biz[i][0], test_user_biz[i][1], pred_outs[i])
    with open(output_file, "w") as f:
        f.write(output_str)
    print("DONE")
