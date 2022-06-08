import math
import numpy as np
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# display full Dataframe without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

read_metadata_rating = pd.read_csv("ratings.csv", encoding="ISO-8859-1")
read_metadata_users = pd.read_csv("users.csv", encoding="ISO-8859-1")
read_metadata_books = pd.read_csv("books.csv", encoding="ISO-8859-1")

# create general data table "df":
data = {"book_id": read_metadata_books["book_id"]}
df = pd.DataFrame(data)
counts_per_book = read_metadata_rating.pivot_table(index=['book_id'], aggfunc='size')
df.insert(1, "counts", counts_per_book.values)
sum_rating_per_book = read_metadata_rating.groupby("book_id")["rating"].sum()
df.insert(2, "sum_rating", sum_rating_per_book.values)
df.insert(3, "average_rating", sum_rating_per_book.values/counts_per_book.values)
df.insert(4, "book_name", read_metadata_books["title"])

# general functions:
def create_dic_book_data(list_books_id, list_books_names):
    dic = {list_books_id[i]: list_books_names[i] for i in range(len(list_books_id))}
    return dic

# This function return minimum numbers of votes that needed in order to consider them in Non personalized recommenders
def weighted_average_rating(v, r, m, c):
    w_r = ((v / (v + m)) * r) + ((m / (v + m)) * c)
    return w_r

# This function return minimum numbers of votes that needed in order to consider them in Non personalized recommenders
def min_votes(df, title):
    min_v = df[title].quantile(0.9)
    return min_v

# keys are user id and values are the data that related to the user
def create_dic_users_data(list_users_id, list_data_on_users, size):
    dic = {list_users_id[i]: list_data_on_users[i] for i in range(size)}
    return dic

def cut_data_by_min_v(minimum_votes, metadata, title):
    q_metadata = metadata.copy().loc[metadata[title] >= minimum_votes]
    return q_metadata

def cut_data_by_string(string_data, df, title):
    cut_df = df.copy().loc[df[title] == string_data]
    return cut_df

def cut_data_by_range_num(x, y, metadata, title):
    q_metadata = metadata.copy().loc[(metadata[title] >= x) & (metadata[title] <= y)]
    return q_metadata

##### part 1 - non personalized #####

# dic that connect between book_id to book_name
dic_name_book = create_dic_users_data(read_metadata_books["book_id"],read_metadata_books["title"], len(read_metadata_books["book_id"]))

def get_simply_recommendation(k):
    k = int(k)
    average_rating_score = df["average_rating"].mean()
    m_vote = min_votes(df, "counts")
    cut_df = cut_data_by_min_v(m_vote, df, "counts")
    # add column of weighted_rating for each line
    cut_df["weighted_rating"] = cut_df.apply(lambda row: weighted_average_rating(row["counts"], row["average_rating"], m_vote, average_rating_score), axis=1)
    # sort by weighted rating
    cut_df = cut_df.sort_values("weighted_rating", ascending=False)
    print(cut_df[['book_id', 'book_name', 'counts', 'average_rating', 'weighted_rating']].head(k))

def get_simply_place_recommendation(place, k):
    k = int(k)
    data_location = {"user_id": read_metadata_rating["user_id"], "book_id": read_metadata_rating["book_id"], "rating": read_metadata_rating["rating"]}
    df_location = pd.DataFrame(data_location)
    dic_location_users = create_dic_users_data(read_metadata_users["user_id"], read_metadata_users["location"], max(read_metadata_rating["user_id"]))
    # for each book, we know the user that rate - and now we add the location of this user
    df_location["location"] = df_location.apply(lambda row: dic_location_users[row["user_id"]], axis=1)
    # cut all users that not match to the relevant place\location is given
    cut_df_location = cut_data_by_string(place, df_location, "location")
    # count all users from location that voted per book
    counts_per_book_loc = cut_df_location.pivot_table(index=['book_id'], aggfunc='size')
    data_loc_summery = {"book_id": counts_per_book_loc.index, "counts": counts_per_book_loc.values}
    df_loc_summery = pd.DataFrame(data_loc_summery)
    sum_rating_per_book_loc = cut_df_location.groupby("book_id")["rating"].sum()
    df_loc_summery.insert(2, "sum_rating", sum_rating_per_book_loc.values)
    df_loc_summery.insert(3, "average_rating", sum_rating_per_book_loc.values / counts_per_book_loc.values)
    df_loc_summery.insert(4, "book_name", lambda row: dic_name_book[row["book_id"]])
    df_loc_summery["book_name"] = df_loc_summery.apply(lambda row: dic_name_book[row["book_id"]], axis=1)
    average_rating_score = df_loc_summery["average_rating"].mean()
    m_vote = min_votes(df_loc_summery, "counts")
    df_loc_summery = cut_data_by_min_v(m_vote, df_loc_summery, "counts")
    df_loc_summery["score"] = df_loc_summery.apply(lambda row: weighted_average_rating(row["counts"], row["average_rating"], m_vote, average_rating_score), axis=1)
    # sort by weighted rating
    df_loc_summery = df_loc_summery.sort_values("score", ascending=False)
    print(df_loc_summery[['book_id', 'book_name', 'counts', 'average_rating', 'score']].head(k))

def get_simply_age_recommendation(age, k):
    age = int(age)
    range_x1 = age - age % 10 + 1
    range_y0 = range_x1 + 9
    data_location = {"user_id": read_metadata_rating["user_id"], "book_id": read_metadata_rating["book_id"], "rating": read_metadata_rating["rating"]}
    df_age = pd.DataFrame(data_location)
    dic_age_users = create_dic_users_data(read_metadata_users["user_id"], read_metadata_users["age"], max(read_metadata_rating["user_id"]))
    # for each book, we know the user that rate - and now we add the location of this user
    df_age["age"] = df_age.apply(lambda row: dic_age_users[row["user_id"]], axis=1)
    # cut all users that not match to the relevant age is given
    cut_df_location = cut_data_by_range_num(range_x1, range_y0, df_age, "age")
    # count all users from location that voted per book
    counts_per_book_age = cut_df_location.pivot_table(index=['book_id'], aggfunc='size')
    data_age_summery = {"book_id": counts_per_book_age.index, "counts": counts_per_book_age.values}
    df_age_summery = pd.DataFrame(data_age_summery)
    sum_rating_per_book_loc = cut_df_location.groupby("book_id")["rating"].sum()
    df_age_summery.insert(2, "sum_rating", sum_rating_per_book_loc.values)
    df_age_summery.insert(3, "average_rating", sum_rating_per_book_loc.values / counts_per_book_age.values)
    df_age_summery["book_name"] = df_age_summery.apply(lambda row: dic_name_book[row["book_id"]], axis=1)
    average_rating_score = df_age_summery["average_rating"].mean()
    m_vote = min_votes(df_age_summery, "counts")
    df_age_summery = cut_data_by_min_v(m_vote, df_age_summery, "counts")
    df_age_summery["score"] = df_age_summery.apply(lambda row: weighted_average_rating(row["counts"], row["average_rating"], m_vote, average_rating_score), axis=1)
    # sort by weighted rating
    df_age_summery = df_age_summery.sort_values("score", ascending=False)
    print(df_age_summery[['book_id', 'book_name', 'counts', 'average_rating', 'score']].head(k))

#check section
#get_simply_age_recommendation(18,10)

# general functions:

def create_dic_book_data(list_books_id, list_books_names):
    dic = {list_books_id[i]: list_books_names[i] for i in range(len(list_books_id))}
    return dic

def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    # replace anything lower than the cut off with 0
    arr[arr < smallest] = 0
    return arr

def get_key(val, dic):
    for key, value in dic.items():
        if val == value:
            return key

##### part 2 - Collaborative filtering user based#####

def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    arr[arr < smallest] = 0 # replace anything lower than the cut off with 0
    return arr

# create book_id dictionary "book_dict"
book_dict = {}
for i in range(len(read_metadata_books)):
    book_dict[read_metadata_books['book_id'][i]] = i

n_books = read_metadata_rating.book_id.unique().shape[0]
n_users = read_metadata_rating.user_id.unique().shape[0]

data_table = np.empty((n_users, n_books))
data_table[:] = np.nan
# create ranking data table
for line in read_metadata_rating.itertuples():
    user, book, rating = line[1]-1, book_dict[line[2]], line[3]
    data_table[user, book] = rating

# Return top k movies that the user did not see!
def get_recommendations(row_of_predicted_ratings, data_table_row, k):
    predicted_ratings_unrated = row_of_predicted_ratings
    predicted_ratings_unrated[~np.isnan((data_table_row))] = 0
    idx = np.argsort(-predicted_ratings_unrated)
    sim_scores = idx[0:k]
    dict_k_books = {}
    for sims in sim_scores:
        for x, v in book_dict.items():
            if sims == v:
                predicted_idx = x
                dict_k_books[predicted_idx] = dic_name_book[predicted_idx]
    top_k_books = pd.DataFrame(dict_k_books.items(), columns=["book_id", "title"])
    return top_k_books

def build_CF_prediction_matrix(sim):
    user_mean_rating = np.nanmean(data_table, axis=1).reshape(-1, 1)
    # calculate the mean
    rating_diff = (data_table - user_mean_rating)
    rating_diff[np.isnan(rating_diff)] = 0 # convert NaN to 0
    if (not sim == 'jaccard'):
        user_similarity = 1 - pairwise_distances(rating_diff, metric=sim)
    else: # sim == 'jaccard'
        user_similarity = 1 - pairwise_distances(np.array(rating_diff, dtype=bool), metric=sim)

    user_similarity = np.array([keep_top_k(np.array(arr), 10) for arr in user_similarity])
    prediction = user_mean_rating + user_similarity.dot(rating_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    return prediction

def get_CF_recommendation(user_id, k):
    # disclaimer: i asked Osnat about the "sim" value ,she told me to keep it as "cosine".
    prediction_matrix = build_CF_prediction_matrix('cosine')
    row_of_predicted_ratings = prediction_matrix[user_id-1]
    user_row_in_data_table = data_table[user_id-1]
    return get_recommendations(row_of_predicted_ratings, user_row_in_data_table, k)

# check section
# print(get_CF_recommendation(1,10))

##### part 3 - contact based filtering #####

metadata = pd.read_csv('books.csv', low_memory=False, encoding="ISO-8859-1")
# remove bad rows - as Osnat did in her code.
metadata = metadata.drop([1613])
# choose features name:
features_metadata = {"book_id": metadata["book_id"], "authors": metadata["authors"],
                     "original_publication_year": metadata["original_publication_year"],
                     "original_title": metadata["original_title"], "language_code": metadata["language_code"]}
df_features_metadata = pd.DataFrame(features_metadata)

def create_soup(x):
    return ' ' + ''.join(str(x['authors'])) + ' ' + ''.join(str(x['original_publication_year'])) + ' '\
           + ''.join(str(x['original_title'])) + ' ' + ''.join(str(x['language_code']))

# Create a new soup feature "df_features_metadata"
with pd.option_context('mode.chained_assignment', None):
    df_features_metadata['soup'] = df_features_metadata.apply(create_soup, axis=1)

def build_contact_sim_metrix():
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df_features_metadata['soup'])
    # Compute the Cosine Similarity matrix based on the count_matrix
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim2

# Reset index of your main DataFrame and construct reverse mapping as before
df_features_metadata = df_features_metadata.reset_index()
indices = pd.Series(df_features_metadata.index, index=df_features_metadata['original_title']).drop_duplicates()

# Function that takes in book title as input and outputs most similar books
def get_contact_recommendation(book_name, k):
    # Get the pairwsie similarity scores of all books with that book
    idx = indices[book_name]
    if (idx.size > 1):
        idx = idx[0]
    cosine_sim = build_contact_sim_metrix()
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar books (the first is the movie we asked)
    sim_scores = sim_scores[1:k + 1]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar books
    print(df_features_metadata['original_title'].iloc[book_indices])

# check section
#get_contact_recommendation("Twilight", 10)

##### part 4 - evaluation functions #####
# read as latin style
test_metadata = pd.read_csv("test.csv", encoding="latin-1")

def RMSE():
    pred = build_CF_prediction_matrix('cosine')
    up_fraction_sum = 0
    # check if real rating == predicted rating
    for index, row in test_metadata.iterrows():
        predicted_rating, real_rating = pred[row['user_id']-1, book_dict[row['book_id']]], row['rating']
        up_fraction_sum += math.pow((predicted_rating-real_rating), 2)
    cosine_ans = math.sqrt(up_fraction_sum/len(test_metadata))

    pred = build_CF_prediction_matrix('euclidean')
    up_fraction_sum = 0
    # check if real rating == predicted rating
    for index, row in test_metadata.iterrows():
        predicted_rating, real_rating = pred[row['user_id'] - 1, book_dict[row['book_id']]], row['rating']
        up_fraction_sum += math.pow((predicted_rating - real_rating), 2)
    euclidean_ans = math.sqrt(up_fraction_sum / len(test_metadata))

    pred = build_CF_prediction_matrix('jaccard')
    up_fraction_sum = 0
    # check if real rating == predicted rating
    for index, row in test_metadata.iterrows():
        predicted_rating, real_rating = pred[row['user_id'] - 1, book_dict[row['book_id']]], row['rating']
        up_fraction_sum += math.pow((predicted_rating - real_rating), 2)
    jaccard_ans = math.sqrt(up_fraction_sum / len(test_metadata))

    ans_list = []
    ans_list.append(cosine_ans)
    ans_list.append(euclidean_ans)
    ans_list.append(jaccard_ans)
    return (ans_list)

# helper function to precision_k and ARHR
def hits_counter(k_recommand, k_test):
    hits = []
    for test in k_test:
        for index, rec in k_recommand.iterrows():
            if (test[1] == rec['book_id']):
                hits.append(rec['book_id'])
    return hits

def precision_k(k):
    # filter only 4,5 ratings
    test45 = test_metadata.copy().loc[test_metadata["rating"] >= 4]
    test45 = test45.sort_values('user_id', ascending=False)
    user_votes_counter = test45['user_id'].value_counts()

    # save users with 10 or more votes
    hits, num_of_users = 0, 0
    for user in user_votes_counter.iteritems():
        if (user[1] >= 10 ):
            relevant_users_file = test45.copy().loc[test45["user_id"] == user[0]]
            results = list(relevant_users_file.to_records(index=False))
            # recommendation for the checked user:
            prediction_matrix = build_CF_prediction_matrix('cosine')
            row_of_predicted_ratings = prediction_matrix[user[0] - 1]
            user_row_in_data_table = data_table[user[0] - 1]
            k_recommands = get_recommendations(row_of_predicted_ratings, user_row_in_data_table, k)
            # count the hits = correct predictions
            hits += len(hits_counter(k_recommands, results))
            num_of_users = 1 + num_of_users
    cosine_ans = hits/(k*num_of_users)

    # save users with 10 or more votes
    hits, num_of_users = 0, 0
    for user in user_votes_counter.iteritems():
        if (user[1] >= 10 ):
            relevant_users_file = test45.copy().loc[test45["user_id"] == user[0]]
            results = list(relevant_users_file.to_records(index=False))
            # recommendation for the checked user:
            prediction_matrix = build_CF_prediction_matrix('euclidean')
            row_of_predicted_ratings = prediction_matrix[user[0] - 1]
            user_row_in_data_table = data_table[user[0] - 1]
            k_recommands = get_recommendations(row_of_predicted_ratings, user_row_in_data_table, k)
            # count the hits = correct predictions
            hits += len(hits_counter(k_recommands, results))
            num_of_users = 1 + num_of_users
    euclidean_ans = hits/(k*num_of_users)

    # save users with 10 or more votes
    hits, num_of_users = 0, 0
    for user in user_votes_counter.iteritems():
        if (user[1] >= 10 ):
            relevant_users_file = test45.copy().loc[test45["user_id"] == user[0]]
            results = list(relevant_users_file.to_records(index=False))
            # recommendation for the checked user:
            prediction_matrix = build_CF_prediction_matrix('jaccard')
            row_of_predicted_ratings = prediction_matrix[user[0] - 1]
            user_row_in_data_table = data_table[user[0] - 1]
            k_recommands = get_recommendations(row_of_predicted_ratings, user_row_in_data_table, k)
            # count the hits = correct predictions
            hits += len(hits_counter(k_recommands, results))
            num_of_users = 1 + num_of_users
    jaccard_ans = hits/(k*num_of_users)

    ans_list = []
    ans_list.append(cosine_ans)
    ans_list.append(euclidean_ans)
    ans_list.append(jaccard_ans)
    return (ans_list)

# helper function to ARHR
def list_convert(df):
    list = []
    for index, rec in df.iterrows():
        list.append(rec['book_id'])
    return list

def ARHR(k):
    # filter only 4,5 ratings
    test45 = test_metadata.copy().loc[test_metadata["rating"] >= 4]
    test45 = test45.sort_values('user_id', ascending=False)
    user_votes_counter = test45['user_id'].value_counts()
    # save users with 10 or more votes
    up_fraction_sum, num_of_users = 0, 0
    hits = [] # hits is a dictionary this time
    for user in user_votes_counter.iteritems():
        if (user[1] >= 10):
            only_relevant_users = test45.copy().loc[test45["user_id"] == user[0]]
            prediction_matrix = build_CF_prediction_matrix('cosine')
            row_of_predicted_ratings = prediction_matrix[user[0] - 1]
            user_row_in_data_table = data_table[user[0] - 1]
            # recommendation for the checked user:
            k_recommand = get_recommendations(row_of_predicted_ratings, user_row_in_data_table, k)
            k_recommands_list = list_convert(k_recommand)
            # count the hits = correct predictions
            hits = hits_counter(k_recommand, list(only_relevant_users.to_records(index=False)))
            for i_hit in range(len(hits)):
                for j in range(k):
                    if (hits[i_hit] == k_recommands_list[j]):
                        up_fraction_sum += 1 / (j+1)
            num_of_users += 1
    cosine_ans = up_fraction_sum/num_of_users

    # save users with 10 or more votes
    up_fraction_sum, num_of_users = 0, 0
    hits = []  # hits is a dictionary this time
    for user in user_votes_counter.iteritems():
        if (user[1] >= 10):
            only_relevant_users = test45.copy().loc[test45["user_id"] == user[0]]
            prediction_matrix = build_CF_prediction_matrix('euclidean')
            row_of_predicted_ratings = prediction_matrix[user[0] - 1]
            user_row_in_data_table = data_table[user[0] - 1]
            # recommendation for the checked user:
            k_recommand = get_recommendations(row_of_predicted_ratings, user_row_in_data_table, k)
            k_recommands_list = list_convert(k_recommand)
            # count the hits = correct predictions
            hits = hits_counter(k_recommand, list(only_relevant_users.to_records(index=False)))
            for i_hit in range(len(hits)):
                for j in range(k):
                    if (hits[i_hit] == k_recommands_list[j]):
                        up_fraction_sum += 1 / (j + 1)
            num_of_users += 1
    euclidean_ans = up_fraction_sum / num_of_users

    # save users with 10 or more votes
    up_fraction_sum, num_of_users = 0, 0
    hits = []  # hits is a dictionary this time
    for user in user_votes_counter.iteritems():
        if (user[1] >= 10):
            only_relevant_users = test45.copy().loc[test45["user_id"] == user[0]]
            prediction_matrix = build_CF_prediction_matrix('jaccard')
            row_of_predicted_ratings = prediction_matrix[user[0] - 1]
            user_row_in_data_table = data_table[user[0] - 1]
            # recommendation for the checked user:
            k_recommand = get_recommendations(row_of_predicted_ratings, user_row_in_data_table, k)
            k_recommands_list = list_convert(k_recommand)
            # count the hits = correct predictions
            hits = hits_counter(k_recommand, list(only_relevant_users.to_records(index=False)))
            for i_hit in range(len(hits)):
                for j in range(k):
                    if (hits[i_hit] == k_recommands_list[j]):
                        up_fraction_sum += 1 / (j + 1)
            num_of_users += 1
    jaccard_ans = up_fraction_sum / num_of_users

    ans_list = []
    ans_list.append(cosine_ans)
    ans_list.append(euclidean_ans)
    ans_list.append(jaccard_ans)
    return (ans_list)
