
import pandas as pd
import numpy as np

# Here is my working version of the code that does not have the proper output (meaning the code still uses the movieId and the output is not int he right format).
# This code is based off the example code provided by the professor.
# I may resubmit or send a new, working copy later to the professor (not looking for an improvement to the grade) to show improvement to the assignment.


# from data sets: only need raitngs.csv and movies.csv
movies = pd.read_csv('./movies.csv')
ratings = pd.read_csv('./ratings.csv')
# Join ratigs and movies on movieId
temp = pd.merge(ratings, movies, on="movieId")
# Pivot temp to have rows based on userIds
print("creating movie matrix")
movieMatrix = temp.pivot_table(
    index='userId', columns='title', values='rating')
# Run the pearson correlation on the new pivoted table to get the centered cosine similarity

print("creating correlation matrix")
# create the correlation matrix (centered cosine similarity; pearson's)
corrMatrix = movieMatrix.corr(method='pearson', min_periods=5)

# while the output file is open
with open("output.txt", 'w') as dataout:
    # for every row in movieMatrix (every user)
    for i in range(1, len(movieMatrix)):
        # remove na's
        userRatings = movieMatrix.iloc[i].dropna()
        # create in individual recommendations list for the user
        recommended = pd.Series()
        # for every column (movies)
        for j in range(0, len(userRatings)):
            # drop na values
            sim = corrMatrix[userRatings.index[j]].dropna()
            # multiply rating by user raitng
            sim = sim.map(lambda x: x * userRatings[j])
        # append to the recommendations list
        recommended = recommended.append(sim)
        # Sort recommendations on ratings
        recommended.sort_values(inplace=True, ascending=False)
        # write out user info
        dataout.write('\n' + str(i) + '\n')
        # write out top 5 recommendations
        dataout.write(recommended.head(5).to_string())
