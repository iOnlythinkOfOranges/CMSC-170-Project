# PSEUDOCODE FOR GAUSSIAN NAIVE BAYES

# GOAL:    Given the last N days of weather, predict whether it will rain tomorrow (1) or stay dry (0).

# INPUT:   X = a table of numbers. Each ROW is one example (the weather of the last N days). 
#           Each COLUMN is one feature (e.g., humidity on day 1). Values are already normalized.
#          y = a list of answers, one per row of X. (1 = it rained the next day, 0 = it did not)

# OUTPUT:  the probability of rain (a number from 0 to 1),
#          and a final prediction: 1 (rain) or 0 (dry).



# 1: FIT (learning from the training data)

# PURPOSE: Learn what "typical rainy-before" and "typical dry-before" weather looks like. 
#           "Learning" here only means calculating averages. There is no repeated training loop.
# NOTE:    Only use the TRAINING data here (never the validation or test data). (THE SPLIT IS YET TO BE FINALIZED)

# FIT(X_train, y_train):

#     for each class c in [0, 1]:
#     # Do everything below once for dry (0), then once for rain (1)

#         days_c = the rows of X_train where y_train equals c
#         # collect only the days that belong to this class. (ex. for c = 1, keep only days followed by rain)

#         prior[c] = (number of days in days_c) / (total number of training days)
#         # PRIOR: how common this class is in the training data.
#         # Lab version: P(Spam) and P(Ham). Our version: P(Rain) and P(Dry). (ex. 300 rainy days out of 1000 -> prior[1] = 0.30)

#         for each feature j (each column of X):
#         # repeat for every feature, such as humidity, pressure, etc

#             mean[c][j] = average value of feature j across days_c
#             # the "typical" value of this feature for this class (ex. average humidity before rainy days)

#             var[c][j] = variance of feature j across days_c
#             # variance = how spread out the values are around the mean. (The smoothing is added after this loop.)

#     smoothing = 0.000000001 * (the largest value among all var[c][j])
#     # a tiny number based on the biggest variance. (purpose -> to make sure we never divide by zero later)

#     for each class c and each feature j:
#         var[c][j] = var[c][j] + smoothing
#         # add the tiny number to every variance.


# 2: SCORE (how well does one day fit one class?)

# PURPOSE: Give a score showing how well a day's weather matches a class. Higher score = better match.

# SCORE(x, c):
# # x = one day's features (a list of numbers)
# # c = the class we are checking (0 = dry, 1 = rain)

#     score = log(prior[c])
#     # start with how common the class is. 
#     # use log() because we will ADD scores instead of multiplying probabilities (multiplying many tiny numbers can round down to 0)

#     for each feature j:
#     # look at the features one at a time. !! This is the "naive" part, we treat each feature as separate from the others

#         score = score
#                 - 0.5 * log(2 * pi * var[c][j])
#                 - (x[j] - mean[c][j])^2 / (2 * var[c][j])
#         # This line is the bell-curve (Gaussian) formula, written in log form. In plain words:
#         #   - If x[j] is CLOSE to the class's mean[c][j], the second part (the squared difference) is small, so the score stays high.
#         #   - If x[j] is FAR from the mean, that part gets large and the score drops.
#         #   - Dividing by var[c][j] means a feature that varies a lot is allowed to be further from the mean.

#     return score
#     # a single number (how well this day fits class c)


# BLOCK 3: PREDICT_PROBA (turn two scores into a probability)

# PURPOSE: Compare the "dry" score and the "rain" score, and convert them into a probability of rain.

# PREDICT_PROBA(x):

#     s0 = SCORE(x, 0)
#     # how well this day fits "dry"

#     s1 = SCORE(x, 1)
#     # how well this day fits "rain"

#     m = the larger of s0 and s1
#     # helper. The scores are large negative numbers, and computing the exp() of them directly can round to 0.
#     # Subtracting the larger score keeps the numbers safe.

#     p_rain = exp(s1 - m) / ( exp(s0 - m) + exp(s1 - m) )
#     # turns the two scores into a share of 100% (ex. if rain fits 3x better than dry, p_rain = 0.75)

#     return p_rain
#     # probability between 0 and 1


# BLOCK 4: PREDICT (make the final yes/no decision)

# PURPOSE: Convert the probability into "rain" or "dry".

# PREDICT(x, threshold = 0.5):
# # threshold = how sure we must be before saying "rain". (0.5 -> placeholder lang pede pa paltan)
#     if PREDICT_PROBA(x) >= threshold:
#         return 1   # predict RAIN
#     else:
#         return 0   # predict DRY