import matplotlib.pyplot as plt
import pandas as pd

twitter_data = pd.read_csv('results_olympics1.csv', sep=',',  encoding='latin-1')

print(twitter_data.corr())
plt.scatter(twitter_data.retwc, twitter_data.polarity)