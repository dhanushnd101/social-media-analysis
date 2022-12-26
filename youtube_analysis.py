import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

youtube_data = pd.read_csv('video_result.csv')

#plt.figure()
#hist1,edges1 = np.histogram(youtube_data.viewCount)
#plt.bar(edges1[:-1],hist1,width=edges1[1:]-edges1[:-1])

print(youtube_data.corr())
plt.scatter(youtube_data.viewCount,youtube_data.likeCount)
y = youtube_data.likeCount
x = youtube_data.viewCount
x = sm.add_constant(x)

lr_model = sm.OLS(y,x).fit()
print(lr_model.summary())

xPrime = np.linspace(x.viewCount.min(),x.viewCount.max(),100)
xPrime = sm.add_constant(xPrime)

yHat = lr_model.predict(xPrime) 

plt.scatter( x.viewCount, y)
plt.xlabel('View Count')
plt.ylabel('Like Count')
plt.plot(xPrime[:,1],yHat)