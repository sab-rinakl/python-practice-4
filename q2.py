import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot

# create statistical summary
commute_data = pd.read_csv('CommuteStLouis.csv', usecols=['Age', 'Distance', 'Time'])
summary = commute_data.describe()
print(summary, '\n')

# plot histogram
plt.hist(commute_data['Age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Freq')
plt.title('Histogram of Age')
plt.show()

# create correlation matrix
matrix = commute_data.corr()
print(matrix)
print("Highly correlated: Time and Distance")
print("Correlation coefficient: 0.830241")
print("")

# create scatterplot matrix
pd.plotting.scatter_matrix(commute_data, hist_kwds={'bins': 10})
plt.show()
print("The figures in the diagonal are histograms of age, distance, and time, like the histogram of time from part 1.")
print("The skew shows how there are much more younger divers in the data than older drivers.")
print("")

# create side-by-side boxplot
boxplot_data = pd.read_csv('CommuteStLouis.csv')
boxplot_data = boxplot_data.groupby('Sex')
m_data = boxplot_data.get_group('M')['Distance'].values
f_data = boxplot_data.get_group('F')['Distance'].values
plt.boxplot(m_data, patch_artist=True, widths=.8, positions=[1], boxprops=dict(color='blue'))
plt.boxplot(f_data, patch_artist=True, widths=.8, positions=[2], boxprops=dict(color='orange'))
plt.xlabel('Sex')
plt.ylabel('Distance')
plt.xticks([1, 2], ['M', 'F'])
plt.show()
print("The data indicates that women generally tend to commute smaller distances, but there are a few women that "
      "commute much farther than men.")
print("")

# superimpose a linear regression on 2a plot 1
time_data = commute_data['Time'].values
distance_data = commute_data['Distance'].values
m, b = np.polyfit(distance_data, time_data, 1)
y = []
for i in range(0, len(distance_data)):
    y.append(m*distance_data[i]+b)
plt.plot(distance_data, time_data, '.')
plt.plot(distance_data, y, '-', color='blue')
plt.show()

# show the distribution of residuals of the data from question 3
x = distance_data.reshape(-1, 1)
y = time_data
ridge = Ridge()
v = ResidualsPlot(ridge)
v.fit(x, y)
v.show()
