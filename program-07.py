"""

Created on April 19, 2020
by Kush Paliwal

Code to analyze the data about earthquakes for past 30 days 
The data was downloaded at 12:36 pm on 19/04/2020
"""

# Import modules
import pandas as pd
from scipy import stats
import scipy as si
import numpy as np
import matplotlib.pyplot as plt

# Read data from csv file
df = pd.read_table('all_month.csv', header=0, sep=',')

# Plot histogram of the magnitude of earthquakes
boundary = range(0, 11, 1)
plt.hist(df['mag'].dropna(), bins=boundary)
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.title('Histogram: Magnitude of Earthquakes')
plt.show()

# Plot KDE 
kde = stats.gaussian_kde(df['mag'].dropna())
kde.covariance_factor = lambda:0.1
kde._compute_covariance()
a = np.sort(df['mag'].dropna())
plt.plot(a, kde(a))
plt.xlabel('Magnitude')
plt.ylabel('Density')
plt.title('KDE Plot')
plt.show()


# Plot latitude vs longitude
plt.scatter(df['longitude'], df['latitude'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Location of earthquakes')
plt.show()

# Plot normalized cumulative distribution for depth
d = np.sort(df['depth'])
cdf = np.linspace(0, 1, len(d))
plt.plot(d, cdf)
plt.xlabel('Depth (km)')
plt.ylabel('Cumulative Distribution')
plt.title('CDF Plot')
plt.show()


# Plot scatter of magnitude with depth
plt.scatter(df['mag'], df['depth'])
plt.xlabel('Magnitude')
plt.ylabel('Depth')
plt.title('Magnitude VS Depth')
plt.show()

# Q-Q plot of magnitude
si.stats.probplot(df['mag'].dropna(), dist = 'norm', plot=plt)
plt.show()