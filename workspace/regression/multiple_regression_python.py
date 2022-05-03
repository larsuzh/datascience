import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as sm
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('co2_pm25_deathrates_cleaned.csv')
data.drop(['Unnamed: 0'], axis=1)

africa = data.loc[data['country'] == 'Africa']
asia = data.loc[data['country'] == 'Asia']
europe = data.loc[data['country'] == 'Europe']
north_america = data.loc[data['country'] == 'North America']

x_param = 'co2'
y_param = 'PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)'
z_param = 'rate Deaths - Cause: All causes - Risk: Outdoor air pollution - OWID - Sex: Both - Age: All Ages (Number)'

# Regression for africa

df = pd.DataFrame({'x': africa[x_param],
                   'y': africa[y_param],
                   'z': africa[z_param]
                  })

# linear regression fit
reg = sm.ols(formula='z ~ x + y', data=df).fit()
print(reg.summary())

a0, a1, a2 = reg.params
print('The fitting formula is: z = {0:} + {1:} x + {2:} y'.format(round(a0, 3),round(a1, 3), round(a2, 3)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(africa[x_param], africa[y_param], africa[z_param])

X = np.linspace(600, 1200, 1000)
Y = np.linspace(40, 60, 100)
XX, YY = np.meshgrid(X, Y)
ZZ = a0 + a1 * XX + a2 * YY
ax.plot_surface(XX, YY, ZZ, alpha=0.3, color='orange')

# Regression for asia

df = pd.DataFrame({'x': asia[x_param],
                   'y': asia[y_param],
                   'z': asia[z_param]
                  })

# linear regression fit
reg = sm.ols(formula='z ~ x + y', data=df).fit()
print(reg.summary())

a0, a1, a2 = reg.params
print('The fitting formula is: z = {0:} + {1:} x + {2:} y'.format(round(a0, 3),round(a1, 3), round(a2, 3)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(asia[x_param], asia[y_param], asia[z_param])

X = np.linspace(6000, 20000, 10000)
Y = np.linspace(40, 60, 1000)
XX, YY = np.meshgrid(X, Y)
ZZ = a0 + a1 * XX + a2 * YY
ax.plot_surface(XX, YY, ZZ, alpha=0.3, color='orange')

# Regression for europe

df = pd.DataFrame({'x': europe[x_param],
                   'y': europe[y_param],
                   'z': europe[z_param]
                  })

# linear regression fit
reg = sm.ols(formula='z ~ x + y', data=df).fit()
print(reg.summary())

a0, a1, a2 = reg.params
print('The fitting formula is: z = {0:} + {1:} x + {2:} y'.format(round(a0, 3),round(a1, 3), round(a2, 3)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(europe[x_param], europe[y_param], europe[z_param])

X = np.linspace(5000, 8100, 1000)
Y = np.linspace(10, 20, 1000)
XX, YY = np.meshgrid(X, Y)
ZZ = a0 + a1 * XX + a2 * YY
ax.plot_surface(XX, YY, ZZ, alpha=0.3, color='orange')

# Regression for north america


df = pd.DataFrame({'x': north_america[x_param],
                   'y': north_america[y_param],
                   'z': north_america[z_param]
                  })

# linear regression fit
reg = sm.ols(formula='z ~ x + y', data=df).fit()
print(reg.summary())

#%%
a0, a1, a2 = reg.params
print('The fitting formula is: z = {0:} + {1:} x + {2:} y'.format(round(a0, 3),round(a1, 3), round(a2, 3)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(north_america[x_param], north_america[y_param], north_america[z_param])

X = np.linspace(5500, 7500, 1000)
Y = np.linspace(5, 13, 1000)
XX, YY = np.meshgrid(X, Y)
ZZ = a0 + a1 * XX + a2 * YY
ax.plot_surface(XX, YY, ZZ, alpha=0.3, color='orange')

plt.show()