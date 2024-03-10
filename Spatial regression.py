#!/usr/bin/env python
# coding: utf-8

# <h1><center> Climated emergencies in Colombia: Spatial Analysis </center></h1>
# <h1><left> General Objective </left></h1>

# Estimation of a model taking into account risk and changes in temperature and precipitation with an application of spatial regression. The estimate was based on geostatistical studies for rainfall emergencies in Nigeria and on the Institute of Development Studies' compilation work for the UK government on predictive models in humanitarian actions.
# 
# Importing packaged and data collected for the 32 deparments of Colombia. The data includes people benefited, resources invested, risk climate index, average temperature and precipitation.

# In[191]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
import pandas as pd

data2= pd.read_excel('Emer.xlsx', sheet_name="Python")
import numpy as np

y_transformed = np.sqrt(data2['money_real'])
# Update the data frame with the transformed variable
data2['money_spent_transformed'] = y_transformed
data2['cost']= data2['money_real']/data2['people_benefited']
data2['cost']= data2['cost'].fillna(0)
data2['cost_transformed']= np.sqrt(data2['cost'])
data2


# Extract the information about polygons of each department in order to analyze relationships for each deparment with its neighbors.

# In[192]:


import geopandas as gpd
df= data2
df['Departamento'] = df['Departamento'].replace({'ARAUCA': 'Arauca', 'BOLÍVAR': 'Bolívar', 'LA GUAJIRA':'La Guajira',
                                                 'META':'Meta','NORTE DE SANTANDER':'Norte de Santander',
                                                 'SANTANDER':'Santander'})
shapefile_path = 'inputLayers.shp'
gdf = gpd.read_file(shapefile_path)
# Get unique values in Departamento column
deptos = df['Departamento'].unique()

# Filter gdf based on unique values
gdf_filtered = gdf[gdf['NOMBREDEPT'].isin(deptos)]
# Perform left join between df and gdf
df = gdf[['FID_', 'geometry', 'NOMBREDEPT']].merge(df, left_on='NOMBREDEPT', right_on='Departamento', how='right')
df3 = df.drop('FID_', axis=1)
df3['FID_'] = df3.index
df3 = df3.drop(['money_spent','Departamento','people_benefited','money_real','Temperature','Average temperature',
               'Diff_Temp','Precipitacion','Diff_Prec','cost','cost_transformed'], axis=1)
df3


# Use K-nearest neighbors algorithm to create a matrix with binary values if a deparment shares a frontier with another deparment.

# In[193]:


import libpysal
w = libpysal.weights.KNN.from_dataframe(df3)
import numpy as np
from scipy.sparse import dok_matrix

# Convert the weights object to a matrix
w_matrix = dok_matrix((len(w.id_order), len(w.id_order)), dtype=float)
for i in range(len(w.id_order)):
    neighbors, weights = w.neighbors[i], w.weights[i]
    for neighbor, weight in zip(neighbors, weights):
        w_matrix[i, neighbor] = weight

# Convert the matrix to a DataFrame
w_df = pd.DataFrame(np.array(w_matrix.todense()), index=w.id_order, columns=w.id_order)
w_df


# In[194]:


import numpy as np
import pandas as pd
from statsmodels.api import OLS
from spreg import GM_Lag, GM_Error_Het, ML_Lag, ML_Error

data=df3
# convert data to numpy arrays
y = data["money_spent_transformed"].values
x = data[['risk_index', 'Porc_Temp', 'Porc_Prec']].values
w = libpysal.weights.KNN.from_dataframe(data,  k=3)


# <h1><left> Spatial regression </left></h1>

# In[150]:


import numpy as np
import pandas as pd
import libpysal
from esda.moran import Moran
from esda.smaup import Smaup
from statsmodels.api import OLS
from scipy import stats
from spreg import GM_Lag, GM_Error_Het, ML_Lag, ML_Error

# Compute the residuals from the spatial regression model
residuals = model_s2lq.u

# Compute the spatial weights matrix
w = libpysal.weights.KNN.from_dataframe(data,  k=3)
w.transform = 'r'

# Compute Moran's I statistic
moran = Moran(residuals, w)
print("Moran's I: {:.4f}".format(moran.I))
print("p-value: {:.4f}".format(moran.p_sim))

from statsmodels.stats.diagnostic import het_breuschpagan

# Compute the residuals and predicted values from the spatial regression model
residuals = model_s2lq.u
predicted_values = model_s2lq.predy

# Compute the squared residuals and the product of the squared residuals and the predicted values
squared_residuals = residuals**2
product = squared_residuals * predicted_values

# Regress the squared residuals and the product on the original predictors
_, p1, _, _ = het_breuschpagan(squared_residuals, x)
_, p2, _, _ = het_breuschpagan(product, x)

# Compute the Breusch-Pagan statistic and p-value
bp_statistic = p1 + p2
bp_pvalue = stats.chi2.sf(bp_statistic, x.shape[1])
print("Breusch-Pagan statistic: {:.4f}".format(bp_statistic))
print("p-value: {:.4f}".format(bp_pvalue))


# <h1><left> Prediction </left></h1>

# In[195]:


# Prepare new data
new_data = pd.DataFrame({
    'Constant': [1,1,1,1,1,1],
    'risk_index': [0.1618, 0.1867, 0.1566, 0.737, 0.544, 0.451 ],
    'Porc_Temp': [0.0016, 0.14, 0.1531, 0.2, 0.05, 0.05],
    'Porc_Prec': [-0.26, -0.28, -0.15, -0.15, -0.001, -0.001],
'Departamento': ['Meta', 'Bolívar', 'Santander', 'SAN ANDRÉS PROVIDENCIA', 'Vaupés', 'Amazonas']})


# Get unique values in Departamento column
deptos = new_data['Departamento'].unique()

# Filter gdf based on unique values
gdf_filtered = gdf[gdf['NOMBREDEPT'].isin(deptos)]
# Perform left join between df and gdf
new_data = gdf[['FID_', 'geometry', 'NOMBREDEPT']].merge(new_data, left_on='NOMBREDEPT', right_on='Departamento', how='right')
new_data = new_data.drop('FID_', axis=1)
new_data['FID_'] = new_data.index
new_data

# Load weights matrix
w_new = libpysal.weights.KNN.from_dataframe(new_data, ids='FID_', k=3)
w_new.transform = 'r'

x_new = new_data[['Constant','risk_index', 'Porc_Temp', 'Porc_Prec']].values
from spreg.utils import inverse_prod
from spreg.sputils import spdot
import locale

# set the locale to your preferred format
locale.setlocale(locale.LC_ALL, 'en_US')

# obtain predicted values for the original data
y_pred = model_s2lq.predy
# Load weights matrix
w_new = libpysal.weights.KNN.from_dataframe(new_data, ids='FID_', k=3)
w_new.transform = 'r'


b = model_s2lq.betas
xb= spdot(x_new,b[:-1])
y_pred_new= inverse_prod(w_new.sparse, xb, model_s2lq.rho, inv_method= 'power_exp')
# apply inverse transformation
y_pred_new = np.square(y_pred_new)

# original predicted values
y_pred_dollars = np.array(y_pred_new)

# convert to dollars with two decimal places and thousands separators
# convert to dollars with two decimal places and thousands separators
y_pred_dollars_str = []
for value in y_pred_dollars:
    formatted_value = '{:,.2f}'.format(float(value))
    y_pred_dollars_str.append(f'${formatted_value}')

print(y_pred_dollars_str)


# In[196]:


from pysal.model.spreg import ML_Lag
from pysal.model.spreg import diagnostics as diag
from pysal.model import spreg
from esda.geary import Geary

# calculate Moran's I
moran = Moran(y, w)

# test for spatial autocorrelation
print("Moran's I: ", moran.I)
print("p-value: ", moran.p_sim)
geary = Geary(y, w)
print("Geary's I: ", geary.C)
print("p-value: ", geary.p_sim)


# <h1><left> Graphics and statistics </left></h1>

# In[197]:


import matplotlib.pyplot as plt
from libpysal.weights.contiguity import Queen
from libpysal import examples
import geopandas as gpd
from esda.moran import Moran
from splot.esda import plot_moran
plot_moran(moran)


# In[198]:


from esda.moran import Moran_Local
lisa = Moran_Local(y, w)


# In[199]:


import seaborn  # Graphics
# Draw KDE line
ax = seaborn.kdeplot(lisa.Is)
# Add one small bar (rug) for each observation
# along horizontal axis
seaborn.rugplot(lisa.Is, ax=ax);


# In[200]:


import matplotlib.pyplot as plt
from splot.esda import lisa_cluster
# Set up figure and axes
f, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
# Make the axes accessible with single indexing
axs = axs.flatten()

# Subplot 1 #
# Choropleth of local statistics
# Grab first axis in the figure
ax = axs[0]
# Assign new column with local statistics on-the-fly
df3.assign(
    Is=lisa.Is
    # Plot choropleth of local statistics
).plot(
    column="Is",
    cmap="plasma",
    scheme="quantiles",
    k=5,
    edgecolor="white",
    linewidth=0.1,
    alpha=0.75,
    legend=True,
    ax=ax,
)

# Subplot 2 #
# Quadrant categories
# Grab second axis of local statistics
ax = axs[1]
# Plot Quadrant colors (note to ensure all polygons are assigned a
# quadrant, we "trick" the function by setting significance level to
# 1 so all observations are treated as "significant" and thus assigned
# a quadrant color
lisa_cluster(lisa, df3, p=1, ax=ax)

# Subplot 3 #
# Significance map
# Grab third axis of local statistics
ax = axs[2]
#
# Find out significant observations
labels = pd.Series(
    1 * (lisa.p_sim < 0.05),  # Assign 1 if significant, 0 otherwise
    index=df3.index  # Use the index in the original data
    # Recode 1 to "Significant and 0 to "Non-significant"
).map({1: "Significant", 0: "Non-Significant"})
# Assign labels to `db` on the fly
df3.assign(
    cl=labels
    # Plot choropleth of (non-)significant areas
).plot(
    column="cl",
    categorical=True,
    k=2,
    cmap="Paired",
    linewidth=0.1,
    edgecolor="white",
    legend=True,
    ax=ax,
)


# Subplot 4 #
# Cluster map
# Grab second axis of local statistics
ax = axs[3]
# Plot Quadrant colors In this case, we use a 5% significance
# level to select polygons as part of statistically significant
# clusters
lisa_cluster(lisa, df3, p=0.05, ax=ax)

# Figure styling #
# Set title to each subplot
for i, ax in enumerate(axs.flatten()):
    ax.set_axis_off()
    ax.set_title(
        [
            "Local Statistics",
            "Scatterplot Quadrant",
            "Statistical Significance",
            "Moran Cluster Map",
        ][i],
        y=0,
    )
# Tight layout to minimize in-between white space
f.tight_layout()

# Display the figure
plt.show()


# In[265]:


import numpy as np
import pandas as pd
import libpysal
from esda.moran import Moran
from esda.smaup import Smaup
from statsmodels.api import OLS
from scipy import stats
from libpysal.weights import W
data= df3
# calculate spatial lags of variables
# convert data to numpy arrays
w = libpysal.weights.KNN.from_dataframe(data,  k=1)
y_lag = pd.Series(libpysal.weights.lag_spatial(w, df3['money_spent_transformed']), name='y_lag')
x1_lag = pd.Series(libpysal.weights.lag_spatial(w, df3['risk_index']), name='x1_lag')
x2_lag = pd.Series(libpysal.weights.lag_spatial(w, df3['Porc_Temp']), name='x2_lag')
x3_lag = pd.Series(libpysal.weights.lag_spatial(w, df3['Porc_Prec']), name='x3_lag')


# In[266]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

# combine original and lag variables
X_new = pd.concat([data[['year']], y_lag, x1_lag, x2_lag, x3_lag], axis=1)
# Split data into training and testing sets
train_data = X_new[X_new['year'] >= 2020]
test_data = X_new[X_new['year'] < 2020]
w = libpysal.weights.KNN.from_dataframe(data,  k=1)
# Define the dependent and independent variables
X= X_new[['x1_lag', 'x2_lag', 'x3_lag']]
y= X_new[['y_lag']]
y_train = train_data['y_lag']
X_train = train_data[['x1_lag', 'x2_lag', 'x3_lag']]
y_test = test_data['y_lag']
X_test = test_data[['x1_lag', 'x2_lag', 'x3_lag']]


# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Regression": SVR(),
    "XGBoost": XGBRegressor(),
    "LightGBM": LGBMRegressor()
}

# Evaluate models with cross-validation
for name, model in models.items():
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"{name}: RMSE = {rmse_scores.mean():.2f} +/- {rmse_scores.std():.2f}")
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    print(f"{name}: R^2 = {r2_scores.mean():.2f} +/- {r2_scores.std():.2f}")


# In[267]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
for name, model in models.items():
    # fit the model on the training set
    model.fit(X_train, y_train)
    # evaluate the model on the test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name} - MAE on test set: {mae:.2f}")
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name} - MSE on test set: {mse:.2f}")
    # calculate R2 score
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - R2 on test set: {r2:.2f}")
    ev = explained_variance_score(y_test, y_pred)
    print(f"{name} - Explained Variance Score: {ev:.2f}")
mean_y_test = np.mean(y_test)
print(f"Mean value of dependent variable on test set: {mean_y_test:.2f}")


# In[273]:


# Initialize models
rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
xgb = XGBRegressor(n_estimators=50, max_depth=5, random_state=42)


# In[274]:


rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)


# In[275]:


y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"rf - MAE on test set: {mae:.2f}")
mse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"rf - MSE on test set: {mse:.2f}")
# calculate R2 score
r2 = r2_score(y_test, y_pred)
print(f"rf - R2 on test set: {r2:.2f}")
ev = explained_variance_score(y_test, y_pred)
print(f"rf - Explained Variance Score: {ev:.2f}")


# In[276]:


y_pred = xgb.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"xgb - MAE on test set: {mae:.2f}")
mse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"xgb - MSE on test set: {mse:.2f}")
# calculate R2 score
r2 = r2_score(y_test, y_pred)
print(f"xgb - R2 on test set: {r2:.2f}")
ev = explained_variance_score(y_test, y_pred)
print(f"xgb - Explained Variance Score: {ev:.2f}")


# In[272]:


from sklearn.model_selection import GridSearchCV

# Define parameter grid for random forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'random_state': [42]
}

# Define parameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'random_state': [42]
}

# Perform grid search for random forest
rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5, scoring='neg_mean_squared_error')
rf_grid.fit(X_train, y_train)

# Perform grid search for XGBoost
xgb_grid = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, cv=5, scoring='neg_mean_squared_error')
xgb_grid.fit(X_train, y_train)

# Print the best hyperparameters for each model
print(f"Random Forest: {rf_grid.best_params_}")
print(f"XGBoost: {xgb_grid.best_params_}")


# In[277]:


# Make predictions on new data
# Prepare new data
new_data = pd.DataFrame({
    'Constant': [1,1,1,1,1,1,1,1,1,1],
    'risk_index': [0.1618, 0.1867, 0.1566, 0.172300,0.150465, 0.737, 0.544, 0.451, 0.39,0.265],
    'Porc_Temp': [0.016, 0.014, 0.015,0.1,0.016, 0.1, 0.05, 0.05, 0.05,0.02],
    'Porc_Prec': [-0.16, -0.18, -0.15,-0.15, -0.16,-0.2, -0.01, -0.01, -0.01, -0.15],
'Departamento': ['Meta', 'Bolívar', 'Santander','La Guajira','Casanare','SAN ANDRÉS PROVIDENCIA', 'Vaupés', 'Amazonas', 'Guainía','Chocó']})


# Get unique values in Departamento column
deptos = new_data['Departamento'].unique()

# Filter gdf based on unique values
gdf_filtered = gdf[gdf['NOMBREDEPT'].isin(deptos)]
# Perform left join between df and gdf
new_data = gdf[['FID_', 'geometry', 'NOMBREDEPT']].merge(new_data, left_on='NOMBREDEPT', right_on='Departamento', how='right')
new_data = new_data.drop('FID_', axis=1)
new_data['FID_'] = new_data.index
new_data

# calculate spatial lags of variables
# convert data to numpy arrays
w = libpysal.weights.KNN.from_dataframe(data,  k=3)
x1_lag= pd.Series(libpysal.weights.lag_spatial(w_new, new_data['risk_index']), name='x1_lag')
x2_lag = pd.Series(libpysal.weights.lag_spatial(w_new, new_data['Porc_Temp']), name='x2_lag')
x3_lag = pd.Series(libpysal.weights.lag_spatial(w_new, new_data['Porc_Prec']), name='x3_lag')
X_new = pd.concat([x1_lag, x2_lag, x3_lag], axis=1)
y_pred_rf = rf.predict(X_new)
y_pred_xgb= xgb.predict(X_new)

# Print the predicted values
print(np.square(y_pred_rf))
print(np.square(y_pred_xgb))

mean_y_pred_rf = np.mean(y_pred_rf)
mean_y_pred_xgb=np.mean(y_pred_xgb)
import locale

# set the locale to use US dollars format
locale.setlocale(locale.LC_ALL, 'en_US')
# format the mean prediction value as US dollars
mean_y_pred_formattedrf = locale.currency(np.square(mean_y_pred_rf), grouping=True)
mean_y_pred_formattedxgb =locale.currency(np.square(mean_y_pred_xgb), grouping=True)
print("Mean prediction value rf:", mean_y_pred_formattedrf)
print("Mean prediction value xgb:", mean_y_pred_formattedxgb)                                           
# Print the predicted values in money format
for pred in y_pred_rf:
    print('${:,.2f}'.format(np.square(pred)))
for pred in y_pred_xgb:
    print('${:,.2f}'.format(np.square(pred)))


# In[226]:


import numpy as np
import matplotlib.pyplot as plt


# get predicted values and standard deviations
preds = xgb.predict(X_test)
stds = np.std([xgb.predict(X_test) for i in range(1000)], axis=0)

# plot the results with 95% confidence intervals
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(y_test, preds)
ax.errorbar(y_test, preds, yerr=1.96*stds, fmt='none', alpha=0.5)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--')
ax.set_xlabel('True Values')
ax.set_ylabel('Predictions')
plt.show()


# In[145]:


y_pred = xgb.predict(X)
y_pred=pd.Series(xgb.predict(X), name='y_pred')
residuals = y - y_pred
lisa = Moran_Local(residuals, w)


# In[146]:


data.assign(
    Is=lisa.Is)


# In[ ]:




