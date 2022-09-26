#!/usr/bin/env python
# coding: utf-8

# In[183]:



# Import important python packages needed for this project

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None



# Now we need to read in the data
df = pd.read_csv(r'C:\Users\ernes\OneDrive\Documents\movies.csv')


# In[135]:


# To examine the data

df.head()


# In[136]:


# To check for missing values
# Loop through the data and verify if there is missing data

for col in df.columns:
    number_missing = np.sum(df[col].isnull())
    print('{} - {}'.format(col, number_missing))


# In[137]:


df= df.dropna()


# In[38]:


for col in df.columns:
    number_missing = np.sum(df[col].isnull())
    print('{} - {}'.format(col, number_missing))


# In[138]:


# Data Types for our columns

print(df.dtypes)


# In[139]:


# Chnage data type of the columns

df['budget'] = df['budget'].astype('int64')

df['gross'] = df['gross'].astype('int64')

df['votes'] = df['votes'].astype('int64')

df.head()


# In[140]:


# Creating correct year column

df['yearcorrect'] = df['released'].astype(str).str[:4]

df.head()


# In[146]:


df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[145]:


pd.set_option('display.max_rows', None)


# In[151]:


#To Check remove duplicate

df.drop_duplicates()


# In[ ]:


# Budget high correlation
# Company high correlation


# In[160]:


# Scatter plot of budget vs gross

plt.scatter(x = df['budget'], y = df['gross'])

plt.title('Budget vs Gross Earnings')

plt.xlabel('Budget for a film')

plt.ylabel('Gross Earning')

plt.show()


# In[161]:


df.head()


# In[164]:


# Regression Plot: how much is budget correlated to gross revenue?

sns.regplot(x = 'budget', y = 'gross', data = df, scatter_kws={'color':'red'}, line_kws={'color':'blue'})


# In[165]:


#finding correlation. Method; pearson, kendall, spearman. Peason is the default

df.corr(method='pearson') 

# Confirm high correlation b/w budget and gross (0.74)


# In[185]:


# correlation b/w budget and gross with Heatmap 

correlation_matrix = df.corr(method = 'pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title('Correlation Matrix for Numeric Movie Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[167]:


# Look at the company

df.head()


# In[173]:


df_numerized = df

for col_name in df_numerized.columns: 
    if(df_numerized[col_name].dtype =='object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

#df_numerized


# In[174]:


# correlation b/w budget and gross with Heatmap using new df

correlation_matrix = df_numerized.corr(method = 'pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title('Correlation Matrix for Numeric Movie Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[ ]:


#df_numerized.corr()


# In[176]:


# see correlation  for each variable

correlation_mat = df_numerized.corr()

corr_pairs = correlation_mat.unstack()

corr_pairs.head(10)


# In[180]:


correlation_pairs = correlation_mat.unstack()

sorted_pairs = correlation_pairs.sort_values()

sorted_pairs.head(10)


# In[181]:


# Show only strong positive correlations

high_corr = sorted_pairs[(sorted_pairs) > 0.5]

high_corr.head(10)


# In[ ]:




