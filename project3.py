import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model, metrics
from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict

% matplotlib inline

df = pd.read_csv('Iowa_Liquor_Sales_reduced.csv', infer_datetime_format=True)
# Renaming the columns to make them easier to call
columns = ['date','store','city','zip','county_num','county','category',
        'category_name','vendor_num','item_num','item_descr','vol',
        'bottle_cost','bottle_retail','bottles_sold','sale_dollars','vol_sold',
        'vol_sold_gal']
df.columns = columns

# Removing the dollar sign from monetary columns
remove_sign = ['bottle_cost','bottle_retail','sale_dollars']
for col in remove_sign:
    df[col] = df[col].str.replace('$','')

# Converting strings to numeric
to_num = ['bottle_cost','bottle_retail','sale_dollars']
for col in to_num:
    df[col] = pd.to_numeric(df[col])

# Converting date from string to datetime
df.date = pd.to_datetime(df.date, infer_datetime_format=True)

# Deleting the vol_sold_gal column due to redunancy with liters
df.drop('vol_sold_gal',axis=1,inplace=True)
# Add a margin column
margin = pd.Series((df.bottle_retail-df.bottle_cost)*df.bottles_sold)
df['margin'] = margin
# Fixing two rows where the vol_sold was 0, which caused PPL to be infinity
df.set_value(815951,'vol',50)
df.set_value(815951,'vol_sold',600)
df.set_value(2007100,'vol',50)
df.set_value(2007100,'vol_sold',600)
# A store in Dunlap, IA had a zip code of 712-2. A google search showed the
# correct zip code for Dunlap is 51529
wrong_zip = df[df.zip == '712-2']
for index in wrong_zip.index:
    df.set_value(index,'zip',51529)
# Add a price per liter column
ppl = pd.Series((df.bottle_retail/df.vol)*1000)
df['price_per_liter'] = ppl

# Sales per store in 2015
df.sort_values(by=["store", "date"], inplace=True)
start_date = pd.Timestamp("20150101")
end_date = pd.Timestamp("20151231")
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
sales_2015 = df[mask]
sales_2015 = sales_2015.groupby(by=['store'],as_index=False)
sales_2015 = sales_2015.agg({"sale_dollars": [np.sum, np.mean],
                   "vol_sold": [np.sum, np.mean],
                   "margin": np.mean,
                   "price_per_liter": np.mean,
                   "zip": lambda x: x.iloc[0],
                   "city": lambda x: x.iloc[0],
                   "county_num": lambda x: x.iloc[0],
                   "county": lambda x: x.iloc[0]})
sales_2015.columns =[' '.join(col).strip() for col in sales_2015.columns.values]
sales_cols = ['store','county','city','sale_dollars_sum','sale_dollars_mean',
            'zip','ppl_mean','margin_mean','vol_sold_sum','vol_sold_mean',
            'county_num']
sales_2015.columns = sales_cols

# Daily sales per store in 2015
df.sort_values(by=["store", "date"], inplace=True)
start_date = pd.Timestamp("20150101")
end_date = pd.Timestamp("20151231")
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
day_sales_2015 = df[mask]
day_sales_2015.head(100)
day_sales_2015 = day_sales_2015.groupby(by=['store','date'],as_index=False)
day_sales_2015 = day_sales_2015.agg({"sale_dollars": np.mean,
                        "margin": np.mean,
                        "bottles_sold": np.mean,
                        "vol_sold": np.mean})
day_sales_2015.head(200)

# Sales per store 2015 Q1
df.sort_values(by=["store","date"], inplace=True)
start_date = pd.Timestamp("20150101")
end_date = pd.Timestamp("20150331")
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
sales_2015Q1 = df[mask]
sales_2015Q1 = sales_2015Q1.groupby(by=['store'],as_index=False)
sales_2015Q1 = sales_2015Q1.agg({"sale_dollars": [np.sum, np.mean],
                   "vol_sold": [np.sum, np.mean],
                   "margin": np.mean,
                   "price_per_liter": np.mean,
                   "zip": lambda x: x.iloc[0],
                   "city": lambda x: x.iloc[0],
                   "county_num": lambda x: x.iloc[0],
                   "county": lambda x: x.iloc[0]})
sales_2015Q1.columns = [' '.join(col).strip() for col in sales_2015Q1.columns.values]
sales_2015Q1.columns = sales_cols

# Sales per store 2016 Q1
df.sort_values(by=["store","date"], inplace=True)
start_date = pd.Timestamp("20160101")
end_date = pd.Timestamp("20160331")
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
sales_2016Q1 = df[mask]
sales_2016Q1 = sales_2016Q1.groupby(by=['store'],as_index=False)
sales_2016Q1 = sales_2016Q1.agg({"sale_dollars": [np.sum, np.mean],
                   "vol_sold": [np.sum, np.mean],
                   "margin": np.mean,
                   "price_per_liter": np.mean,
                   "zip": lambda x: x.iloc[0],
                   "city": lambda x: x.iloc[0],
                   "county_num": lambda x: x.iloc[0],
                   "county": lambda x: x.iloc[0]})
sales_2016Q1.columns = [' '.join(col).strip() for col in sales_2016Q1.columns.values]
sales_2016Q1.columns = sales_cols

# Creating a list of store that were open for all of 2015
# We do this to remove outliers from our data set
stores = df.store.unique()
store_totals = pd.DataFrame(columns=['first_sale', 'last_sale', '2015',
                'Q1_2015', 'Q1_2016', 'total_volumes', 'mean_volume',
                'mean_sale', 'mean_margin', 'mean_PperL', 'city','zip',
                'county'],
                index = stores)
for store in stores:
    store_df = df[df.store == store]
    store_max = max(store_df.date)
    store_min = min(store_df.date)
    store_totals['first_sale'][store] = store_min
    store_totals['last_sale'][store] = store_max
feb = pd.Timestamp("20150201")
nov = pd.Timestamp("20151130")
store_totals = store_totals[store_totals.first_sale <= feb]
store_totals = store_totals[store_totals.last_sale >= nov]

# Delete stores from store_totals that aren't open in 2016
for index, store_num in store_totals.iterrows():
    if index not in sales_2016Q1.store.values:
        store_totals.drop(index,axis=0,inplace=True)
# Delete stores from sales_2016Q1 that weren't open in 2015
for index, store_num in sales_2016Q1.iterrows():
    if store_num.store not in store_totals.index.values:
        sales_2016Q1.drop(index,axis=0,inplace=True)

# Building the store_totals DataFrame from the 3 DataFrames created above
for index, store_num in store_totals.iterrows():
    store_totals['2015'][index]=sales_2015[sales_2015.store == index].sale_dollars_sum.iloc[0]
    store_totals['Q1_2015'][index]=sales_2015Q1[sales_2015Q1.store == index].sale_dollars_sum.iloc[0]
    store_totals['Q1_2016'][index]=sales_2016Q1[sales_2016Q1.store == index].sale_dollars_sum.iloc[0]
    store_totals['city'][index]=sales_2015[sales_2015.store == index].city.iloc[0]
    store_totals['zip'][index]=sales_2015[sales_2015.store == index].zip.iloc[0]
    store_totals['county'][index]=sales_2015[sales_2015.store == index].county.iloc[0]
    # For 2015 data:
    store_totals['total_volumes'][index]=sales_2015[sales_2015.store == index].vol_sold_sum.iloc[0]
    store_totals['mean_sale'][index]=sales_2015[sales_2015.store == index].sale_dollars_mean.iloc[0]
    store_totals['mean_margin'][index]=sales_2015[sales_2015.store == index].margin_mean.iloc[0]
    store_totals['mean_PperL'][index]=sales_2015[sales_2015.store == index].ppl_mean.iloc[0]
    store_totals['mean_volume'][index]=sales_2015[sales_2015.store == index].vol_sold_mean.iloc[0]
    # For 2016 data:
    # store_totals['mean_margin_16']=sales_2016Q1[sales_2016Q1.store == index].margin_mean.iloc[0]
    # store_totals['mean_vol_16']=sales_2016Q1[sales_2016Q1.store == index].vol_sold_mean.iloc[0]
    # store_totals['mean_ppl_16']=sales_2016Q1[sales_2016Q1.store == index].ppl_mean.iloc[0]
store_totals.head()

model_df = pd.DataFrame({'2015_Q1':store_totals['Q1_2015']})
y = store_totals['2015']
# X_train, X_test, y_train, y_test = train_test_split(model_df,y,test_size=.2)
# print X_train.shape, y_train.shape
# print X_test.shape, y_test.shape
lm = linear_model.LinearRegression()
# model = lm.fit(X_train,y_train)
model = lm.fit(model_df,y)
# predictions = lm.predict(X_test)

scores = cross_val_score(model,model_df,y,cv=12)
print 'Cross-validated scores:',scores
predictions = cross_val_predict(model,model_df,y,cv=12)
plt.scatter(y,predictions,alpha=.3)
# plt.show()
accuracy = metrics.r2_score(y,predictions)
print 'Cross-predicted accuracy:',accuracy

x_2016 = pd.DataFrame({'2016_Q1_sales':store_totals['Q1_2016']})
predictions_2016 = lm.predict(x_2016)
predictions_2016
# Change the store number from the DF index to a column
# Need to do this to include store number in our pivot tables
store_totals = store_totals.reset_index()
store_totals.columns.values[0] = 'store'

# Creating a pivot table grouped by zip code
zip_pivot = pd.pivot_table(store_totals,
                        index=['zip'],
                        values=['2015','Q1_2015','Q1_2016','store'],
                        aggfunc={'2015':np.sum,'Q1_2015':np.sum,
                        'Q1_2016':np.sum,'store':lambda x: len(x.unique())})
# Create a blank DataFrame for the zip code and average sales per store
zip_normalized = pd.DataFrame(columns=['zip','normalized'])
# For each zip code we predict the total 2016 sales based on Q1 2016 sales
for zip_code in zip_pivot.index.values:
    zip_predict = lm.predict(zip_pivot['Q1_2016'][zip_code])
    zip_count = zip_pivot['store'][zip_code]
    # Divide zip code projected sales by number of stores in the zip to
    # normalize between zip codes
    zip_norm = zip_predict/zip_count
    print 'Predicted:%f Count:%f Normalized:%f' % (zip_predict,zip_count,zip_norm)
    zip_dict = {'zip':zip_code,'normalized':zip_norm}
    zip_normalized.loc[len(zip_normalized)] = zip_dict
# Sort the new DataFrame by projected sales per zip code
zip_norm_sorted = zip_normalized.sort_values(by='normalized',ascending=False)
zip_norm_sorted.head()
# We also calculate the top selling zip code (normalized for number of stores)
# for 2015 to see how projected top zip codes compare to 2015 top zip codes
zip_pivot_2015 = zip_pivot.drop(['Q1_2015','Q1_2016'],axis=1)
zip_pivot_2015['normalized'] = zip_pivot_2015['2015']/zip_pivot_2015['store']
zip_pivot_2015.sort_values(by='normalized',ascending=False).head(10)


# Creating a pivot table grouped by county
county_pivot = pd.pivot_table(store_totals,
                    index=['county'],
                    values=['2015','Q1_2015','Q1_2016','store'],
                    aggfunc={'2015':np.sum,'Q1_2015':np.sum,'Q1_2016':np.sum,
                            'store':lambda x: len(x.unique())})
# Create a blank DataFrame for the county and average sales per store
county_normalized = pd.DataFrame(columns=['county','normalized'])
# For each county we predict the total 2016 sales based on Q1 2016 sales
for county in county_pivot.index.values:
    county_predict = lm.predict(county_pivot['Q1_2016'][county])
    county_count = county_pivot['store'][county]
    # Divide county projected sales by number of stores in the county to
    # normalize between county
    county_norm = county_predict/county_count
    print 'Predicted:%f Count:%f Normalized:%f' % (county_predict,county_count,county_norm)
    county_dict = {'county':county,'normalized':county_norm}
    county_normalized.loc[len(county_normalized)] = county_dict
# Sort the new DataFrame by projected sales per county
county_norm_sorted = county_normalized.sort_values(by='normalized',
                                                   ascending=False)
county_norm_sorted.head()
# We also calculate the top selling county (normalized for number of stores)
# for 2015 to see how projected top counties compare to 2015 top counties
county_pivot_2015 = county_pivot.drop(['Q1_2015','Q1_2016'],axis=1)
county_pivot_2015['normalized'] = county_pivot_2015['2015']/county_pivot_2015['store']
county_pivot_2015.sort_values(by='normalized',ascending=False).head(10)





# store_totals2 = store_totals
# store_totals2.head()
# store_totals2.drop(['first_sale','last_sale','store','mean_sale','mean_margin','mean_bottle','mean_day_sales','mean_day_count','mean_day_L','mean_day_margin','city','zip','county'],axis=1,inplace=True)
# df_array = store_totals2.as_matrix()
# cor = np.corrcoef(df_array)
