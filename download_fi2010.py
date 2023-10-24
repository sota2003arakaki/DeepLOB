from fi2010 import fetch_fi2010
import pandas as pd

dataframe = fetch_fi2010()
print(dataframe.head())

dataframe = dataframe.drop(dataframe.columns[[42,43,44,45,46,]],axis=1)
dataframe['PRICE_MID'] = (dataframe['PRICE_ASK_0']+dataframe['PRICE_BID_0'])/2

dataframe.to_csv('data/FI2010.csv')
#for column_name, item in dataframe.iterrows():
#    print(item)