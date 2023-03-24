import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

main_df = pd.read_csv(r"C:\Users\Niraj N\Downloads\archive\train.csv")
main_df = main_df.head(600)
main_df.head(20)

cmpi=['UNDER_CONSTRUCTION','RERA' , 'BHK_NO.' , 'SQUARE_FT' , 'READY_TO_MOVE' , 'RESALE', 'TARGET(PRICE_IN_LACS)']
corr_matrix = main_df[cmpi].corr()
tempdf = main_df[cmpi]

tempdf['sqft_pr'] = tempdf['SQUARE_FT'] / tempdf['BHK_NO.']
corr_matrix = tempdf.corr()

skew = ['READY_TO_MOVE','RESALE','UNDER_CONSTRUCTION','RERA']
shuf =  StratifiedShuffleSplit(n_splits = 1,test_size=0.2,random_state=42)
for trindx , tstindx in shuf.split(tempdf,tempdf[skew]):
    df_train = tempdf.loc[trindx] 
    df_test = tempdf.loc[tstindx]

propertiesnum = ['UNDER_CONSTRUCTION','RERA' , 'BHK_NO.' , 'SQUARE_FT' , 'READY_TO_MOVE' , 'RESALE', 'sqft_pr']
x_train = df_train[propertiesnum]
y_train=df_train['TARGET(PRICE_IN_LACS)']
x_test = df_test[propertiesnum]
y_test = df_test['TARGET(PRICE_IN_LACS)']

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = 1)
model.fit(x_train,y_train)

pred = model.predict(x_train)
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(pred, y_train)

pred = model.predict(x_test)
val_mae = mean_absolute_error(pred, y_test)

from sklearn.ensemble import RandomForestRegressor
maxl = [2000,1900 ,1875,1850]
bml = []
for i in maxl:
    deci = model = RandomForestRegressor(random_state = 1, max_leaf_nodes = i)
    deci.fit(x_train,y_train)
    pred = deci.predict(x_test)
    print(mean_absolute_error(pred, y_test))
    bml.append(mean_absolute_error(pred, y_test))
m = min(bml)
print("---",m)
q = bml.index(m)
print(q)
max_l_n = maxl[q]
max_l_n

from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1,max_leaf_nodes = 1875)
forest_model.fit(x_train,y_train)
melb_preds = forest_model.predict(x_test)
print(mean_absolute_error(y_test, melb_preds))