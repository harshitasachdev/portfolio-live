from sklearn.linear_model import ElasticNet,ElasticNetCV,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats.mstats import winsorize
from xgboost import XGBRegressor
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import zscore

from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np


def select_features(X,Y,corr_threshold):
    #n_features = X.shape[1]
    corrmat = pd.concat([X,Y],axis = 1).corr().iloc[-1]
    features = corrmat[abs(corrmat)>corr_threshold].index[:-1]
    if len(features)>1:
        X = X[features]
    return X
    
def constraint_prediction(data,threshold = 0.1):
    data[data>threshold] = threshold
    data[data<-threshold]=-threshold
    return data


def extract_pcs(tempX,testX,n_components):
    pca = PCA(n_components=min(n_components,tempX.shape[1]))
    pca.fit(tempX)  
    tempX = pca.transform(tempX)
    testX = pca.transform(testX)  
    #tempX = tempX.iloc[:,0]
    #testX = testX.iloc[:,0]
    return tempX,testX

def rolling_model(fwd_rets,features,lookback_wdw,model='eNet',
                  perform_PCA = True,
                  feature_selection = False,
                  return_type ='returns',winsor = True):
    merged = features.copy()
    lkback =lookback_wdw
    targets = fwd_rets.copy()
    returns_pred = []
    if model == 'eNet':
        model_apply =ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42)
    elif model=='linear':
        model_apply = LinearRegression()
    elif model =='RF':
        model_apply = RandomForestRegressor(max_depth = 6,n_estimators = 100)
    elif model =='XGB':
        model_apply = XGBRegressor(max_depth = 4,learning_rate = 0.04,n_estimators = 200)


    #for col in targets.columns:
    outputs= [0]
    for i in range(1,targets.shape[0]):
        tempY = targets.iloc[max(0,i-lkback):i]
        tempX = merged.iloc[max(0,i-lkback):i]
        #Select features based on correlation
        if select_features==True:
            tempX = select_features(tempX,tempY,0.01)
        tempXcols = tempX.columns

        scaler = StandardScaler()
        scaler.fit(tempX)
        tempX = scaler.transform(tempX)

        #For test input change the columns to that of tempX
        testX = merged[tempXcols].iloc[[i]]
        testX = scaler.transform(testX)
        
        if perform_PCA==True and i >tempX.shape[1]:
            tempX,testX = extract_pcs(tempX,testX)  
        
        if model == 'linear' or model =='eNet':
            tempX = sm.add_constant(tempX)
            testX = [1]+list(testX[0])
            testX = [testX]
            #print(testX)
            #testX = sm.add_constant(testX)

        
        model_apply.fit(tempX,tempY)
        pred = model_apply.predict(testX)[0]
        
        if return_type=='returns':
            outputs.append(pred)
        elif return_type =='rsq':
            model_rsq = r2_score(tempY,model_apply.predict(tempX))
            outputs.append(model_rsq)
    returns_pred = pd.DataFrame(outputs).fillna(0)
    returns_pred.index = targets.index
    returns_pred.columns = targets.columns
    if winsor==True and return_type=='returns':
        returns_pred = constraint_prediction(returns_pred,0.05)
    return returns_pred


def extract_rolling_PCs(X,lkback = 250*2,n_components=2):

    outputs= [[0]*n_components]
    for i in range(1,X.shape[0]):
        tempX = X.iloc[max(0,i-lkback):i]
        testX = X.iloc[[i]]
        scaler = StandardScaler()
        scaler.fit(tempX)
        tempX = scaler.transform(tempX)

        #For test input change the columns to that of tempX
        #testX = testX[tempXcols].iloc[[i]]
        testX = scaler.transform(testX)
        try:
            tempX,testX = extract_pcs(tempX,testX,n_components)
            testX = testX[0]
        except:
            testX = np.array([0]*n_components)
        outputs.append(testX.tolist())
    pcs = pd.DataFrame(outputs)
    #return outputs
    pcs.index = X.index
    #pcs.columns = X.columns
    return pcs


def rolling_model2(fwd_rets,features,lookback_wdw,model='eNet',
                  return_type ='returns',winsor = True):
    merged = features.copy()
    lkback =lookback_wdw
    targets = fwd_rets.copy()
    returns_pred = []
    if model == 'eNet':
        model_apply =ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42)
    elif model=='linear':
        model_apply = LinearRegression()
    elif model =='RF':
        model_apply = RandomForestRegressor(max_depth = 6,n_estimators = 100)
    elif model =='XGB':
        model_apply = XGBRegressor(max_depth = 4,learning_rate = 0.04,n_estimators = 200)


    #for col in targets.columns:
    outputs= [0]
    c = 0
    for i in range(1,targets.shape[0]):
        tempY = targets.iloc[max(0,i-lkback):i]
        tempX = merged.iloc[max(0,i-lkback):i]

        testX = merged.iloc[[i]]
        #print(testX)
        if model == 'linear' or model =='eNet':
            tempX = sm.add_constant(tempX)
            testX = sm.add_constant(testX)
        if c==1:
            return testX
        c+=1
        
        model_apply.fit(tempX,tempY)
        pred = model_apply.predict(testX)
        
        if return_type=='returns':
            outputs.append(pred[0])
        elif return_type =='rsq':
            model_rsq = r2_score(tempY,model_apply.predict(tempX))
            outputs.append(model_rsq)
    returns_pred = pd.DataFrame(outputs).fillna(0)
    returns_pred.index = targets.index
    returns_pred.columns = targets.columns
    if winsor==True and return_type=='returns':
        returns_pred = constraint_prediction(returns_pred,0.05)
    return returns_pred

def rolling_lr(fwd_rets,features,idio_features,lookback_wdw,model = 'linear'):
    ret_df = []
    for col in fwd_rets.columns:
        ret_preds = [0]
        features_combined = pd.concat([features,idio_features[col]],axis = 1)
        c = 0
        print(col)
        for i in range(1,len(features)):
            tempY = fwd_rets.iloc[max(i-lookback_wdw,0):i][col]
            tempX = features_combined.iloc[max(i-lookback_wdw,0):i]
            testX = features_combined.iloc[[i]]
            testX = pd.DataFrame(testX)
            if model =='linear':
                model_apply = LinearRegression()
                model_apply.fit(tempX,tempY)
            elif model == 'eNet':
                try:
                    model_apply = ElasticNetCV()#(alpha = 0.2,l1_ratio=0.5)
                    model_apply.fit(tempX,tempY)
                except:
                    model_apply = ElasticNet(alpha = 0.1,l1_ratio = 0.5)
                    model_apply.fit(tempX,tempY)
            elif model =='xgb':
                model_apply = XGBRegressor(verbosity = 0,eval_metric = 'rmse',n_estimators = 20)
                #print('xgb')
                model_apply.fit(tempX,tempY)
            #model.fit(tempX,tempY)
            pred = model_apply.predict(testX)
            ret_preds.append(pred[0])
            c+=1
        ret_pred = pd.DataFrame(ret_preds,index = fwd_rets.index,columns = [col])
        ret_df.append(ret_pred)
    print(ret_df)
    return pd.concat(ret_df,axis = 1)


def calculate_metrics(returns_series):
    """
    Calculate financial metrics from a returns series.
    
    Args:
        returns_series (pd.Series): Series of returns with datetime index
        
    Returns:
        pd.DataFrame: DataFrame containing:
            - rolling_6m_returns: 6-month rolling returns
            - zscore_1y: 1-year price z-score
            - rolling_6m_skew: 6-month rolling skewness
    """
    # Convert returns to prices (assuming returns are in decimal form, e.g., 0.01 for 1%)
    prices = (1 + returns_series).cumprod()
    
    # 1. Rolling 6-month returns
    rolling_6m_returns = prices.pct_change(periods=20*6)  # ~6 months (assuming daily data)
    
    # 2. 1-year price z-score
    rolling_1y_mean = prices.rolling(window=252).mean()  # ~1 year (assuming daily data)
    rolling_1y_std = prices.rolling(window=252).std()
    zscore_1y = (prices - rolling_1y_mean) / rolling_1y_std
    
    # 3. Rolling 6-month skew
    rolling_6m_skew = returns_series.rolling(window=126).skew()
    rolling_6m_vol = returns_series.rolling(window=126).std()
    
    # Combine into DataFrame
    result_df = pd.DataFrame({
        'rolling_6m_returns': rolling_6m_returns,
        'zscore_1y': zscore_1y,
        'rolling_6m_skew': rolling_6m_skew,
        'rolling_6m_realized_vol':rolling_6m_vol
    })
    
    return result_df
