#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
import scipy as sp
#import matplotlib.pyplot as plt
from scipy import stats
import os
import time
import itertools
from collections import Counter
from os.path import isfile, join
from random import randint, shuffle
import warnings
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.optimizers import SGD
#from keras.callbacks import ModelCheckpoint
import sklearn
from sklearn.preprocessing import StandardScaler
import gc
from joblib import Parallel, delayed 
from multiprocessing import *
import sys, os
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.svm import SVR




sys.path.append(os.path.dirname(os.path.expanduser('../../pers_not.py')))
from pers_not import *
#send_personal_notification('{} started :sunglasses:'.format(sys.argv[0]))


warnings.filterwarnings("ignore")


# In[ ]:

kiii = 'k-SVMLIN'



# # Metodi per processare i dati di input

# In[5]:


def get_features(df1, feature):
    df = df1.copy()
    cols = [x for x in df.columns.values if x.split('_')[1] == feature]
    df = df[cols]
    return df


def apply_filter(filter_, data_):
    filter_.inputformat(data_)
    return filter_.filter(data_)


def rabalta_feature(df1):
    df = df1.copy()
    cols_top = [x.split('_')[0] for x in df.columns.values]
    cols_meth = [x.split('_')[1] for x in df.columns.values]

    col_idx_arr = list(zip(cols_meth, cols_top))
    col_idx = pd.MultiIndex.from_tuples(col_idx_arr)
    df.columns = col_idx

    rab = df.stack()
    rab.index = [rab.index.get_level_values(
        0), rab.index.map('{0[0]}_{0[1]}'.format)]
    rab.reset_index(inplace=True)

    rab.drop(rab.columns[[0]], axis=1, inplace=True)

    rab.rename(columns={'level_1': 'IND'}, inplace=True)
    rab.set_index('IND', inplace=True)

    rab['system'] = [x.split('_')[0] for x in rab.index.values]
    rab['system'] = [x.replace('_', '.') for x in rab['system']]
    rab['topic'] = [x.split('_')[1] for x in rab.index.values]

    return rab


def rabalta_true(df1):
    """
    Questo metodo produce una nuova tabella a partire dal dataframe di input,
    definita dalle colonne 'IND', system', 'topic' e 'AP'.
    """
    df = df1.copy()
    df = df.stack().to_frame().reset_index()
    df.columns = ['system', 'topic', 'AP']

    cols_top = df['system'].values
    cols_top = [x.replace('_', '.') for x in cols_top]
    cols_meth = df['topic'].values

    col_idx_arr = list(zip(cols_top, cols_meth))
    col_idx = pd.MultiIndex.from_tuples(col_idx_arr)
    df.index = col_idx

    df.index = [df.index.get_level_values(
        0), df.index.map('{0[0]}_{0[1]}'.format)]
    df.reset_index(inplace=True)
    df.drop('level_0', 1, inplace=True)
    df.rename(columns={'level_1': 'IND'}, inplace=True)
    df.set_index('IND', inplace=True)
    return df


def move_column(df1, coltomove, position):
    """
    :param df1: data frame
    :param coltomove: colonna da spostare
    :param position: posizione in cui collocare la colonna
    :return: data frame aggiornato
    """
    df = df1.copy()
    cols = list(df)
    cols.insert(position, cols.pop(cols.index(coltomove)))
    df = df.ix[:, cols]
    return df


# # Dataset

# In[6]:


collections = ['TREC3', 'TREC5', 'TREC6', 'TREC7', 'AH99', 'TREC2001',
               'R04', 'TB04', 'R05', 'TB05', 'TB06', 'WEB11', 'WEB12', 'WEB13', 'WEB14gd']
collectionsy = ['1994', '1996', '1997', '1998', '1999', '2001', '2004',
                '2004', '2005', '2005', '2006', '2011', '2012', '2013', '2014']
rev_collections = list(reversed(collections))

collections_DATA = pd.DataFrame(columns=['collection', 'year'])
collections_DATA['collection'] = collections
collections_DATA['year'] = collectionsy

collections_DATA['collection'] = collections_DATA['collection'].astype(str)
collections_DATA['year'] = collections_DATA['year'].astype(int)


# # Machine Learning
# 

# In[7]:


def prepara(collection, s_tablepath, s_dataframesmlpath):

    truemat = pd.read_csv(s_tablepath + 'Tables/' +
                          collection + '.csv', index_col=0)
    # truemat = truemat.sub(truemat.mean(axis=0), axis=1)
    truemat.sort_index(inplace=True)
    truemat = rabalta_true(truemat)
    m_nodiv = s_dataframesmlpath + '{}_Features.pickle'.format(collection)
    m_nodiv = pd.read_pickle(m_nodiv)

    feats = np.unique([x.split('_')[1] for x in m_nodiv.columns])

    m_nodiv_norm = pd.DataFrame()

    for feat in feats:
        giuste = [x for x in m_nodiv.columns if x.split('_')[1] == feat]
        subs = m_nodiv[giuste]
        # subs = subs.sub(subs.mean(axis=0), axis=1)
        m_nodiv_norm = pd.concat([m_nodiv_norm, subs], 1)

    m_nodiv = rabalta_feature(m_nodiv_norm)
    systems = np.unique(m_nodiv['system'])
    merged = pd.merge(truemat[['AP']], m_nodiv,
                      left_index=True, right_index=True)
    merged = move_column(merged, 'AP', len(merged.columns.values))
    
    merged.drop(['system', 'topic'], 1, inplace=True)

    return merged


# In[ ]:





# In[61]:




def fun(args):
    collection_eval, others_coll = args
    
    rev_collections = list(reversed(collections))

    s_tablepath = '../src/'
    s_dataframesmlpath = '../src/features/'
    improve_kind = 'base'
    
    df_result = pd.DataFrame(
    columns=['train_on', 'eval_on', 'improve_kind', 'kind', 'df'])

    m_test = prepara(collection_eval, s_tablepath, s_dataframesmlpath)
    for i_others, collection in enumerate(others_coll):
        au = prepara(collection, s_tablepath, s_dataframesmlpath)
        if i_others == 0:
            m_train = pd.DataFrame(columns=au.columns.values)
            m_train = pd.concat([m_train, au])
        else:
            m_train = pd.concat([m_train, au])

    m_train = move_column(m_train, 'AP', len(m_train.columns.values))
    col_order = m_train.columns
    m_test = m_test[col_order]
    m_train = m_train.fillna(0.0)
    m_test = m_test.fillna(0.0)

    ind_res = m_test.index.values

    #display(m_train.head())
    #display(m_test.head())
    
    
    ###########################
    
    train_values = m_train.values
    last_train_column_index = train_values[0, :].size - 1
    x_train = train_values[:, 0:last_train_column_index]
    y_train = train_values[:, last_train_column_index]

    test_values = m_test.values
    last_test_column_index = test_values[0, :].size - 1
    x_test = test_values[:, 0:last_test_column_index]
    y_test = test_values[:, last_test_column_index]
    
    ####################################
    
    # vectorize
    #x_train = json.loads(  x_train.to_json(orient='records')  )
    #x_test = json.loads(  x_test.to_json(orient='records')  )
    #v = DictVectorizer()
    #X_tr = v.fit_transform(x_train)
    #X_te = v.fit_transform(x_test)


    #model
    model = SVR(kernel='linear', C=1e3)

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    
    
    
    #print(predictions)
    #print(y_test)
    print('train: {} - eval: {} '.format(others_coll[0], collection_eval )   )
    

    df_res = pd.DataFrame(index=ind_res)
    df_res['actual'] = y_test
    df_res['predicted'] = predictions
    df_res['system'] = [x.split('_')[0] for x in df_res.index.values]
    df_res['topic'] = [x.split('_')[1] for x in df_res.index.values]

    df_result.loc[len(df_result)] = [others_coll[0],
                                     collection_eval, 'Base', kiii, df_res]
    
    df_result.to_pickle('./_pickles/partial/singletrain-train{}-eval{}-alg{}.pickle'.format(others_coll[0], 
                                                                       collection_eval,
                                                                       kiii))
    
    gc.collect()
    return 1
    
    

arg_instances = []
for collection_eval in collections:
    for others_coll in collections:
    	if collection_eval != others_coll:
	        arg_instances.append(   (collection_eval, [others_coll])    )
	        arg_instances.append(   (others_coll, [collection_eval])    )
    
    
print(arg_instances)
#assert False
results = Parallel(n_jobs=10,verbose=5)(delayed(fun)(arg) for arg in arg_instances)



for i, (collection_eval, others_coll) in enumerate(arg_instances):
    sub = pd.read_pickle('./_pickles/partial/singletrain-train{}-eval{}-alg{}.pickle'.format(others_coll[0], collection_eval, kiii))
    if i==0:
        dfres = pd.DataFrame(columns=sub.columns)
    dfres = pd.concat([dfres,sub])
dfres.to_pickle('./_pickles/singletrain-{}.pickle'.format(kiii))
#display(dfres)



send_personal_notification('{} finished :sunglasses:'.format(sys.argv[0]))
print('done.')


# In[ ]:





# In[ ]:





# In[ ]:




