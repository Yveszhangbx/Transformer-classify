#!/usr/bin/env python
# coding: utf-8

# In[3]:

import sys
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import os


# In[ ]:


inpath = sys.argv[1]
outpath = sys.argv[2]

#Positive sample set
buys = pd.read_table(inpath+'/yoochoose-buys.dat',
names=['s_id','time_strap','i_id','price','quantity'],sep=',',dtype={'s_id':np.int64,'i_id':np.int64,'price':np.int64,'quantity':np.int64})
buy=np.unique(buys.s_id)

#Convert the time stramp
def timeconv(string):
    pattern = re.compile(r'\d+')
    m = pattern.findall(string)[1:4]
    m[2] = str(int(m[2])//4)
    return ''.join(m)


#Concatenate data in same session
def to_bin(df,ts,ind,cat,y,buy):
    last_id = -1
    new_t,new_i,new_c=[],[],[]
    for index,row in df.iterrows():
        s_id = row.s_id
        if s_id != last_id:
            ts.append(new_t)
            ind.append(new_i)
            cat.append(new_c)
            new_t=[row.time_strap]
            new_i=[row.i_id]
            new_c=[row.c_id]
            if s_id in buy:
                y.append(1)
            else:
                y.append(0)
        else:
            new_t.append(row.time_strap)
            new_i.append(row.i_id)
            new_c.append(row.c_id)
        last_id = s_id
    
    ts.append(new_t)
    ind.append(new_i)
    cat.append(new_c)

def preprocess(data,d_type):
    #Train data
    raw_data = pd.read_table(inpath+'/yoochoose-{0}.dat'.format(data),
    names=['s_id','time_strap','i_id','c_id'],error_bad_lines=False,
                       sep=',',chunksize=1000000,converters={'time_strap':timeconv},dtype={'i_id':str,'c_id':str})
    data_lst=[]
    for chunk in raw_data:
        chunk = chunk[chunk.i_id.notnull()&chunk.c_id.notnull()]  # filter out rows with NaN
        data_lst.append(chunk)
    Data = pd.concat(data_lst,axis=0)
    
    
    for i in range(34):
        ts=[]
        ind=[]
        cat=[]
        y=[]
        to_bin(Data.iloc[1000000*i:1000000*(i+1)],ts,ind,cat,y,buy)
        ts[:2]=[ts[1]]
        ind[:2]=[ind[1]]
        cat[:2]=[cat[1]]
        with open(outpath+'/{0}/time{1}.txt'.format(d_type,i),'w') as f:
            for j in ts:
                f.write(str(j).replace('[','').replace(']','').replace("'",'')+'\n')
        with open(outpath+'/{0}/idx{1}.txt'.format(d_type,i),'w') as f:
            for j in ind:
                f.write(str(j).replace('[','').replace(']','').replace("'",'')+'\n')
        with open(outpath+'/{0}/cat{1}.txt'.format(d_type,i),'w') as f:
            for j in cat:
                f.write(str(j).replace('[','').replace(']','').replace("'",'')+'\n')
        with open(outpath+'/{0}/y{1}.txt'.format(d_type,i),'w') as f:
            f.write(str(y).replace('[','').replace(']',''))

preprocess('clicks','train_data')
preprocess('test','test_data')
