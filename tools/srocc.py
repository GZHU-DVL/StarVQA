# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:42:05 2022

@author: Matebook D 15
"""



import torch
import json
import numpy as np
import pandas as pd
from sklearn import svm 
import math
import csv 

test_path=r"C:\Users\Matebook D 15\Desktop\data\LSVQ\train.csv"

mos=[]
csv_reader=csv.reader(open(test_path))
for row in csv_reader:
    mos.append(float(row[1])/20)

# codebook=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
# codebook=[0,2.5,5]
# codebook=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5]
# codebook=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5]
codebook=[0,1,2,3,4,5]

pro_label=[]
for j in range(0,len(mos)):
    D=[]
    for i in range(0,len(codebook)):
        dis_mos=math.exp(-(mos[j]-codebook[i])**2)
        D.append(dis_mos)
    pro_label.append(np.array(D)/sum(D))
model_svr=svm.SVR()
model_svr.fit(pro_label,mos)
train_mos=model_svr.predict(pro_label)
srocc = pd.DataFrame({'A':(train_mos),'B':mos})
print(srocc.corr('spearman'))
print(srocc.corr())


path_file=r'C:\Users\Matebook D 15\Desktop\StarVQA\tabel1\LSVQ\1k\25\idx.json'

path=open(path_file,mode='r')
index=json.load(path)  
index_first=[]
for i in range(0,len(index)) :
    for j in range(0,len(index[i])):
        index_first.append(index[i][j])
aaa,bbb=torch.sort(torch.tensor(index_first))

path_file=r'C:\Users\Matebook D 15\Desktop\StarVQA\tabel1\LSVQ\1k\25\value.json'
path=open(path_file,mode='r')
labels_encode=json.load(path)  
label_first=[]
label=[]
for i in range(0,len(labels_encode)) :
    for j in range(0,len(labels_encode[i])):
        label_first.append(labels_encode[i][j])  
preds_mos_ori=model_svr.predict(label_first)
# preds_mos_ori=label_first

for i in range(0,len(bbb)):
    label.append(preds_mos_ori[bbb[i]])
i=0 
index_co=[] 
while i<len(label):
    if label[i]== label[i+1] and label[i+1] ==label[i+2]:
        i=i+3
    else:
        index_co.append(i)
        i=i+1
for i in range(0,len(index_co)):
    
    del label[index_co[i]]    
          
path_file=r'C:\Users\Matebook D 15\Desktop\StarVQA\tabel1\LSVQ\1k\25\preds.json'
path=open(path_file,mode='r')
labels_encode=json.load(path)  
label_first=[]
set_label=[]
for i in range(0,len(labels_encode)) :
    for j in range(0,len(labels_encode[i])):
            label_first.append(labels_encode[i][j]) 
p_mos=model_svr.predict(label_first)
preds_mos=[]
for i in range(0,len(bbb)):
    preds_mos.append(p_mos[bbb[i]])

for i in range(0,len(index_co)):
    
    del preds_mos[index_co[i]] 
    
mos=[]
for i in range(0,len(label)):
    if i%3==0:
        mos.append(label[i])

mos_1=[]
mos_2=[]
mos_3=[]
for i in range(0,len(preds_mos)):
    if i%3==0:
        mos_1.append(preds_mos[i])
    if i%3==1:
        mos_2.append(preds_mos[i])
    if i%3==2:
        mos_3.append(preds_mos[i])        
        
  
mos_4=np.mean([mos_1,mos_2,mos_3],axis=0)

srocc = pd.DataFrame({'A':mos,'B':mos_4})

# import scipy.io
# scipy.io.savemat('result.mat', mdict={'bData': mos, 'aData': mos_4,})
print(srocc.corr('spearman'))
print(srocc.corr())