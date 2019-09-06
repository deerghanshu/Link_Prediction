#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[5]:


train=pd.read_csv("train.csv")
feat=pd.read_csv("user_features.csv")


# In[6]:


feat.shape


# In[7]:


feat.set_index(['node_id'],inplace=True)


# In[8]:


feat.tail(5)


# In[9]:


for column in feat:
    feat[column]= feat[column] - feat[column].mean()
print(feat.head())


# In[10]:


feat.f1/=31
feat.f2/=31
feat.f3/=31
feat.f4/=31
feat.f5/=31
feat.f6/=31
feat.f7/=31
feat.f8/=31
feat.f9/=31
feat.f10/=31
feat.f11/=31
feat.f12/=31
feat.f13/=15


# In[11]:


feat.tail(5)


# In[12]:


train.tail(5)


# In[13]:


sum(train.is_chat)


# In[14]:


train[:10]


# In[15]:


m=train.shape[0]
n=m/4096.0
mb={}
for i in range(int(n)):
    mb[i]=train[4096*i:4096*(i+1)]
    print(sum(mb[i].is_chat))
mb[int(n)]=train[4096*int(n):m]
print(sum(mb[int(n)].is_chat))


# In[16]:


#number of minibatches
len(mb)


# In[17]:


temp=np.asarray(mb[i])
fda=abs(np.asarray(feat.loc[temp[:,0]])-np.asarray(feat.loc[temp[:,1]]))
yval=np.asarray(temp[:,2])
yvalt=yval>0.5
yvalt=np.array(yvalt)
print(type(yvalt))


# In[18]:


ttrain = train[train['is_chat'] == 1]
ftrain = train[train['is_chat'] == 0]
print(ttrain.head())
print(ttrain.shape)
print(ftrain.head())
print(ftrain.shape)


# In[19]:


ftrain1 = ftrain[(ftrain.index)%20==0]
print(ftrain1.head())
print(ftrain1.shape)


# In[20]:


big = pd.concat([ttrain, ftrain1], ignore_index=True)
big = big.sample(frac=1).reset_index(drop=True)
big


# In[21]:


big.head()


# In[22]:


m1=big.shape[0]
n1=m1/4096.0
mb1={}
for i in range(int(n1)):
    mb1[i]=big[4096*i:4096*(i+1)]
    print(sum(mb1[i].is_chat))
mb1[int(n1)]=big[4096*int(n1):m1]
print(sum(mb1[int(n1)].is_chat))


# In[23]:


len(mb1)


# In[24]:


#neural network using
#fda as X,yval as Y
#two hidden layers 
#parameters={}
W1 = np.random.rand(8,13)
b1 = np.zeros((8,1))
W2 = np.random.rand(3,8)
b2 = np.zeros((3,1))
W3 = np.random.rand(1,3)
b3 = np.zeros((1,1))
#parameters["W1"]=W1
#parameters["W2"]=W2
#parameters["W3"]=W3
#parameters["b1"]=b1
#parameters["b2"]=b2
#parameters["b3"]=b3


# In[25]:


def sigmoid(Z):
    return 1/(1+np.exp(-Z))


# In[103]:


for i in range(445,1385):
    #preperation of the X,Y values from the dictionary mb
    temp=np.asarray(mb1[i])
    fda=abs(np.asarray(feat.loc[temp[:,0]])-np.asarray(feat.loc[temp[:,1]]))
    yval=np.asarray(temp[:,2])
    X=np.transpose(fda)
    #print(X.shape)
    yval = yval.reshape((4096,1))
    Y=np.transpose(yval)
    #print(Y.shape)
    for j in range(20000):
        #forward propagation

        #W1=parameters["W1"]
        #W2=parameters["W2"]
        #W3=parameters["W3"]
        #b1=parameters["b1"]
        #b2=parameters["b2"]
        #b3=parameters["b3"]

        Z1=np.dot(W1,X)+b1
        A1=sigmoid(Z1)
        Z2=np.dot(W2,A1)+b2
        A2=sigmoid(Z2)
        Z3=np.dot(W3,A2)+b3
        A3=sigmoid(Z3)

        #calculation of cost

        cost = (-1/4096.0)*np.sum(np.multiply(Y,np.log(A3))+np.multiply(1-Y,np.log(1-A3)))
        #print(j,cost)

        #back propagation

        dZ3=(1/4096.0)*(A3-Y)
        dW3=np.dot(dZ3,np.transpose(A2))
        db3=np.sum(dZ3,axis=1,keepdims=True)

        dZ2=np.dot(np.transpose(W3),dZ3)*A2*(1-A2)
        dW2=np.dot(dZ2,np.transpose(A1))
        db2=np.sum(dZ2,axis=1,keepdims=True)

        dZ1=np.dot(np.transpose(W2),dZ2)*A1*(1-A1)
        dW1=np.dot(dZ1,np.transpose(X))
        db1=np.sum(dZ1,axis=1,keepdims=True)

        #updating parameters using learning rate alpha

        W1=W1-0.5*dW1
        W2=W2-0.5*dW2
        W3=W3-0.5*dW3
        b1=b1-0.5*db1
        b2=b2-0.5*db2
        b3=b3-0.5*db3
    print(i,cost)


# In[30]:


def pred(fdat,yvalt):
    Z1t=np.dot(W1,fdat)+b1
    A1t=sigmoid(Z1t)
    Z2t=np.dot(W2,A1t)+b2
    A2t=sigmoid(Z2t)
    Z3t=np.dot(W3,A2t)+b3
    A3t=sigmoid(Z3t)
    
    #A3t=A3t>0.5
    op=np.sum(yvalt)
    pred=A3t*yvalt
    t1=np.sum(A3t)
    p=np.sum(pred)
    A3t = (A3t - A3t.min())/(A3t.max() - A3t.min())
    A3t = A3t>0.5
    return A3t


# In[64]:


tempt=np.asarray(mb1[1300])
fdat=abs(np.asarray(feat.loc[tempt[:,0]])-np.asarray(feat.loc[tempt[:,1]]))
fdat=np.transpose(fdat)
yvalt=np.asarray(tempt[:,2])
yvalt=yvalt.reshape((4096,1))
yvalt=np.transpose(yvalt)
a = pred(fdat,yvalt)
a = a.transpose()


# In[65]:


a = a.reshape(4096)
b = np.array([int(a[i]) for i in range(4096)])


# In[66]:


def score(b,yvalt):
    b = b+1
    yvalt = yvalt+1
    c=b*yvalt
    c=c==2
    print((yvalt.shape[1]-np.sum(c))/yvalt.shape[1])
score(b,yvalt)


# In[67]:


test = pd.read_csv('test.csv')
test.head()


# In[76]:


test = test.drop(['id'], axis=1)
test.head()


# In[104]:


tempt=np.asarray(test)
fdat=abs(np.asarray(feat.loc[tempt[:,0]])-np.asarray(feat.loc[tempt[:,1]]))
fdat=np.transpose(fdat)
Z1t=np.dot(W1,fdat)+b1
A1t=sigmoid(Z1t)
Z2t=np.dot(W2,A1t)+b2
A2t=sigmoid(Z2t)
Z3t=np.dot(W3,A2t)+b3
A3t=sigmoid(Z3t)
print(A3t[:5])


# In[105]:


A3t =A3t.reshape(11776968)
A3t = A3t.tolist()


# In[106]:


len(A3t)


# In[107]:


ans = {'id': [i for i in range(1,test.shape[0]+1)], 'is_chat': A3t}
ans = pd.DataFrame(ans)
ans.set_index(['id'],inplace=True)
ans.head()


# In[108]:


ans.to_csv('ans.csv')


# In[110]:


ans.is_chat.max()


# In[113]:


import matplotlib.pyplot as plt


# In[131]:


ans1 = pd.read_csv('ans1.csv')
ans2 = pd.read_csv('ans.csv')


# In[132]:


z1=ans1.is_chat
z1=z1.tolist()
n, bins, patches = plt.hist(z1, 50, density=True, facecolor='g', alpha=0.75)
plt.show()
z2=ans2.is_chat
z2=z2.tolist()
n, bins, patches = plt.hist(z2, 50, density=True, facecolor='g', alpha=0.75)
plt.show()


# In[ ]:




