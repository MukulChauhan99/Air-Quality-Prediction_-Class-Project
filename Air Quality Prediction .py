#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

X=pd.read_csv('./Linear_X_Train.csv')
Y=pd.read_csv('./Linear_Y_Train.csv')

#convert pandas into numpy 
X=X.values
Y=Y.values

#Normalization 

u=X.mean()
std=X.std() 

X=(X-u)/std

#visualize 
plt.style.use('seaborn')
plt.scatter(X,Y,alpha=0.5)
plt.show()


# In[3]:


## Linear Regression 

def hypothesis(x,theta):
   y_=theta[0]+theta[1]*x 
   return y_


# In[4]:


def gradient(X,Y,theta):
    m=X.shape[0]
    grad=np.zeros((2,))
    
    for i in range(m):
        y_=hypothesis(X[i],theta)
        grad[0]+=(y_-Y[i])
        grad[1]+=(y_-Y[i])*X[i]
        
    return grad/m
    


# In[5]:


def error(X,Y,theta):

    total_error=0.0
    m=X.shape[0]
    
    for i in range(m):
        y_=hypothesis(X[i],theta)
        total_error+=(y_-Y[i])**2
        
    return total_error/m


# In[6]:


def gradientDescent(X,Y,lr=0.1):
    theta=np.zeros((2,))
    max_steps=100
    error_list=[]
    
    for i in range(max_steps):
         
            
        grad=gradient(X,Y,theta)
        e=error(X,Y,theta)
        error_list.append(e)
        
        theta[0]=theta[0]-lr*grad[0]
        
        theta[1]=theta[1]-lr*grad[1]
        
        
    return theta,error_list


# In[ ]:



    


# In[7]:


theta,error_list=gradientDescent(X,Y)


# In[8]:


theta


# In[9]:


print(error)


# In[10]:


error_list


# In[11]:



#Predictions and BestLine

y_=hypothesis(X,theta)


# In[12]:


#Training + Prediction 

plt.scatter(X,Y)
plt.style.use('seaborn')

plt.plot(X,y_,color='orange')


# In[21]:


X_test=pd.read_csv('Linear_X_Test_sub.csv').values
y_test=hypothesis(X_test,theta)

df=pd.DataFrame(data=y_test,columns=["y"])
df.to_csv('Y_prediction.csv',index=False)


# In[22]:


def r2score(Y,Y_):
    
    num=np.sum((Y-Y_)**2)
    den=np.sum((Y-Y_.mean())**2)
    
    score=1-(num/den)
    
    return score*100
    
    


# In[15]:


print(r2score(Y,y_))


# In[ ]:




