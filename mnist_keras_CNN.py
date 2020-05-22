#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[60]:


dataset = mnist.load_data('mymnist.db')


# In[84]:


len(dataset)


# In[85]:


train , test = dataset


# In[86]:


len(train)


# In[87]:


X_train , y_train = train


# In[88]:


X_train.shape


# In[89]:


X_test , y_test = test


# In[90]:


X_test.shape


# In[91]:


img1 = X_train[7]


# In[92]:


img1.shape


# In[93]:


import cv2


# In[94]:


img1_label = y_train[7]


# In[95]:


img1_label


# In[96]:


img1.shape


# In[97]:


import matplotlib.pyplot as plt


# In[98]:


plt.imshow(img1 , cmap='gray')


# In[99]:


img1.shape


# In[100]:


img1_1d = img1.reshape(28*28)


# In[101]:


img1_1d.shape


# In[102]:


X_train.shape


# In[103]:


#X_train_1d = X_train.reshape(-1 , 28*28)
#X_test_1d = X_test.reshape(-1 , 28*28)


# In[104]:


X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)


# In[105]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[106]:


X_train.shape


# In[26]:


y_train.shape


# In[27]:


from keras.utils.np_utils import to_categorical


# In[28]:


y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# In[29]:


y_train_cat


# In[ ]:





# In[30]:


y_train_cat[7]


# In[31]:


from keras.models import Sequential


# In[32]:


from keras.layers import Dense


# In[33]:


model = Sequential()


# In[34]:


from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[35]:


model.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())


# In[36]:


model.add(Dense(units=512, input_dim=28*28, activation='relu'))


# In[37]:


model.summary()


# In[38]:


model.add(Dense(units=256, activation='relu'))


# In[39]:


model.add(Dense(units=128, activation='relu'))


# In[40]:


model.add(Dense(units=32, activation='relu'))


# In[41]:


model.summary()


# In[42]:


model.add(Dense(units=10, activation='softmax'))


# In[43]:


model.summary()


# In[44]:


from keras.optimizers import RMSprop


# In[45]:


model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[46]:


h = model.fit(X_train, y_train_cat, epochs=10)


# In[108]:


scores = model.evaluate(X_test, y_test_cat, verbose=0)


# In[110]:


scores[1]*100


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




