#%%
import numpy as np
import pandas as pd
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D,Flatten,Reshape, Input, LSTM
from keras.optimizers import Adam
from keras import Model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Conv2D, Dense, Reshape, Input, LSTM
from keras import Model, backend
import tensorflow as tf
#%%
ori_data = pd.read_csv("./data/FI2010.csv", index_col=0)
data = ori_data.drop(['STOCK','DAY','PRICE_MID'],axis=1)
lob = data.values
lob = lob.reshape(-1,2)
lob = (lob - lob.mean(axis=0)) / lob.std(axis=0)
lob = lob.reshape(-1,40)
lob_df = pd.DataFrame(lob)
lob_df['mid'] = ori_data['PRICE_MID']
print("finish")

#%%
p = 50
k = 50
alpha = 0.0003
# パラメータをもとに仲値からラベルを作成する。
lob_df['lt'] = (lob_df['mid'].rolling(window=k).mean().shift(-k)-lob_df['mid'])/lob_df['mid']
lob_df = lob_df.dropna()
lob_df['label'] = 0
lob_df.loc[lob_df['lt']>alpha, 'label'] = 1
lob_df.loc[lob_df['lt']<-alpha, 'label'] = -1
# %%
X = np.zeros((len(lob_df)-p+1, p, 40, 1))
lob = lob_df.iloc[:,:40].values
for i in range(len(lob_df)-p+1):
    X[i] = lob[i:i+p,:].reshape(p,-1,1)
y = to_categorical(lob_df['label'].iloc[p-1:], 3)
print(X.shape, y.shape)
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# %%
import tensorflow as tf
tf.Graph()
backend.clear_session()

inputs = Input(shape=(p,40,1))
x = Conv2D(8, kernel_size=(1,2), strides=(1,2), activation='relu')(inputs)
x = Conv2D(8, kernel_size=(1,2), strides=(1,2), activation='relu')(x)
x = Conv2D(8, kernel_size=(1,10), strides=1, activation='relu')(x)
x = Reshape((p, 8))(x)
x = LSTM(8, activation='relu')(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(3, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
epochs = 50
batch_size = 256
history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(X_test, y_test))

# %%
import matplotlib.pyplot as plt
hist_loss = history.history['loss']

hist_acc = history.history['accuracy']

plt.plot(np.arange(len(hist_loss)),
         hist_loss,
         label='loss'
         )

plt.plot(np.arange(len(hist_acc)),
         hist_acc,
         label='accuracy'
         )

plt.legend()
plt.show()
# %%
