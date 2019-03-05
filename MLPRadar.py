
from __future__ import print_function
 
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import np_utils
import numpy as np
import keras 
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


np.random.seed(813306)

X = sio.loadmat('C:/Users/Hamidreza/Desktop/data/tsnew.mat')
#X = sio.loadmat('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/ts.mat')
X=X['Data'];
import csv
#with open('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/lab.csv', 'r') as mf:
with open('C:/Users/Hamidreza/Desktop/data/labfall.csv', 'r') as mf:

     re = csv.reader(mf,delimiter=',',quotechar='|')
     re=np.array(list(re))
     label = re.astype(np.float64)
     Y_t=np.squeeze(label) 
  
nb_epochs =200

x_train, x_test, y_train, y_test = train_test_split(X, Y_t, test_size=0.5)


nb_classes =len(np.unique(y_test))
batch_size = min(x_train.shape[0]/8, 16)

y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
 
x_train_mean = x_train.mean()
x_train_std = x_train.std()
x_train = (x_train - x_train_mean)/(x_train_std)
x_test = (x_test - x_train_mean)/(x_train_std) 

#scaler = MinMaxScaler(feature_range=(0, 1))
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.fit_transform(x_test)

x = Input(x_train.shape[1:])
y= Dropout(0.2)(x)
y = Dense(200, activation='relu')(x)
y = Dropout(0.2)(y)
y = Dense(200, activation='relu')(y)
y = Dropout(0.2)(y)
y = Dense(200, activation = 'relu')(y)
y = Dropout(0.2)(y)
out = Dense(nb_classes, activation='softmax')(y)
 
model = Model(input=x, output=out)
 
optimizer = keras.optimizers.Adam()    
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
 
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                  patience=200, min_lr=0.001)

hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(x_test, Y_test), 
            #callbacks = [TestCallback((x_train, Y_train)), reduce_lr, keras.callbacks.TensorBoard(log_dir='./log'+fname, histogram_freq=1)])
             callbacks=[reduce_lr])
#model.evaluate(x_test, Y_test, batch_size=batch_size, verbose=1, sample_weight=None)
#model.fit_generator((x_test, Y_test), samples_per_epoch=2947, nb_epoch=nb_epochs, verbose=1)
predict = model.predict(x_test)
preds = np.argmax(predict, axis=1)
#Print the testing results which has the lowest training loss.
log = pd.DataFrame(hist.history)
print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])

print(hist.history.keys())
# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

labels = {1:'Non-Fall', 2:'Fall'}
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(preds, y_test)

fig = plt.figure(figsize=(2,2))
width = np.shape(conf_mat)[1]
height = np.shape(conf_mat)[0]

res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
for i, row in enumerate(conf_mat):
    for j, c in enumerate(row):
        if c>0:
            plt.text(j-.2, i+.1, c, fontsize=16)
            
cb = fig.colorbar(res)
plt.title('Confusion Matrix')
_ = plt.xticks(range(2), [l for l in labels.values()], rotation=90)
_ = plt.yticks(range(2), [l for l in labels.values()])




