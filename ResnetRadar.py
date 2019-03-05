

from __future__ import print_function
  
from keras.models import Model
from keras.layers import Input, Dense, merge, Activation
from keras.utils import np_utils
import numpy as np
import keras 
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split


np.random.seed(813306)
 
def build_resnet(input_shape, n_feature_maps, nb_classes):
    print ('build conv_x')
    x = Input(shape=(input_shape))
    conv_x = keras.layers.normalization.BatchNormalization()(x)
    conv_x = keras.layers.Conv2D(n_feature_maps, (9, 1), padding='same')(conv_x)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps, (7, 1), padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps, (5, 1), padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps, (1, 1),padding='same')(x)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x)
    print ('Merging skip connection')
    y = merge([shortcut_y, conv_z], mode='sum')
    y = Activation('relu')(y)
     
#    print ('build conv_x')
#    x1 = y
#    conv_x = keras.layers.Conv2D(n_feature_maps*2, (8, 1), border_mode='same')(x1)
#    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
#    conv_x = Activation('relu')(conv_x)
#     
#    print ('build conv_y')
#    conv_y = keras.layers.Conv2D(n_feature_maps*2, (5, 1), border_mode='same')(conv_x)
#    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
#    conv_y = Activation('relu')(conv_y)
#     
#    print ('build conv_z')
#    conv_z = keras.layers.Conv2D(n_feature_maps*2, (3, 1), border_mode='same')(conv_y)
#    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)
#     
#    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
#    if is_expand_channels:
#        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, 1,border_mode='same')(x1)
#        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
#    else:
#        shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
#    print ('Merging skip connection')
#    y = merge([shortcut_y, conv_z], mode='sum')
#    y = Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps*2, (9, 1), padding='same')(x1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps*2, (7, 1), padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps*2, (5, 1), padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, (1, 1),padding='same')(x1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
    print ('Merging skip connection')
    y = merge([shortcut_y, conv_z], mode='sum')
    y = Activation('relu')(y)
     
    full = keras.layers.pooling.GlobalMaxPooling2D()(y)   
    out = Dense(nb_classes, activation='softmax')(full)
    print ('        -- model was built.')
    return x, out
 
       
#X = sio.loadmat('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/ts.mat')
X = sio.loadmat('C:/Users/Hamidreza/Desktop/data/tsnew.mat')
X=X['Data'];
import csv
#with open('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/lab.csv', 'r') as mf:
with open('C:/Users/Hamidreza/Desktop/data/labfall.csv', 'r') as mf:

     re = csv.reader(mf,delimiter=',',quotechar='|')
     re=np.array(list(re))
     label = re.astype(np.float64)
     Y_t=np.squeeze(label) 
  
nb_epochs = 100
x_train, x_test, y_train, y_test = train_test_split(X, Y_t, test_size=0.3)
#y_train =Y_t[:158]
#y_test =Y_t[158:]
#     
#x_train=X[:158]
#x_test=X[158:]
    
    
nb_classes = len(np.unique(y_test))
batch_size = min(x_train.shape[0]/8, 16)
 
#y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
#y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
 
 
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
 
#x_train_mean = x_train.mean()
#x_train_std = x_train.std()
#x_train = (x_train - x_train_mean)/(x_train_std)
#x_test = (x_test - x_train_mean)/(x_train_std)
  
x_train = x_train.reshape(x_train.shape + (1,1,))
x_test = x_test.reshape(x_test.shape + (1,1,))
 
 
x , y = build_resnet(x_train.shape[1:], 16, nb_classes)
model = Model(input=x, output=y)
optimizer = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.save_weights("modelres.h5")
  
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                  patience=100, min_lr=0.001) 
hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
          verbose=1, validation_data=(x_test, Y_test), callbacks = [reduce_lr])

predict = model.predict(x_test)
preds = np.argmax(predict, axis=1)
log = pd.DataFrame(hist.history)
print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])

labels = {1:'Non-Fall', 2:'Fall'}
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(preds, y_test,
                            target_names=[l for l in labels.values()]))

conf_mat = confusion_matrix(preds, y_test)

fig = plt.figure(figsize=(2,2))
width = np.shape(conf_mat)[1]
height = np.shape(conf_mat)[0]

res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
for i, row in enumerate(conf_mat):
    for j, c in enumerate(row):
        if c>0:
            plt.text(j-.2, i+.1, c, fontsize=16)
            
#cb = fig.colorbar(res)
plt.title('Confusion Matrix')
_ = plt.xticks(range(2), [l for l in labels.values()], rotation=90)
_ = plt.yticks(range(2), [l for l in labels.values()])

plt.plot(preds, 'b.')
plt.plot(y_test, 'r.')

#from sklearn.metrics import roc_curve, auc
#fpr, tpr, thresholds = roc_curve(preds, y_test)
#roc_auc = auc(fpr, tpr)

#plt.figure()
#plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
#plt.legend(loc="lower right")
#plt.show()