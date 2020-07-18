from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
#from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from scipy import io as spio

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import random

#print(tf.__version__)

# Parameters
mum = 2
M1 = 2
T1 = 1

#model = Sequential()
#add model layers
#model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
#model.add(Conv2D(32, kernel_size=3, activation='relu'))
#model.add(Flatten())
#model.add(Dense(120, activation='relu'))
#model.add(Dense(80, activation='relu'))
#model.add(Dense(60,activation='relu'))
#model.add(Dense(27, activation='softmax'))


def geneticCode(V):
        model = Sequential()
        model.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(28, 28,1)))
        model.add(Conv2D(64,kernel_size=3,activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
	#model.add(Conv2D(64,kernel_size=3,activation='relu'))
	model.add(Conv2D(128, kernel_size=3, activation='relu'))
	model.add(Conv2D(128, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        for v in V:
                if v != 0:
                        model.add(Dense(v, activation='relu'))

        model.add(Dense(27, activation='softmax'))

        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=1)

        validation_loss, validation_acc = model.evaluate(validation_images, validation_labels)

        print('Validation accuracy:', validation_acc)

        return validation_acc



emnist = spio.loadmat('emnist-letters.mat')


X_train = emnist['dataset'][0][0][0][0][0][0]
X_train = X_train.astype(np.float32)
y_train = emnist["dataset"][0][0][0][0][0][1]

print(y_train)

X_test = emnist["dataset"][0][0][1][0][0][0]
X_test = X_test.astype(np.float32)

# load test labels
y_test = emnist["dataset"][0][0][1][0][0][1]

#reshape data to fit model
X_train = X_train.reshape(124800,28,28,1)
X_test = X_test.reshape(20800,28,28,1)


X_train = X_train / 255.
test_images = X_test / 255.




# Validation set
validation_images = X_train[104000:124800]
validation_labels = y_train[104000:124800]

train_images = X_train[0:104000]
train_labels = y_train[0:104000]

train_labels= to_categorical(train_labels)
validation_labels = to_categorical(validation_labels)
test_labels = to_categorical(y_test)

V1 = [344, 100]#[256]
V2 = [361, 85]#[344, 85]
Nset = [V1,V2]

#ma1 = geneticCode(V1)
#ma2 = geneticCode(V2)

ma1 = 0.9334190836162892
ma2 = 0.9338690640740329

# DESCOMENTAR ma = geneticCode(V)

#ma = 0.9302836266582527

maxis = [ma1,ma2]

# Fision binaria

NsetAux = [V1,V2]
maxisAux= [ma1,ma2]

L = []
MAXIS = []

r = False

timeA = time.time()

for t in range(3):

        # Fision binaria

        print("__________________")

        # Depredacion
        l = len(Nset)
        #if l > 5: # PARAMETRO 5


        if t == M1:
                print("Depredacion")
                r = True
                maxiscopy = maxis[:]
                maxiscopy.sort(reverse=True)
                #maxiscopy = np.array(maxiscopy) + 0.0001*np.random.rand(len(maxiscopy))
                print(maxiscopy)
                print(mum) # PAREMETRO 2
                theta = maxiscopy[mum] # VERIFICAR
                print("theta=",theta)
                Nset2 = Nset[:]
                maxis2= maxis[:]
                h = 0
                for i in range(l):
                        if maxis[i] <= theta:
                                print(maxis[i])
                                print(maxis2[i-h])
                                del Nset2[i-h]
                                del maxis2[i-h]
                                h += 1
                Nset = Nset2[:]
                maxis= maxis2[:]
                NsetAux = Nset2[:]
                maxisAux = maxis2[:]
                l = len(Nset)
                L.append(l)

        print("__")
        print(Nset)
        print(maxis)

        for N in Nset: # Reproduccion
                #NsetAux.append(N)
                N1 = N[:]
                alea = random.random()
		k1 = random.randrange(70)
                if alea < 0.7: # Desicion alpha/beta
                        #alpha-mutacion
                        l = len(N)
                        elec = random.randrange(l)
                        if alea < 0.65:
                                N1[elec] += k1
                        else:
                                if N1[elec] - k1 > 0:
                                        N1[elec] -= k1
                                else:
                                        N1[elec] = 0
                else:
                        N1.append(k1)

                # N1 es el hijo. N se divide en N y N1
                #NsetAux.append(N1)

                ban = True
                for i in range(len(NsetAux)): #NsetAux
                        if NsetAux[i] == N1:
                                ban = False
                                #print(NsetAux)
                                #print(maxisAux)
                                m1 = maxisAux[i]

                NsetAux.append(N1)

                if ban:
                        m1 = geneticCode(N1)
                maxisAux.append(m1)



        Nset  = NsetAux[:]
        maxis = maxisAux[:]
        maxis = np.array(maxis) + 0.0001*np.random.rand(len(maxis))
        maxis = list(maxis)
        print("Redes")
        print(Nset)
        print("Maximos")
        print(maxis)
        print
        print
        maxisAux = maxis[:]

        l = len(Nset)
        L.append(l)
        MAXIS.append(max(maxis))

        if r:
                M1 += T1
                r = False

timeB = time.time()

#print(MAXIS)

print(L)

fig,ax = plt.subplots()

ax.plot(MAXIS, color = (0.4,0.7,0.3))
ax.set_title(u"Accuracy")
ax.spines['left'].set_position(('outward',10))
ax.spines['bottom'].set_position(('outward',10))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_yticks(range(10),minor=True)
#ax.set_ylim([0,100])
ax.legend(framealpha=1)
#ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.legend()#loc="upper left", bbox_to_anchor=(0.8,0.2))
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.show()
#plt.plot(MAXIS)
#plt.show()

print(max(MAXIS))

fig,ax = plt.subplots()
ax.plot(L, color = (0.4,0.3,0.7))
ax.spines['left'].set_position(('outward',10))
ax.spines['bottom'].set_position(('outward',10))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_yticks(range(10),minor=True)
#ax.set_ylim([0,100])
ax.legend(framealpha=1)
#ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.legend()#loc="upper left", bbox_to_anchor=(0.8,0.2))
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.show()
#plt.plot(MAXIS)
#plt.show()

print(L)

print(timeB-timeA)
