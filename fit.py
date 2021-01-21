from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.models import load_model

from imread import imread
from change_brightness import change_brightness
from optic_flow import optic_flow
from model import model as make_model

import numpy as np
import sys

from globl import h, w, label

lr = 1e-4
epoch = 100

X_len = 16000
V_len = 3000

batch_size_x = 32
batch_size_v = 100

batch_num_x = X_len // batch_size_x
batch_num_v = V_len // batch_size_v

index_x = np.arange(X_len)
index_v = np.arange(X_len, X_len + V_len)

def generator_x():
    x = np.zeros((batch_size_x, h, w, 3))
    y = np.zeros((batch_size_x))

    c = 0
    while True:
        if c == 0:
            np.random.shuffle(index_x)

        mini = index_x[batch_size_x * c: batch_size_x * (c + 1)]

        for i in range(len(mini)):
            bf = np.random.uniform(0.2, 1)

            j = mini[i]

            curr = imread(j)
            next = imread(j + 1)

            curr = change_brightness(curr, bf)
            next = change_brightness(next, bf)

            diff = optic_flow(curr, next)

            x[i] = diff
            y[i] = np.mean(label[j: j + 2])

        yield (x, y)

        if c == batch_num_x - 1:
            c = 0
        else:
            c += 1

def generator_v():
    x = np.zeros((batch_size_v, h, w, 3))
    y = np.zeros((batch_size_v))

    c = 0
    while True:
        slice = index_v[batch_size_v * c: batch_size_v * (c + 1)]

        for i in range(len(slice)):
            j = slice[i]

            curr = imread(j)
            next = imread(j + 1)

            diff = optic_flow(curr, next)

            x[i] = diff
            y[i] = np.mean(label[j: j + 2])

        yield (x, y)

        if c == batch_num_v - 1:
            c = 0
        else:
            c += 1

adam = Adam(lr)

es = EarlyStopping(monitor='val_loss',
                   min_delta=0.001,
                   patience=100,
                   restore_best_weights=True)

cp = ModelCheckpoint('model.h5',
                     save_best_only=True)

class PrintLoss(Callback):
    def on_epoch_end(self, epoch, logs):
        l = logs['loss']
        v = logs['val_loss']
        print('\n🍺 %d - %f - %f\n' % (epoch, l, v))

if len(sys.argv) == 1:
    model = make_model()
else:
    model = load_model('model.h5')

model.compile(optimizer=adam,
              loss='mse')

try:
    history = model.fit(generator_x(),
                        epochs=epoch,
                        batch_size=batch_size_x,
                        steps_per_epoch=batch_num_x,
                        validation_data=generator_v(),
                        validation_steps=batch_num_v,
                        callbacks=[es, cp, PrintLoss()],
                        verbose=2)

    print(history.history['loss'])
    print(history.history['val_loss'])
except KeyboardInterrupt:
    model.summary()
    sys.exit()
