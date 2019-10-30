from mancala_new import X_final,Y_final,sample_size
import time, tensorflow as tf, numpy as np, matplotlib.pyplot as plt,warnings
warnings.filterwarnings("ignore")


model = tf.keras.Sequential([
    tf.keras.layers.Dense(28, activation = tf.nn.relu, input_shape=(14,)),
    tf.keras.layers.Dense(14, activation = tf.nn.softmax)
     ])

optimizer = tf.keras.optimizers.Adam(0.01)

model.compile(loss=tf.keras.losses.categorical_crossentropy.__name__, optimizer=optimizer,metrics=['accuracy'])
epochs = 200
#converting X_final and Y_final to a numpy array to make keras happy
X_final = np.array(X_final, ndmin=2).astype(float)
Y_final = np.array(Y_final, ndmin = 2)
ips,ops = X_final,Y_final

history = model.fit(ips,ops,batch_size=sample_size,epochs=epochs,validation_split=0.2,verbose=1)

model.save('new_model.h5')
fig = plt.figure()
p1 = fig.add_subplot(221)
p2 = fig.add_subplot(222)
p3 = fig.add_subplot(223)
p4 = fig.add_subplot(224)
p2.set_ylim(0,1)
p4.set_ylim(0,1)
p1.grid()
p2.grid()
p3.grid()
p4.grid()
p2.set_yticks(np.arange(0,1,0.1))
p4.set_yticks(np.arange(0,1,0.1))
x = [i for i in range(epochs)]
y = history.history['loss']
y2 = history.history['acc']
y3 = history.history['val_loss']
y4 = history.history['val_acc']
p1.plot(x,y, 'r', label='train_loss')
p1.legend()
p2.plot(x,y2, 'b', label='train_accuracy')
p2.legend()
p3.plot(x,y3, 'r', label='val_loss')
p3.legend()
p4.plot(x,y4, 'b', label='val_accuracy')
p4.legend()
plt.show()


