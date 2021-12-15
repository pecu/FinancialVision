import numpy as np
import tensorflow as tf
import pickle
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from sklearn.metrics import confusion_matrix

# Baseline model
class Baseline():
  def __init__(self,learning_rate,epochs):
    self.learning_rate=learning_rate
    self.epochs = epochs
  
  def load_candlestick(self):
    fn = "ETH_gaf.pkl"
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return (data['train_culr_gaf'], data['train_onehot'], data['val_culr_gaf'], data['val_onehot'], data['test_culr_gaf'], data['test_onehot']) 
  
  def baseline_model(self):
    model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(16, 2,
                                  strides=(1, 1),
                                  padding='same',
                                  activation='sigmoid',
                                  input_shape=(10, 10, 4)),
          
          tf.keras.layers.Conv2D(16, 2,
                                  strides=(2, 2),
                                  padding='same',
                                  activation='sigmoid'),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.Dense(8,activation="softmax")
        ])
    return model

  def baseline_train(self):
    train_data, train_labels, val_data, val_labels, test_data, test_labels = self.load_candlestick()

    optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate,momentum= 0.9,nesterov=True)
  
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
    model = self.baseline_model()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    self.history = model.fit(train_data.astype(np.float32), train_labels.astype(np.float32),
          epochs=self.epochs,
          validation_data=(val_data.astype(np.float32), val_labels.astype(np.float32)),
          batch_size=100)

    test_pred = model.predict(test_data)
    test_pred = np.argmax(test_pred, axis=1)
    test_true = np.argmax(test_labels, axis=1)
    test_result_cm = confusion_matrix(test_true, test_pred, labels=range(8))
    print(test_result_cm)
    count = 0
    for r in range(8):
        count += test_result_cm[r, r] 
    print('testing accuracy:', count/np.sum(test_result_cm))

# DP-SGD Model
class Training():
  def __init__(self,l2_norm_clip,noise_multiplier,learning_rate,epochs):
    self.l2_norm_clip=l2_norm_clip
    self.noise_multiplier=noise_multiplier
    self.learning_rate=learning_rate
    self.epochs = epochs
    
  def load_candlestick(self):
    fn = "ETH_gaf.pkl"
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return (data['train_culr_gaf'], data['train_onehot'], data['val_culr_gaf'], data['val_onehot'], data['test_culr_gaf'], data['test_onehot']) 
    
  def create_model(self):
    model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(16, 2,
                                  strides=(1, 1),
                                  padding='same',
                                  activation='relu',
                                  input_shape=(10, 10, 4)),
          
          tf.keras.layers.Conv2D(16, 2,
                                  strides=(2, 2),
                                  padding='same',
                                  activation='relu'),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.Dense(8,activation="softmax")
        ])
    return model

  def compute_epsilon(self,steps):
    """Computes epsilon value for given hyperparameters."""
    # if noise_multiplier == 0.0:
    #     return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = 100 / 13170
    rdp = compute_rdp(q=sampling_probability,
                      noise_multiplier=self.noise_multiplier,
                      steps=steps,
                      orders=orders)
    # Delta is set to 1e-5 because MNIST has 60000 training points.
    return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

  def star_train(self):
    train_data, train_labels, val_data, val_labels, test_data, test_labels = self.load_candlestick()

    optimizer = DPKerasSGDOptimizer(
      l2_norm_clip=self.l2_norm_clip,
      noise_multiplier=self.noise_multiplier,
      num_microbatches=30,
      learning_rate=self.learning_rate, 
      momentum=0.9
      )
  
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
    model = self.create_model()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    self.history = model.fit(train_data.astype(np.float32), train_labels.astype(np.float32),
          epochs=self.epochs,
          validation_data=(val_data.astype(np.float32), val_labels.astype(np.float32)),
          batch_size=30)

    test_pred = model.predict(test_data)
    test_pred = np.argmax(test_pred, axis=1)
    test_true = np.argmax(test_labels, axis=1)
    test_result_cm = confusion_matrix(test_true, test_pred, labels=range(8))
    print(test_result_cm)
    count = 0
    for r in range(8):
        count += test_result_cm[r, r] 
    print('testing accuracy:', count/np.sum(test_result_cm))
  
    self.eps = self.compute_epsilon(self.epochs * train_data.shape[0] // 100)
    print('For delta=1e-5, the current epsilon is: %.2f' % self.eps)


if __name__ == "__main__":
    # Hyperparameters 
    b_learning_rate = 0.0006
    epochs = 120
    l2_norm_clip = [1, 1.5]
    learning_rate = 0.0006
    noise_multiplier = [0.1, 0.3, 0.5, 0.7, 1]
    
    baseline = Baseline(b_learning_rate, epochs)
    baseline.baseline_train()
    # DP-SGD main loop
    for l in l2_norm_clip:
        for noise in noise_multiplier:
            print("=" * 13, "l2_norm_clip = ", l, " ", "noise=", noise, "=" * 13)
            a = Training(l, noise, learning_rate, epochs)
            a.star_train()