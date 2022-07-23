import numpy as np
import pickle
import tensorflow as tf
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix


def partition_dataset(data, labels, nb_teachers, teacher_id):
  """
  Simple partitioning algorithm that returns the right portion of the data
  needed by a given teacher out of a certain nb of teachers
  :param data: input data to be partitioned
  :param labels: output data to be partitioned
  :param nb_teachers: number of teachers in the ensemble (affects size of each
                      partition)
  :param teacher_id: id of partition to retrieve
  :return:
  """

  # Sanity check
  assert len(data) == len(labels)
  assert int(teacher_id) < int(nb_teachers)

  # This will floor the possible number of batches
  batch_len = int(len(data) / nb_teachers)

  # Compute start, end indices of partition
  start = teacher_id * batch_len
  end = (teacher_id+1) * batch_len

  # Slice partition off
  np.random.seed(5)
  data_label = list(zip(data, labels))
  np.random.shuffle(data_label)
  data[:], labels[:] = zip(*data_label)

  partition_data = data[start:end]
  partition_labels = labels[start:end]

  return partition_data, partition_labels

def load_candlestick():
    fn = "ETH_gaf.pkl"
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return (data['train_culr_gaf'], data['train_onehot'], data['val_culr_gaf'], data['val_onehot'], data['test_culr_gaf'], data['test_onehot'])

def create_model():
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
  
def training(model, train_data, train_labels, val_data, val_labels):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0006,momentum= 0.9,nesterov=True)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
    model = create_model()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    history = model.fit(train_data.astype(np.float32), train_labels.astype(np.float32),
          epochs=100,
          validation_data=(val_data.astype(np.float32), val_labels.astype(np.float32)),
          batch_size=100)
    
    return (model, history.history)

def train_teacher (nb_teachers, teacher_id):
  """
  This function trains a single teacher model with responds teacher's ID among an ensemble of nb_teachers
  models for the dataset specified.
  The model will be save in directory. 
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # Retrieve subset of data for this teacher
  train_data, train_labels,val_data,val_labels, test_data, test_labels = load_candlestick()
  data, labels = partition_dataset(train_data, train_labels,
                                            nb_teachers,
                                            teacher_id)

  print("Length of training data: " + str(len(labels)))

  # Define teacher checkpoint filename and full path
  filename2 = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.h5'
 
  # Perform teacher training need to modify 

  # Create teacher model
  model = create_model()
  model, hist = training(model, data, labels, val_data, val_labels)
  model.save(filename2)
  
  print("Saved model to disk")
  return hist

def labels_from_probs(probs):
  """
  Helper function: computes argmax along last dimension of array to obtain
  labels (max prob or max logit value)
  :param probs: numpy array where probabilities or prob are on last dimension
  :return: array with same shape as input besides last dimension with shape 1
          now containing the labels
  """
  # Compute last axis index
  last_axis = len(np.shape(probs)) - 1
  
  # Label is argmax over last dimension
  labels = np.argmax(probs, axis=last_axis)

  # Return as np.int32
  return np.asarray(labels, dtype=np.int32)


def noisy_max(prob, lap_scale):
  """
  This aggregation mechanism takes the softmax/logit output of several models
  resulting from inference on identical inputs and computes the noisy-max of
  the votes for candidate classes to select a label for each sample: it
  adds Laplacian noise to label counts and returns the most frequent label.
  :param prob: prob or probabilities for each sample
  :param lap_scale: scale of the Laplacian noise to be added to counts
  :return: pair of result and (if clean_votes is set to True) the clean counts
           for each class per sample and the the original labels produced by
           the teachers.
  """

  # Compute labels from prob/probs and reshape array properly
  labels = labels_from_probs(prob) # (N, 10) > (N, )

  labels_shape = np.shape(labels)
  labels = labels.reshape((labels_shape[0], labels_shape[1]))
  
  # Initialize array to hold final labels
  result = np.zeros(int(labels_shape[1])) # (N, 10)

  # Parse each sample
  for i in range(int(labels_shape[1])):
    # Count number of votes assigned to each class
    label_counts = np.bincount(labels[:, i], minlength=10)
    
    # Cast in float32 to prepare before addition of Laplacian noise
    label_counts = np.asarray(label_counts, dtype=np.float32)

    # Sample independent Laplacian noise for each class change the size of class in here 
    for item in range(1):
      label_counts[item] += np.random.laplace(loc=0.0, scale=float(lap_scale))

    # Result is the most frequent label
    result[i] = np.argmax(label_counts)

    # Cast labels to np.int32 for compatibility with deep_cnn.py feed dictionaries
    result = np.asarray(result, dtype=np.int32)
  
    # Only return labels resulting from noisy aggregation
  return result

def ensemble_preds(nb_teachers, stdnt_data, num_class):
    """
    Given a dataset, a number of teachers, and some input data, this helper
    function queries each teacher for predictions on the data and returns
    all predictions in a single array. 
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :param stdnt_data: unlabeled student training data
    :return: 3d array (teacher id, sample id, probability per class)
    """

    # Compute shape of array that will hold probabilities produced by each
    # teacher, for each training point, and each output class
    result_shape = (nb_teachers, len(stdnt_data), num_class)

    # Create array that will hold result
    result = np.zeros(result_shape, dtype=np.float32)

    # Get predictions from each teacher

    #save model to json and reload https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    for teacher_id in range(nb_teachers):
        # Compute path of weight file for teacher model with ID teacher_id
        filename2 = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.h5'
        model = tf.keras.models.load_model(filename2)
        # Get predictions on our training data and store in result array
        result[teacher_id] = model.predict(stdnt_data)  

        # This can take a while when there are a lot of teachers so output status
        print("Computed Teacher " + str(teacher_id) + "predictions")

    return result

def prepare_student_data(test_data,nb_teachers,lap_scale):
    """
    Takes a dataset name and the size of the teacher ensemble and prepares
    training data for the student model
    :param dataset: string corresponding to mnist, cifar10, or svhn
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :Param: lap_scale: scale of the Laplacian noise added for privacy
    :return: pairs of (data, labels) to be used for student training and testing
    """

    # Compute teacher predictions for student training data
    teachers_preds = ensemble_preds(nb_teachers, test_data,8)
    
    # Aggregate teacher predictions to get student training labels
    stdnt_labels = noisy_max(teachers_preds,lap_scale)
    print('stdnt_labels')
    stdnt_labels = utils.to_categorical(stdnt_labels,8)
    print(len(stdnt_labels))
    print(stdnt_labels.shape)   
    return stdnt_labels

def train_student(nb_teachers,lap):
    """
    This function trains a student using predictions made by an ensemble of
    teachers. The student and teacher models are trained using the same
    neural network architecture.
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :return: True if student training went well
    """
    # you need to change the address of get_dataset() manuly 
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_candlestick()
    
    # Call helper function to prepare student data using teacher predictions
    stdnt_labels= prepare_student_data(train_data, nb_teachers, lap)
    
    # labels acc 
    labels_result_cm = confusion_matrix(train_labels.argmax(axis=1), stdnt_labels.argmax(axis=1), labels=range(8))
    count = 0
    for r in range(8):
      count += labels_result_cm[r, r]
    label_acc = count/np.sum(labels_result_cm)
    print('labels accuracy:', label_acc)

    # Start student training
    model, hist = None, None
    model, hist = training(model, train_data, stdnt_labels, val_data, val_labels)
    
    test_pred = model.predict(test_data)
    test_pred = np.argmax(test_pred, axis=1)
    test_true = np.argmax(test_labels, axis=1)
    test_result_cm = confusion_matrix(test_true, test_pred, labels=range(8))
    print(test_result_cm)
    count = 0
    for r in range(8):
        count += test_result_cm[r, r]
    test_acc = count/np.sum(test_result_cm)
    print('testing accuracy:', count/np.sum(test_result_cm))
    
    # Compute final checkpoint name for student
    print(hist)
    model.save(f'student_teacher{nb_teachers}_lap{lap}.h5')

    return hist, label_acc, test_acc


if __name__ == "__main__":
    # Train teacher models first
    N = [1, 10, 20, 50]
    for n in N:
        for i in range(n):
            history = train_teacher(nb_teachers=n, teacher_id=i)
    # Next, train student models
    lap = [1, 10, 30, 50, 100]
    for l in lap:
        for t in N:
            history, lab_acc, t_acc = train_student(t,l)