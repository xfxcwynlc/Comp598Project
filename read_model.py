import tensorflow as tf
# from model import poses_diff
def getModel():
    DD_Net = tf.keras.models.load_model('DD_Net.h5')
    return DD_Net