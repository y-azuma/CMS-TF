import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D

class Predictor(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super(Predictor, self).__init__()
        self.d0 = Dense(hidden_dim, activation="relu", use_bias=False)
        self.bn0 = BatchNormalization()
        self.d1 = Dense(output_dim)
        
    def call(self,x, training=True):
        x = self.d0(x)
        x = self.bn0(x, training=training)
        x = self.d1(x)
        return x
    
class ProjectionHead(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.gap = GlobalAveragePooling2D()
        self.d0 = Dense(hidden_dim, activation="relu", use_bias=False)
        self.bn0 = BatchNormalization()
        self.d1 = Dense(output_dim)
        self.bn1 = BatchNormalization()
        
    def call(self,x,training=True, is_gap=False):
        if is_gap:
            x = self.gap(x)
        x = self.d0(x)
        x = self.bn0(x, training=training)
        x = self.d1(x)
        x = self.bn1(x)
        return x
    
class CustomModel(tf.keras.Model):
    def __init__(self, encoder:tf.keras.Model, projection_head:tf.keras.Model, predictor:tf.keras.Model):
        super(CustomModel, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.predictor = predictor
        
    def call(self, x, training=True):
        x = self.encoder(x, training)
        x = self.projection_head(x, training)
        
        return x
    
    def prediction(self, x, training=True):
        x = self.encoder(x, training)
        x = self.projection_head(x, training)
        x = self.predictor(x, training)
        
        return x
    
class CMSModel(tf.keras.Model):
    def __init__(self, encoder:tf.keras.Model, projection_head:tf.keras.Model):
        super(CMSModel, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def call(self, x, training=True):
        x = self.encoder(x, training)
        x = self.projection_head(x, training)
        
        return x
        
        
        
    