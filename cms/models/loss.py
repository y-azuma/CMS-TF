import tensorflow as tf
from abc import ABCMeta, abstractmethod
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
import numpy as np

class BaseLoss(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config
        
    @abstractmethod
    def total_loss(self):
        pass
    
    def set_contrastive_loss_condition(self):
        self._scce = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.SUM)
        
    def get_negative_mask(self, batch_size):
        negative_mask = np.ones((batch_size, 2*batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i,i] = 0
            negative_mask[i, i+batch_size] = 0
        return tf.constant(negative_mask)
    
    def _calculate_positive_loss(self, batch1, batch2, batch_size):
        loss_positive = tf.matmul(tf.expand_dims(batch1, 1), tf.expand_dims(batch2,2)) #batch, 1, 1
        loss_positive = tf.reshape(loss_positive, (batch_size,1)) #batch, 1
        loss_positive /= self.config.temperature
        return loss_positive
    
    def _calculate_negative_loss(self,  positive, negatives, batch_size):
        neagtive_mask = self.get_negative_mask(batch_size)
        loss_negative = tf.tensordot(
            tf.expand_dims(positive,1),
            tf.expand_dims(tf.transpose(negatives),0),
            axes = 2,
        ) 
        loss_negative = tf.boolean_mask(loss_negative, neagtive_mask)
        loss_negative = tf.reshape(loss_negative, (batch_size, -1))
        loss_negative /= self.config.temperature
        
        return loss_negative 
    
    
class SupervisedContrastiveLoss(BaseLoss):
    """
    https://arxiv.org/pdf/2004.11362.pdf
    """
        
    def total_loss(self, model, tensors): 

        x1, x2, y =tensors #(sup_batch, feature) (sup_batch, 1)
        
        output_aug_batch1 = model(x1, training=True)
        output_aug_batch2 = model(x2, training=True) 
        output_aug_batch1 = tf.math.l2_normalize(output_aug_batch1, axis=1)
        output_aug_batch2 = tf.math.l2_normalize(output_aug_batch2, axis=1)
        
        output_aug_batch = tf.concat([tf.expand_dims(output_aug_batch1, 2), tf.expand_dims(output_aug_batch2, 2)],axis=2 )  #(sup_batch, feature, channel)
        mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32) #(sup_batch, sup_batch)
        contrast_count = tf.shape(output_aug_batch)[2] #channel
        supervised_batch = tf.shape(output_aug_batch)[0]#sup_batch
        contrast_feature = tf.concat(tf.unstack(output_aug_batch, axis=2), axis=0) #(sup_batch*channel, feature)

        anchor_feature = contrast_feature #(sup_batch*channel, feature)
        anchor_count = contrast_count #channel
        #(sup_batch*channel, sup_batch*channel)
        anchor_dot_contrast = tf.divide(tf.tensordot(tf.expand_dims(anchor_feature,1), tf.expand_dims(tf.transpose(contrast_feature),0),axes = 2),self.config.temperature)
        logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True) #(sup_batch*channel,1)
        logits = anchor_dot_contrast - logits_max
        mask = tf.tile(mask, [anchor_count, contrast_count]) #(sup_batch*channel, sup_batch*channel)
        indices = tf.stack([tf.range(supervised_batch*anchor_count),tf.range(supervised_batch*anchor_count)],axis=1) #(sup_batch*2, 2)
        updates = tf.zeros([supervised_batch*anchor_count]) #(sup_batch*2, )
        logits_mask = tf.tensor_scatter_nd_update(tf.ones_like(mask),indices,updates) 
        
        mask = mask * logits_mask #(sup_batch*channel, sup_batch*channel)
        exp_logits = tf.exp(logits)*logits_mask #(sup_batch*channel, sup_batch*channel)
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True)) #(sup_batch*channel,sup_batch*channel)
        mean_log_prob_pos = tf.reduce_sum(mask* log_prob, axis=1) / tf.reduce_sum(mask, axis=1) #(sup_batch*channel, )
        loss = -(self.config.temperature/ self.config.base_temperature) * mean_log_prob_pos #(sup_batch*channel, )
        total_loss = tf.reduce_mean(loss)
        
        return total_loss

class MeanShiftContrastiveLoss(BaseLoss):
        
    def _mean_shift(self, feature, con_knn_emb):
        mean_shift_feat = (1-self.config.shift_coe) * feature + self.config.shift_coe*con_knn_emb
        norm = tf.sqrt(tf.reduce_mean(tf.pow(mean_shift_feat, 2), axis=-1, keepdims=True))
        mean_shift_feat = mean_shift_feat/norm
        return mean_shift_feat
    
    def total_loss(self, features, batch_size, knn_emb):
        
        # Mean Shift
        shifted_batch1 = self._mean_shift(features[0], knn_emb)
        shifted_batch2 = self._mean_shift(features[1], knn_emb)
        
        output_batch1  = tf.math.l2_normalize(shifted_batch1, axis=1)
        output_batch2  = tf.math.l2_normalize(shifted_batch2, axis=1)
        
        
        """
        Unsupervised Contrastive Learning
        """
        loss_positive = self._calculate_positive_loss(output_batch1, output_batch2, batch_size)
        
        negatives = tf.concat([output_batch2, output_batch1], axis=0)
        total_loss = 0
        for positives in [output_batch1, output_batch2]:
            loss_negative = self._calculate_negative_loss(positives,negatives,batch_size)
            
            labels = tf.zeros(batch_size)
            
            logits = tf.concat([loss_positive, loss_negative], axis=1)
            total_loss += self._scce(y_pred=logits, y_true=labels)
        total_loss = total_loss/(2*batch_size)
        
        
        return total_loss
        
        