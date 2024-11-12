from abc import ABCMeta, abstractmethod
from pathlib import Path
import tensorflow as tf
import numpy as np
from cms.models.loss import SupervisedContrastiveLoss, MeanShiftContrastiveLoss
from tqdm import tqdm
from cms.models.optimizer import get_optimizer

class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, model, config, batch_size, train_metrics):
        
        self.model = model
        self.config = config
        self.optimizer = self._set_optimizer()
        self.batch_size = batch_size
        
        # set metrics
        if train_metrics is not None:
            for met in train_metrics:
                self._set_metrics(met, met[0].replace("/","_"))
                
        self._set_writer()
        self.log_iter = 1
        self.save_iter = 2
    
    def _set_optimizer(self):
        lr_schedule=tf.keras.optimizers.schedules.ExponentialDecay(self.config.lr, self.config.decay_step, self.config.decay_rate)
        return get_optimizer(optimizer_name=self.config.optimizer, lr=lr_schedule, momentum=self.config.momentum)
        
    def _set_metrics(self, metrics, dst="train_metrics"):
        MeanMetrc = tf.keras.metrics.Mean
        _metrics = [[met, MeanMetrc(name=met)] if isinstance(met, str) else [met.name, met] for met in metrics]
        setattr(self, dst, dict(_metrics))
        
    def _set_writer(self):
        self.writer = tf.summary.create_file_writer(str(Path(self.config.save_directory).joinpath("log")))
    
    def _save_weight(self):
        self.model.save_weights(str(Path(self.config.save_directory).joinpath("cms_model.h5")))
        
    def _log_train_metrics(self, metrics):
        with self.writer.as_default():
            for name, metric in metrics.items():
                tf.summary.scalar(name, metric.result(), step=self.iteration)
                metric.reset_states()
                
    @abstractmethod
    def run(self, dataset, iteration):
        pass
        
    @property
    def iteration(self):
        return self.optimizer.iterations.numpy()
        
    @property
    def lr(self):
        return self.optimizer.learning_rate.numpy()
    
class MeanShiftContrastiveTrainer(BaseTrainer):
    def _set_loss_condition(self, loss_config):
        self.supervised_loss_func  = SupervisedContrastiveLoss(loss_config)
        self.supervised_loss_func.set_contrastive_loss_condition()
        self.unsupervised_loss_func = MeanShiftContrastiveLoss(loss_config)
        self.unsupervised_loss_func.set_contrastive_loss_condition()
        
        
    def run(self, dataset,memory_dataset, iteration, epoch_iteration):
        epoch = 0
        
        for tensors in dataset:
            self.batch_size = tensors[0].shape[0]
            if self.iteration ==0:
                tf.profiler.experimental.start(str(Path(self.config.save_directory).joinpath("profile")))
            elif self.iteration == 5:
                tf.profiler.experimental.stop()
            
            if (self.iteration % self.log_iter) == 0:
                self._log_train_metrics(self.train_loss)
                
            if (self.iteration % self.save_iter) == 0:
                self._save_weight()
                
            if (self.iteration % epoch_iteration) == 0:
                epoch += 1
            
            if self.iteration >= iteration:
                break
            # memory 
            memory_feature = self.generate_memory_tensor(memory_dataset)[0]
            
            x1, x2, y = tensors
            y_flattened = tf.reshape(y, [-1])
            labeled_indices = tf.reshape(tf.where(tf.not_equal(y_flattened, -1)), [-1])
            if labeled_indices.shape[0] == 0:
                print("No labeled data")
                self._unsupervised_train_step(tensors,memory_feature)
    
            else:
                x1_labeled, x2_labeled, y_labeled = tf.gather(x1, labeled_indices), tf.gather(x2, labeled_indices), tf.gather(y, labeled_indices)
                labeled_tensors = (x1_labeled, x2_labeled, y_labeled)
                self._semi_supervised_train_step(tensors, memory_feature, labeled_tensors)
            #logger
    
    @tf.function()
    def _unsupervised_train_step(self, tensors, memory_feature):
        with tf.GradientTape() as tape:
            x1, x2, _ = tensors
            feature1 = self.model(x1)
            feature2 = self.model(x2)
            knn_emb = self._calculate_mean_shifted_value(feature1, memory_feature)
            total_loss = self.unsupervised_loss_func.total_loss((feature1,feature2),self.batch_size,knn_emb)
        
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        self.train_loss["train/loss"](total_loss)
        
    @tf.function()
    def _semi_supervised_train_step(self, tensors, memory_feature, labeled_tensors):
        with tf.GradientTape() as tape:
            x1, x2, _ = tensors
            feature1 = self.model(x1)
            feature2 = self.model(x2)
            knn_emb = self._calculate_mean_shifted_value(feature1, memory_feature)
            unsupervised_loss = self.unsupervised_loss_func.total_loss((feature1,feature2),self.batch_size,knn_emb)
            supervised_loss = self.supervised_loss_func.total_loss(model=self.model,tensors =labeled_tensors)
            total_loss = (1-self.config.supervised_loss_weight)*unsupervised_loss + self.config.supervised_loss_weight*supervised_loss
        
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        self.train_loss["train/loss"](total_loss)
        
    def _calculate_mean_shifted_value(self, feature, memory_feature, k_cluster=9):
        class_wise_sim = tf.einsum("b d, n d -> b n", feature, memory_feature)
        _, indices = tf.math.top_k(class_wise_sim, k=k_cluster, sorted=True)
        indices = indices[:,1:]
        
        knn_emb = []
        
        for i in range(indices.shape[0]):
            idx = indices[i]
            knn_feats = tf.reshape(tf.gather(memory_feature, idx), [k_cluster-1, 256])
            knn_emb.append(tf.reduce_mean(knn_feats, axis=0))
        knn_emb = tf.convert_to_tensor(knn_emb)
        
        return knn_emb
    
    def generate_memory_tensor(self, dataset, max_iteration=0):
        if max_iteration == 0:
            max_iteration = tf.data.experimental.cardinality(dataset).numpy()
        return self._concat_tensors(self._run_encode_step(dataset=dataset,training=False),max_iteration=max_iteration)
        
    def _run_encode_step(self, dataset, training=False):
        for tensors in dataset:
            yield self._encode_step(tensors, training=training)
    
    @tf.function()
    def _encode_step(self, tensor, training):
        x1, x2, y = tensor
        feature1 = self.model(x1, training=training)
        feature2 = self.model(x2, training=training)
        
        return feature1, feature2, y
    
    def _concat_tensors(self, datasets, max_iteration):
        tensors_list = []
        
        for i, tensors in tqdm(enumerate(datasets),total=max_iteration):
            if i==max_iteration:
                break
            temp_list = []
            for tensor in tensors:
                if isinstance(tensor, tuple):
                    temp_list += [t for t in tensor]
                else:
                    temp_list.append(tensor)
            tensors_list.append(temp_list)
        result = [list(x) for x in zip(*tensors_list)]
        
        concat_tensors = []
        for comp in result:
            concat_tensors.append(np.concatenate(comp, axis=0))
            return concat_tensors
        
        
            