from abc import ABCMeta, abstractmethod
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
import time
from collections import OrderedDict
from scipy.cluster.hierarchy import linkage, fcluster

from cms.modules.parameter_manager import TrainParam,LossParam
from cms.modules.logger_manager import Logger
from cms.models.optimizer import get_optimizer
from cms.models.loss import SupervisedContrastiveLoss, MeanShiftContrastiveLoss
from cms.modules.metrics import compute_accuracy,reset_list, plot_scatter

class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, model: tf.keras.model, config: TrainParam, batch_size: int, logger: Logger, train_metrics: list):
        
        self.model = model
        self.config = config
        self.optimizer = self._set_optimizer()
        self.batch_size = batch_size
        self.logger = logger
        
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
        
    def _set_metrics(self, metrics: list, dst: str = "train_metrics"):
        MeanMetric = tf.keras.metrics.Mean
        _metrics = [[met, MeanMetric(name=met)] if isinstance(met, str) else [met.name, met] for met in metrics]
        setattr(self, dst, dict(_metrics))
        
    def _set_writer(self):
        self.writer = tf.summary.create_file_writer(str(Path(self.config.save_directory).joinpath("log")))
    
    def _save_weight(self):
        self.model.save_weights(str(Path(self.config.save_directory).joinpath("cms_model.h5")))
        
    def _log_train_metrics(self, metrics: list):
        with self.writer.as_default():
            for name, metric in metrics.items():
                tf.summary.scalar(name, metric.result(), step=self.iteration)
                metric.reset_states()
                
    @abstractmethod
    def run(self, dataset: tf.data.Dataset, iteration: int):
        pass
        
    @property
    def iteration(self):
        return self.optimizer.iterations.numpy()
        
    @property
    def lr(self):
        return self.optimizer.learning_rate.numpy()
    
    def _calculate_mean_shifted_value(self, feature: tf.Tensor, memory_feature: tf.Tensor, k_cluster: int = 9):
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
    
    def _generate_memory_tensor(self, dataset: tf.data.Dataset, max_iteration: int = 0):
        if max_iteration == 0:
            max_iteration = tf.data.experimental.cardinality(dataset).numpy()
        return self._concat_tensors(self._run_encode_step(dataset=dataset,training=False),max_iteration=max_iteration)
    
    def _run_encode_step(self, dataset: tf.data.Dataset, training: bool = False):
        for tensors in dataset:
            yield self._encode_step(tensors, training=training)
    
    @tf.function()
    def _encode_step(self, tensor: tf.Tensor, training: bool):
        x1, x2, y = tensor
        feature1 = self.model(x1, training=training)
        feature2 = self.model(x2, training=training)
        feature1 = tf.math.l2_normalize(feature1, axis=1)
        feature2 = tf.math.l2_normalize(feature2, axis=1)
        
        return feature1, feature2, y
    
    def _concat_tensors(self, datasets: tf.data.dataset, max_iteration: int):
        tensors_list = []
        with tqdm(datasets,total=max_iteration,leave=False,ncols=100) as pbar:
            for i, tensors in enumerate(pbar):
                pbar.set_description("Creating tensor")
                time.sleep(0.01)
                
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
    
class MeanShiftContrastiveTrainer(BaseTrainer):
    def _set_loss_condition(self, loss_config: LossParam):
        self.supervised_loss_func  = SupervisedContrastiveLoss(loss_config)
        self.supervised_loss_func.set_contrastive_loss_condition()
        self.unsupervised_loss_func = MeanShiftContrastiveLoss(loss_config)
        self.unsupervised_loss_func.set_contrastive_loss_condition()
        
        
    def run(self, dataset: tf.data.Dataset,memory_dataset: tf.data.Dataset, iteration: int, epoch_iteration: int):
        epoch = 0
        
        with tqdm(dataset,total=iteration, ncols=150) as pbar:
            for tensors in pbar:
                pbar.set_description("[Epoch {ep}, iteration {ite}]".format(ep=epoch,ite=self.iteration))
                pbar.set_postfix(OrderedDict(loss=list(self.train_loss.values())[0].result().numpy()))
                time.sleep(0.01)
                
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
                memory_feature = self._generate_memory_tensor(memory_dataset)[0]
                
                x1, x2, y = tensors
                y_flattened = tf.reshape(y, [-1])
                labeled_indices = tf.reshape(tf.where(tf.not_equal(y_flattened, -1)), [-1])
                if labeled_indices.shape[0] == 0:
                    self.logger.log("debug", "no labeled data")
                    self._unsupervised_train_step(tensors,memory_feature)
        
                else:
                    x1_labeled, x2_labeled, y_labeled = tf.gather(x1, labeled_indices), tf.gather(x2, labeled_indices), tf.gather(y, labeled_indices)
                    labeled_tensors = (x1_labeled, x2_labeled, y_labeled)
                    self._semi_supervised_train_step(tensors, memory_feature, labeled_tensors)


    @tf.function()
    def _unsupervised_train_step(self, tensors: tf.Tensor, memory_feature: tf.Tensor):
        with tf.GradientTape() as tape:
            x1, x2, _ = tensors
            feature1 = self.model(x1)
            feature2 = self.model(x2)
            feature1 = tf.math.l2_normalize(feature1, axis=1)
            feature2 = tf.math.l2_normalize(feature2, axis=1)
            knn_emb = self._calculate_mean_shifted_value(feature1, memory_feature)
            total_loss = self.unsupervised_loss_func.total_loss((feature1,feature2),self.batch_size,knn_emb)
        
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        self.train_loss["train/loss"](total_loss)
        
    @tf.function()
    def _semi_supervised_train_step(self, tensors: tf.Tensor, memory_feature: tf.Tensor, labeled_tensors: tf.Tensor):
        with tf.GradientTape() as tape:
            x1, x2, _ = tensors
            feature1 = self.model(x1)
            feature2 = self.model(x2)
            feature1 = tf.math.l2_normalize(feature1, axis=1)
            feature2 = tf.math.l2_normalize(feature2, axis=1)
            knn_emb = self._calculate_mean_shifted_value(feature1, memory_feature)
            unsupervised_loss = self.unsupervised_loss_func.total_loss((feature1,feature2),self.batch_size,knn_emb)
            supervised_loss = self.supervised_loss_func.total_loss(model=self.model,tensors =labeled_tensors)
            total_loss = (1-self.config.supervised_loss_weight)*unsupervised_loss + self.config.supervised_loss_weight*supervised_loss
        
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        self.train_loss["train/loss"](total_loss)

class MeanShiftContrastiveEvaluator(BaseTrainer):
    def _set_loss_condition(self, loss_config: LossParam):
        self.loss_func = MeanShiftContrastiveLoss(loss_config)
        self.loss_func.set_contrastive_loss_condition()
        
    def _log_test_metrics(self, metrics: list, epoch: int):
        with self.writer.as_default():
            for name, metric in metrics.items():
                tf.summary.scalar(name, metric.result(), step=epoch)
                metric.reset_states()
    
    def run(self, memory_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, num_clusters: int):
        epoch = 0
        continue_clustering = True

        while continue_clustering:
            # memory 
            if epoch == 0:
                memory_tensors = self._generate_memory_tensor(memory_dataset)
                memory_feature, memory_label = memory_tensors[0], memory_tensors[2]
                
                test_tensors = self._generate_memory_tensor(test_dataset)
                test_feature, test_label = test_tensors[0], test_tensors[2]
                
                split_indices = memory_feature.shape[0]
                features = np.concatenate([memory_feature, test_feature], 0)
                
                self._visualize_tsne(features, memory_label, test_label, split_indices, epoch="default")
                
                mean_shifted_features = self._mean_shifted_step(features)
                
                max_acc = reset_list(len(num_clusters))
                acc = reset_list(len(num_clusters))
                tolerance = reset_list(len(num_clusters))
                final_acc = reset_list(len(num_clusters))
            else:
                mean_shifted_features = self._mean_shifted_step(mean_shifted_features)
            
            self._visualize_tsne(mean_shifted_features, memory_label, test_label, split_indices, epoch)
                
            # clustering
            linked = linkage(mean_shifted_features, method="ward")
            for i in range(len(num_clusters)):
                if num_clusters[i] == 0:
                    continue
                
                if num_clusters[i]:
                    self.logger.log("info","num clusters{num_cluster}".format(num_cluster = num_clusters[i]))
                else:
                    continue
                threshold = linked[:, 2][-num_clusters[i]]
                predict_label = fcluster(linked, t=threshold, criterion='distance')
                acc_train = compute_accuracy(memory_label, predict_label[:split_indices])
                
                tolerance[i] = 0 if max_acc[i] < acc_train else tolerance[i] + 1
                
                if max_acc[i] < acc_train:
                    max_acc[i] = acc_train
                    acc[i] = compute_accuracy(test_label, predict_label[split_indices:])
                self.logger.log("info", "[Epoch {epo}]  train accuracy: {acc_train}".format(epo=epoch, acc_train=max_acc))
                self.logger.log("info", "[Epoch {epo}] test accuracy: {acc_test}".format(epo=epoch, acc_test=acc))
                self.test_accuracy1["test/accuracy1"](acc[0])
                self.test_accuracy2["test/accuracy2"](acc[1])
                self._log_test_metrics(self.test_accuracy1,epoch)
                self._log_test_metrics(self.test_accuracy2,epoch)

                    
                if tolerance[i] >= 2:
                    num_clusters[i] = 0
                    final_acc[i] = acc[i]
                    
            if sum(num_clusters) == 0:
                continue_clustering=False
                self.logger.log("info", "final epoch: {epo}".format(epo=epoch))
                self.logger.log("info", "final train accuracy: {acc_train}".format(acc_train=max_acc))
                self.logger.log("info", "final test accuracy: {acc_test}".format(acc_test=acc))
                
            epoch += 1
    
    def _mean_shifted_step(self, mean_shifted_features: tf.Tensor):
        knn_emb = self._calculate_mean_shifted_value(mean_shifted_features, mean_shifted_features)
        shifted_batch = self.loss_func._mean_shift(mean_shifted_features, knn_emb)
        output_batch  = tf.math.l2_normalize(shifted_batch, axis=1)
        
        return output_batch
    
    def _visualize_tsne(self, features: tf.Tensor, memory_label: tf.Tensor, test_label: tf.Tensor, split_indices: int, epoch: int):
        # t-SNE
        tsne = TSNE(n_components=2, random_state=123)
        tsne_results = tsne.fit_transform(features)

        memory_tsne = tsne_results[:split_indices]
        test_tsne = tsne_results[split_indices:]
        
        # train
        plot_scatter(memory_tsne,memory_label,epoch=epoch, name="Train",save_path=Path(self.config.save_directory).joinpath('tsne/tsne_train_epoch_{epoch}.png'.format(epoch=epoch)))
        
        # test
        plot_scatter(test_tsne,test_label,epoch=epoch, name="Test",save_path=Path(self.config.save_directory).joinpath('tsne/tsne_test_epoch_{epoch}.png'.format(epoch=epoch)))
        