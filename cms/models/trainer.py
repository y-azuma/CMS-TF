from abc import ABCMeta, abstractmethod
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
import time
from collections import OrderedDict
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Tuple, List, Union

from cms.modules.parameter_manager import TrainParam,LossParam
from cms.modules.logger_manager import Logger
from cms.models.optimizer import get_optimizer
from cms.models.loss import SupervisedContrastiveLoss, MeanShiftContrastiveLoss
from cms.modules.metrics import compute_accuracy,reset_list, plot_scatter

class BaseTrainer(metaclass=ABCMeta):
    """
    Base class for train and evaluate functions used in contrastive learning scenarios
    """
    
    def __init__(self, model: tf.keras.Model, config: TrainParam, batch_size: int, logger: Logger, train_metrics: List[str]):
        """
        Args:
            model (tf.keras.Model): Model to be trained
            config (TrainParam): Configuration parameters for training and evaluating
            batch_size (int): The size of the batch
            logger (Logger): Custom logger class
            train_metrics (List[str]): List of metrics to track during training and evaluating
        """
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
    
    def _set_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        Sets the optimizer based on the configuration

        Returns:
            tf.keras.optimizers.Optimizer: The configured optimizer
        """
        lr_schedule=tf.keras.optimizers.schedules.ExponentialDecay(self.config.lr, self.config.decay_step, self.config.decay_rate)
        return get_optimizer(optimizer_name=self.config.optimizer, lr=lr_schedule, momentum=self.config.momentum)
        
    def _set_metrics(self, metrics: List[str], dst: str = "train_metrics") -> None:
        """
        Sets the training metrics.

        Args:
            metrics (List[str]): List of metrics to set.
            dst (str, optional): Attribute name to store the metrics. Defaults to "train_metrics".
        """
        MeanMetric = tf.keras.metrics.Mean
        _metrics = [[met, MeanMetric(name=met)] if isinstance(met, str) else [met.name, met] for met in metrics]
        setattr(self, dst, dict(_metrics))
        
    def _set_writer(self) -> None:
        """
        Initializes the TensorBoard writer for logging.
        """
        self.writer = tf.summary.create_file_writer(str(Path(self.config.save_directory).joinpath("log")))
    
    def _save_weight(self) -> None:
        """
        Saves the model weights to a specified directory based on self.config.save_directory
        """
        self.model.save_weights(str(Path(self.config.save_directory).joinpath("cms_model.h5")))
        
    def _log_train_metrics(self, metrics: List[str]) -> None:
        """
        Logs training metrics to TensorBoard.

        Args:
            metrics (List[str]): List of metric names to log.
        """
        with self.writer.as_default():
            for name, metric in metrics.items():
                tf.summary.scalar(name, metric.result(), step=self.iteration)
                metric.reset_states()
                
    @abstractmethod
    def run(self, dataset: tf.data.Dataset, iteration: int):
        pass
        
    @property
    def iteration(self) -> int:
        """
        Returns:
            int: Current iteration number
        """
        return self.optimizer.iterations.numpy()
        
    @property
    def lr(self) -> float:
        """
        Returns:
            float: Current learning rate value
        """
        return self.optimizer.learning_rate.numpy()
    
    def _calculate_mean_shifted_value(self, feature: tf.Tensor, memory_feature: tf.Tensor, k_cluster: int = 9) -> tf.Tensor:
        """
        Calculates mean shifted values based on features and memory features.

        Args:
            feature (tf.Tensor): Features per batch
            memory_feature (tf.Tensor): All data features
            k_cluster (int, optional): Number of clusters to consider. Defaults to 9.

        Returns:
            tf.Tensor: Mean shifted values tensor.
        """
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
    
    def _generate_memory_tensor(self, dataset: tf.data.Dataset, max_iteration: int = 0) -> List[np.ndarray]:
        """
        Generates memory features and labels from the given dataset

        Args:
            dataset (tf.data.Dataset): Input dataset to process
            max_iteration (int, optional): Maximum number of iterations. If 0, uses the full dataset. Defaults to 0

        Returns:
            List[np.ndarray]: List of concatenated tensors representing the memory
        """
        if max_iteration == 0:
            max_iteration = tf.data.experimental.cardinality(dataset).numpy()
        return self._concat_tensors(self._run_encode_step(dataset=dataset,training=False),max_iteration=max_iteration)
    
    def _run_encode_step(self, dataset: tf.data.Dataset, training: bool = False) :
        """
        Args:
            dataset (tf.data.Dataset): Input dataset to process
            training (bool, optional): Whether to run in training mode. Defaults to False.
        """
        for tensors in dataset:
            yield self._encode_step(tensors, training=training)
    
    @tf.function()
    def _encode_step(self, tensor: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], training: bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Encodes input tensor using the model.

        Args:
            tensor (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Input tensor from dataset
            training (bool): Whether to run the model in training mode

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Normalized first augmentation set of features, second augmentation set of features, and labels.
        """
        x1, x2, y = tensor
        feature1 = self.model(x1, training=training)
        feature2 = self.model(x2, training=training)
        feature1 = tf.math.l2_normalize(feature1, axis=1)
        feature2 = tf.math.l2_normalize(feature2, axis=1)
        
        return feature1, feature2, y
    
    def _concat_tensors(self, datasets: tf.data.Dataset, max_iteration: int) -> List[np.ndarray]:
        """
        Concatenates tensors from the dataset up to max_iteration.

        Args:
            datasets (tf.data.Dataset): The input dataset to process.
            max_iteration (int): Maximum number of iterations to process.

        Returns:
            List[np.ndarray]: List of concatenated tensors.
        """
        
        # tensors (x1, x2, y) from dataset -> [x1_array, x2_array, y_array]
        tensors_list = []
        with tqdm(datasets, total=max_iteration,leave=False, ncols=100) as pbar:
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
    """
    Trainer class for Contrastive Mean-Shift learning, this is a learning method combining supervised and unsupervised approaches.
    """
    
    def _set_loss_condition(self, loss_config: LossParam):
        """
        Sets up the loss functions for supervised and unsupervised learning

        Args:
            loss_config (LossParam): Configuration parameters for the loss functions
        """
        self.supervised_loss_func  = SupervisedContrastiveLoss(loss_config)
        self.supervised_loss_func.set_contrastive_loss_condition()
        self.unsupervised_loss_func = MeanShiftContrastiveLoss(loss_config)
        self.unsupervised_loss_func.set_contrastive_loss_condition()
        
        
    def run(self, dataset: tf.data.Dataset, memory_dataset: tf.data.Dataset, iteration: int, epoch_iteration: int) -> None:
        """
        Runs the training process for the specified number of iterations.

        Args:
            dataset (tf.data.Dataset): Training dataset
            memory_dataset (tf.data.Dataset): Training dataset with no repeats and drop-reminder false
            iteration (int): Total number of training iterations.
            epoch_iteration (int): Number of iterations per epoch.
        """
        
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
                    self._unsupervised_train_step(tensors, memory_feature)
        
                else:
                    x1_labeled, x2_labeled, y_labeled = tf.gather(x1, labeled_indices), tf.gather(x2, labeled_indices), tf.gather(y, labeled_indices)
                    labeled_tensors = (x1_labeled, x2_labeled, y_labeled)
                    self._semi_supervised_train_step(tensors, memory_feature, labeled_tensors)


    @tf.function()
    def _unsupervised_train_step(self, tensors: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], memory_feature: Union[tf.Tensor, np.ndarray]) -> None:
        """
        Performs batch-wise unsupervised training step.

        Args:
            tensors (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Input tensors based on first augmentation set of features, second augmentation set of features, and labels
            memory_feature (Union[tf.Tensor, np.ndarray]): features based on memory datasets
        """
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
    def _semi_supervised_train_step(self, tensors: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], memory_feature: Union[tf.Tensor, np.ndarray], labeled_tensors: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> None:
        """
        Performs batch-wise semi-supervised training step, combining unsupervised and supervised losses.

        Args:
            tensors (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Input tensors based on first augmentation set of features, second augmentation set of features, and labels
            memory_feature (Union[tf.Tensor, np.ndarray]): features based on memory datasets
            labeled_tensors (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Labeled tensors for supervised learning.
        """
        with tf.GradientTape() as tape:
            x1, x2, _ = tensors
            feature1 = self.model(x1)
            feature2 = self.model(x2)
            feature1 = tf.math.l2_normalize(feature1, axis=1)
            feature2 = tf.math.l2_normalize(feature2, axis=1)
            knn_emb = self._calculate_mean_shifted_value(feature1, memory_feature)
            unsupervised_loss = self.unsupervised_loss_func.total_loss((feature1,feature2), self.batch_size, knn_emb)
            supervised_loss = self.supervised_loss_func.total_loss(model=self.model, tensors =labeled_tensors)
            total_loss = (1-self.config.supervised_loss_weight)*unsupervised_loss + self.config.supervised_loss_weight*supervised_loss
        
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        self.train_loss["train/loss"](total_loss)

class MeanShiftContrastiveEvaluator(BaseTrainer):
    """
    Evaluator class for Contrastive  Mean-Shift learning, implementing agglomerative clustering and evaluation methods.
    """
    
    def _set_loss_condition(self, loss_config: LossParam) -> None:
        """
        Sets up the loss function to use mean shift

        Args:
            loss_config (LossParam): Configuration parameters for the loss function
        """
        self
        self.loss_func = MeanShiftContrastiveLoss(loss_config)
        self.loss_func.set_contrastive_loss_condition()
        
    def _log_test_metrics(self, metrics: List[str], epoch: int) -> None:
        """
        Logs test metrics to TensorBoard.

        Args:
            metrics (List[str]): List of metrics to log
            epoch (int): Current epoch number
        """
        with self.writer.as_default():
            for name, metric in metrics.items():
                tf.summary.scalar(name, metric.result(), step=epoch)
                metric.reset_states()
    
    def run(self, memory_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, num_clusters: int) -> None:
        """
        Runs the evaluation process, including clustering and accuracy computation  as described in https://arxiv.org/pdf/2004.11362 (4.3. Estimating the number of clusters)
        
        Args:
            memory_dataset (tf.data.Dataset): train Dataset used for evaluation with no repeats and drop-reminder false
            test_dataset (tf.data.Dataset): test Dataset used for evaluation
            num_clusters (int): Number of clusters for evaluation
        """
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
                self._log_test_metrics(self.test_accuracy1, epoch)
                self._log_test_metrics(self.test_accuracy2, epoch)

                    
                if tolerance[i] >= 2:
                    num_clusters[i] = 0
                    final_acc[i] = acc[i]
                    
            if sum(num_clusters) == 0:
                continue_clustering=False
                self.logger.log("info", "final epoch: {epo}".format(epo=epoch))
                self.logger.log("info", "final train accuracy: {acc_train}".format(acc_train=max_acc))
                self.logger.log("info", "final test accuracy: {acc_test}".format(acc_test=acc))
                
            epoch += 1
    
    def _mean_shifted_step(self, mean_shifted_features: tf.Tensor) -> tf.Tensor:
        """
        Performs mean shift

        Args:
            mean_shifted_features (tf.Tensor): Input features

        Returns:
            tf.Tensor: Normalized output features after mean shift.
        """
        knn_emb = self._calculate_mean_shifted_value(mean_shifted_features, mean_shifted_features)
        shifted_feature = self.loss_func._mean_shift(mean_shifted_features, knn_emb)
        output_feature  = tf.math.l2_normalize(shifted_feature, axis=1)
        
        return output_feature
    
    def _visualize_tsne(self, features: Union[tf.Tensor, np.ndarray], memory_label: Union[tf.Tensor, np.ndarray], test_label: Union[tf.Tensor, np.ndarray], split_indices: int, epoch: int) -> None:
        """
        Visualizes the features using t-SNE and saves the plots.

        Args:
            features (Union[tf.Tensor, np.ndarray]): Features to visualize
            memory_label (Union[tf.Tensor, np.ndarray]): Labels for memory features
            test_label (Union[tf.Tensor, np.ndarray]): Labels for test features
            split_indices (int): Index to split memory and test features
            epoch (int): Current epoch number
        """
        # t-SNE
        tsne = TSNE(n_components=2, random_state=123)
        tsne_results = tsne.fit_transform(features)

        memory_tsne = tsne_results[:split_indices]
        test_tsne = tsne_results[split_indices:]
        
        # train
        plot_scatter(memory_tsne, memory_label, epoch=epoch, name="Train", save_path=Path(self.config.save_directory).joinpath('tsne/tsne_train_epoch_{epoch}.png'.format(epoch=epoch)))
        
        # test
        plot_scatter(test_tsne, test_label, epoch=epoch, name="Test", save_path=Path(self.config.save_directory).joinpath('tsne/tsne_test_epoch_{epoch}.png'.format(epoch=epoch)))
        