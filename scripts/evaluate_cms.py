import argparse
from pathlib import Path
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"
import tensorflow as tf
print("Tensorflow Ver: ", tf.__version__)
from tfimm import create_model

from cms.modules.parameter_manager import LogParam,DatasetParam,ModelParam,LossParam,TrainParam
from cms.modules.logger_manager import Logger
from cms.dataloader.dataloader import CMSDataloader
from cms.models.model import ProjectionHead, CMSModel
from cms.models.trainer import MeanShiftContrastiveEvaluator



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--log_config", type=str, default="../config/log.json")
    parser.add_argument("--dataset_config", type=str, default="../config/dataset.json")
    parser.add_argument("--model_config", type=str, default="../config/model.json")
    parser.add_argument("--loss_config", type=str, default="../config/loss.json")
    parser.add_argument("--train_config", type=str, default="../config/train.json")
    
    return parser.parse_args()

def main(args):
    
    # ------------------
    # parameter
    # ------------------
    log_param = LogParam.from_json(Path(args.log_config).read_text())
    dataset_param = DatasetParam.from_json(Path(args.dataset_config).read_text())
    model_param = ModelParam.from_json(Path(args.model_config).read_text())
    loss_param = LossParam.from_json(Path(args.loss_config).read_text())
    train_param = TrainParam.from_json(Path(args.train_config).read_text())
    
    # ------------------
    # logger
    # ------------------
    logger = Logger(log_param)
    gpus = tf.config.list_physical_devices("GPU")
    logger.log("debug", gpus)
    
    # ------------------
    # dataset
    # ------------------
    dataloader = CMSDataloader(config=dataset_param,num_train=dataset_param.num_clustering_train,drop_label=False)
    memory_ds = dataloader.make_memory_dataset(is_train_augmentation=False) # drop_remainder=False, not repeated
    test_ds = dataloader.make_test_dataset(test_batch_size=dataset_param.batch_size)
    
    # ------------------
    # model
    # ------------------
    base_model = create_model(model_param.model_name,pretrained="timm", nb_classes=0, input_size=(dataset_param.input_size, dataset_param.input_size)) # nb_classes=0 mean base_model will have no classification
    projection_head = ProjectionHead(hidden_dim=model_param.hidden_dimension,output_dim=model_param.output_dimension)
    model = CMSModel(encoder=base_model, projection_head=projection_head)
    model.build(
        input_shape=(None, dataset_param.input_size, dataset_param.input_size, dataset_param.input_channel)
    )
    
    # set fine-tuning layer
    model.encoder.trainable=False
    for layer in model.encoder.layers[-model_param.trainable_layer:]:
        layer.trainable=True

    # load weights
    logger.log("debug","load weights {path}".format(path=str(Path(train_param.save_directory).joinpath("cms_model.h5"))))
    model.load_weights(str(Path(train_param.save_directory).joinpath("cms_model.h5")))
    
    # ------------------
    # train    
    # ------------------
    trainer = MeanShiftContrastiveEvaluator(model=model,config=train_param,batch_size=dataset_param.batch_size,logger=logger, train_metrics=[["test/accuracy1"],["test/accuracy2"]])
    trainer._set_loss_condition(loss_config=loss_param)
    trainer.run(memory_ds,test_ds,num_clusters=train_param.num_clusters)
    
    
if __name__ == "__main__":
    args =parse_args()
    main(args)