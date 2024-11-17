from dataclasses_json import dataclass_json
import dataclasses

@dataclass_json
@dataclasses.dataclass(frozen=True)
class LogParam:
    save_log_path: str
    save_log_level: str
    
@dataclass_json
@dataclasses.dataclass(frozen=True)
class DatasetParam:
    dataset: str
    cycle_length: int
    input_size: int
    input_channel:int
    batch_size: int
    unlabeled_ratio:float
    crop_pct: float
    num_clustering_train:int
    
@dataclass_json
@dataclasses.dataclass(frozen=True)
class ModelParam:
    model_name: str
    trainable_layer: int
    hidden_dimension: int
    output_dimension: int
    
    
@dataclass_json
@dataclasses.dataclass(frozen=True)
class LossParam:
    temperature: float
    base_temperature: float
    shift_coe:float
    
@dataclass_json
@dataclasses.dataclass(frozen=True)
class TrainParam:
    optimizer:str
    lr:float
    decay_step:int
    decay_rate:float
    momentum:float
    max_iteration:int
    num_clusters:list
    save_directory:str
    supervised_loss_weight:float