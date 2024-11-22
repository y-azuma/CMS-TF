from tensorflow.keras.optimizers import Adam, SGD, Optimizer

def get_optimizer(optimizer_name: str, lr: float, momentum: float)  -> Optimizer:
        
    if optimizer_name=="adam":
        optimizer = Adam(learning_rate=lr, beta_1=momentum)
        
    elif optimizer_name=="sgd":
        optimizer= SGD(learning_rate=lr, momentum=momentum)
        
    return optimizer