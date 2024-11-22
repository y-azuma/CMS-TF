from tensorflow.keras.optimizers import Adam, SGD, Optimizer

def get_optimizer(optimizer_name: str, lr: float, momentum: float)  -> Optimizer:
    """
    Get the specified optimizer

    Currently supports 'adam' and 'sgd' optimizers.

    Args:
        optimizer_name (str): Name of the optimizer
        lr (float): Learning rate for the optimizer
        momentum (float): Momentum parameter

    Returns:
        Optimizer: The configured optimizer instance.

    Raises:
        ValueError: If an unsupported optimizer name is provided.
    """
        
    if optimizer_name=="adam":
        optimizer = Adam(learning_rate=lr, beta_1=momentum)
        
    elif optimizer_name=="sgd":
        optimizer= SGD(learning_rate=lr, momentum=momentum)
        
    return optimizer