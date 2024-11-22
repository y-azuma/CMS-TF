import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
from pathlib import Path
from typing import List

def reset_list(num: int) -> List[int]:
    return [0 for _ in range(num)]

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int)
    #old_classes_gt = set(y_true)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # w: pred x label count
    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    #ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    return total_acc

def plot_scatter(tsne_feature: np.ndarray, label: int, epoch: int, name: str, save_path: Path) -> None:
    if not os.path.exists(save_path.parent):
        os.makedirs(save_path.parent)
    save_path.parent
    # train
    plt.figure(figsize=(10, 8))
    
    plt.scatter(tsne_feature[:, 0], tsne_feature[:, 1], c=label, cmap='tab10', marker='o', label=name)

    plt.title('t-SNE Train - Epoch {epoch}'.format(epoch=epoch))
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend()
    plt.colorbar()

    # 画像の保存
    plt.savefig(str(save_path))
    plt.close()