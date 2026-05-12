import matplotlib.pyplot as plt
import numpy as np
from typing import List
from pathlib import Path

path_save_figure = Path.cwd() / 'figure'

def create_figure_loss(history: List[int]) -> None:
    x_arr = np.arange(len(history)) + 1

    fig = plt.figure(figsize=(12, 8))
    plt.plot(x_arr, history, '--o', label='Train loss')
    plt.xlabel('Epoch', size=15)
    plt.ylabel('Loss', size=15)
    plt.legend(fontsize=15)

    plt.savefig(path_save_figure / 'train_loss.png')
    plt.close(fig)