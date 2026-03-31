#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 21:38
# @Author  : 兵
# @email    : 1747193328@qq.com
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/21 16:40
# @Author  : 兵
# @email    : 1747193328@qq.com

import glob
import os

import matplotlib

from NepTrain import utils


matplotlib.use("Agg")

import numpy as np







Config = [
    {"name": "energy", "unit": "eV/atom"},
    {"name": "force", "unit": "eV/A"},
    {"name": "virial", "unit": "eV/atom"},
    {"name": "stress", "unit": "GPa"},
]


def plot_loss_result(axes ):
    loss = np.loadtxt("loss.out")
    axes.loglog(loss[:, 1:7],
                label=['Total', 'L1-regularization',
                       'L2-regularization', 'Energy-train',
                       'Force-train', 'Virial-train'])
    axes.set_xlabel('Generation/100')
    axes.set_ylabel('Loss')
    if np.any(loss[7:10] != 0):
        axes.loglog(loss[:, 7:10], label=['Energy-test', 'Force-test', 'Virial-test'])
    axes.legend(ncol=2, frameon=False)


def plot_train_result(axes , config: dict):
    from sklearn.metrics import r2_score, mean_squared_error

    types = ["train", "test"]
    colors = ['deepskyblue', 'orange']
    xys = [(0.1, 0.7), (0.4, 0.1)]
    for i in range(2):
        data_type = types[i]
        color = colors[i]
        xy = xys[i]
        if not os.path.exists(f"{config['name']}_{data_type}.out"):
            continue
        data = np.loadtxt(f"{config['name']}_{data_type}.out")
        min_value = np.min(data)
        max_value = np.max(data)
        index = data.shape[1] // 2
        axes.plot(data[:, index:], data[:, :index], '.', color=color, label=data_type)
        axes.plot(np.linspace(min_value, max_value, num=10), np.linspace(min_value, max_value, num=10), '-', color="k")

        if mean_squared_error is not None:
            rmse = np.sqrt(mean_squared_error(data[:, :index], data[:, index:]))
            r2 = r2_score(data[:, :index], data[:, index:])
            axes.text(xy[0], xy[1],
                      f'{data_type} RMSE={1000 * rmse:.3f}({"m" + config["unit"] if config["name"] != "stress" else "MPa"} )\n{data_type} $R^2$={r2:.3f}',
                      transform=axes.transAxes, fontsize=13)
        else:
            axes.text(xy[0], xy[1],
                      f'{data_type}',
                      transform=axes.transAxes, fontsize=13)
    handles, labels = axes.get_legend_handles_labels()
    label_dict = dict(zip(labels, handles))
    axes.legend(label_dict.values(), label_dict, frameon=False, ncol=2, columnspacing=1)
    axes.set_xlabel(f'DFT {config["name"]} ({config["unit"]})')
    axes.set_ylabel(f'NEP {config["name"]} ({config["unit"]})')


def _plot_nep_result( ):
    import matplotlib.pyplot as plt

    out_num = len(glob.glob("*_train.out"))
    if os.path.exists("loss.out"):
        out_num+=1


    rows = 2 if out_num >= 4 else 1
    cols = (out_num  ) // rows + (out_num  ) % rows


    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    grids = fig.add_gridspec(rows, cols)

    if os.path.exists("loss.out"):
        axes_index = 0
        axes = fig.add_subplot(grids[axes_index])
        axes_index += 1

        plot_loss_result(axes)
    else:
        axes_index = 0

    for config in Config:
        if not os.path.exists(f"{config['name']}_train.out"):
            continue
        axes = fig.add_subplot(grids[axes_index])
        plot_train_result(axes, config)
        axes_index += 1

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    plt.savefig("nep_result.png", dpi=300)


def plot_nep_result(directory):
    with utils.cd(directory):
        _plot_nep_result()



if __name__ == '__main__':
    plot_nep_result("./")