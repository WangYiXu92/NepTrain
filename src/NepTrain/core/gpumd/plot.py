#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/29 19:56
# @Author  : 兵
# @email    : 1747193328@qq.com
import os.path


import numpy as np


from .utils import   get_dump_interval, calculate_angle, calculate_volume


def plot_md_selected(train_des,md_des,selected_des, save_path,decomposition="pca"):
    # 画一下图
    from matplotlib import pyplot as plt

    config = [
        # (文件名,图例,图例颜色)

    ]


    if md_des is not None:
        if isinstance(md_des,np.ndarray) and md_des.size!=0:
            config.append((md_des, 'new dataset', "#07cd66"))
        elif isinstance(md_des,list) :
            config.extend(md_des)
    if train_des is not None and train_des.size!=0:
        config.append((train_des, "base dataset","gray"))
    if selected_des is not None  and selected_des.size!=0:
        config.append((selected_des,'selected', "red"))

    fit_data = []

    for info in config:
        atoms_list_des= info[0]
        fit_data.append(atoms_list_des)
    if decomposition=="pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
    else:
        from umap import UMAP

        reducer = UMAP(n_components=2)

    reducer.fit(np.vstack(fit_data))
    fig = plt.figure()
    for index, array in enumerate(fit_data):
        proj = reducer.transform(array)
        plt.scatter(proj[:, 0], proj[:, 1], label=config[index][1], c=config[index][2])
    leg_cols=len(config)//3 or 1
    plt.legend(ncols=leg_cols)
    # plt.axis('off')
    plt.savefig(save_path)
    plt.close(fig)









def plot_md_thermo(thermo_path ):
    from matplotlib import pyplot as plt






    fig = plt.figure()
    data = np.loadtxt(thermo_path)

    dump_interval = get_dump_interval()
    time = np.arange(0, len(data) * dump_interval / 1000, dump_interval / 1000)

    # read data
    temperature = data[:, 0]
    kinetic_energy = data[:, 1]
    potential_energy = data[:, 2]
    pressure_x = data[:, 3]
    pressure_y = data[:, 4]
    pressure_z = data[:, 5]

    num_columns = data.shape[1]

    if num_columns == 12:
        box_length_x = data[:, 9]
        box_length_y = data[:, 10]
        box_length_z = data[:, 11]
        volume = box_length_x * box_length_y * box_length_z
    elif num_columns == 18:
        ax, ay, az = data[:, 9], data[:, 10], data[:, 11]
        bx, by, bz = data[:, 12], data[:, 13], data[:, 14]
        cx, cy, cz = data[:, 15], data[:, 16], data[:, 17]

        a_vectors = np.column_stack((ax, ay, az))
        b_vectors = np.column_stack((bx, by, bz))
        c_vectors = np.column_stack((cx, cy, cz))

        box_length_x = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        box_length_y = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        box_length_z = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)

        box_angle_alpha = calculate_angle(b_vectors, c_vectors)
        box_angle_beta = calculate_angle(c_vectors, a_vectors)
        box_angle_gamma = calculate_angle(a_vectors, b_vectors)

        volume = calculate_volume(a_vectors, b_vectors, c_vectors)
    else:
        raise ValueError("Unsupported number of columns in thermo.out. Expected 12 or 18.")



    # Plot the data

    # set the color of the plot
    color_red = '#d22027'  # red
    color_blue = '#015699'  # blue
    color_yellow = '#fac00f'  # yellow

    # Subplot
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=100)

    # Temperature
    axs[0, 0].plot(time, temperature, color=color_blue)
    axs[0, 0].set_xlabel('Time (ps)')
    axs[0, 0].set_ylabel('Temperature (K)')

    # Potential Energy and Kinetic Energy
    axs[0, 1].set_xlabel('Time (ps)')
    axs[0, 1].set_ylabel(r'Potential Energy ($x10^3$ eV)', color=color_red)
    axs[0, 1].plot(time, potential_energy / 1000, label='$P_E$', color=color_red)
    axs[0, 1].tick_params(axis='y', labelcolor=color_red, color=color_red)
    axs[0, 1].legend(frameon=False, bbox_to_anchor=(1.0, 0.4))

    axs_kinetic = axs[0, 1].twinx()
    axs_kinetic.set_ylabel('Kinetic Energy (eV)', color=color_blue)
    axs_kinetic.plot(time, kinetic_energy, label='$K_E$', color=color_blue)
    axs_kinetic.legend(frameon=False, bbox_to_anchor=(1.0, 0.3))
    axs_kinetic.tick_params(axis='y', labelcolor=color_blue, color=color_blue)

    # Pressure
    pressure_max = np.max(np.concatenate((pressure_x, pressure_y, pressure_z)))
    pressure_min = np.min(np.concatenate((pressure_x, pressure_y, pressure_z)))
    pressure_gap = pressure_max - pressure_min
    axs[0, 2].set_ylim(pressure_min - pressure_gap * 0.1, pressure_max + pressure_gap * 0.4)
    axs[0, 2].plot(time, pressure_x, label='Px', color=color_red)
    axs[0, 2].plot(time, pressure_y, label='Py', color=color_blue)
    axs[0, 2].plot(time, pressure_z, label='Pz', color=color_yellow)
    axs[0, 2].set_xlabel('Time (ps)')
    axs[0, 2].set_ylabel('Pressure (GPa)')
    axs[0, 2].legend(frameon=False, loc='upper right')

    # Lattice
    axs[1, 0].plot(time, box_length_x, label='a', color=color_red)
    axs[1, 0].plot(time, box_length_y, label='b', color=color_blue)
    axs[1, 0].plot(time, box_length_z, label='c', color=color_yellow)
    axs[1, 0].set_xlabel('Time (ps)')
    axs[1, 0].set_ylabel(r'Lattice Parameters (Å)')
    axs[1, 0].legend(frameon=False)

    # Volume
    axs[1, 1].plot(time, volume / 1000, label='Volume', color=color_blue)
    axs[1, 1].set_xlabel('Time (ps)')
    axs[1, 1].set_ylabel(r'Volume ($x10^3$ Å$^3$)')



    # Angles (only for triclinic systems)
    if num_columns == 18:
        axs[1, 2].plot(time, box_angle_alpha, label=r'$\alpha$', color=color_red)
        axs[1, 2].plot(time, box_angle_beta, label=r'$\beta$', color=color_blue)
        axs[1, 2].plot(time, box_angle_gamma, label=r'$\gamma$', color=color_yellow)
        axs[1, 2].set_xlabel('Time (ps)')
        axs[1, 2].set_ylabel(r'Angles ($\degree$)')
        axs[1, 2].legend(frameon=False)

    plt.tight_layout()

    plt.savefig(os.path.join(os.path.dirname(thermo_path),"thermo.png") , dpi=300)

    plt.close(fig)
