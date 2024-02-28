import os
import glob
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import savgol_filter
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
import argparse
import json
import matplotlib
import scipy.interpolate as interp
from scipy.ndimage import uniform_filter1d
# from easydict import EasyDict as edict

from mem_ZrAu_withPar_test_ import Memristor
# from Visual.plot_line import plot_twinx
from Visual.visual_utils import set_ticks_label, set_legend
from Utils.mean_over_same_voltageDC import mean_over_same_v
from Utils.utils import utils


# Initial guess for the internal variable of the memristor
g0V4=.3
g0V2=.35
g0V1=.55
# g0_list = [.3, .35, .55]
g0_list = [g0V2, g0V1]
savgol_window = 10
take_only_ind = None

def plot_twinx(x_data, x_label, y_data, y_label, y2_data=[], y2_label=None, save_path=None, figname=None,
               y_scale=['lin', 'lin'], title=None, curve_labels_y=[''], curve_labels_y2=[''], show=False,
               ax=None, y1_ticks=None, y2_ticks=None, x_ticks=None,
               colors=['blue', 'red']):
    from Visual.visual_utils import align_yaxis, set_legend, set_ticks_label

    if type(y_data) is not list:
        y_data = [y_data]
    if type(y2_data) is not list:
        y2_data = [y2_data]

    if y1_ticks is None:
        y1_ticks = [np.min(y_data), 0, np.max(y_data)]
    else:
        pass

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))
    axes = [ax]
    # make a plot
    if y_scale[0] == 'log':
        ax.semilogy(x_data, y_data, color="red", marker="o", markersize=.5)
    else:
        for (y, cur_lab, c) in zip(y_data, curve_labels_y, colors):
            ax.plot(x_data, y, color=c, label=cur_lab, marker='o',
                linewidth=3, markersize=4)
    if x_ticks is None:
        set_ticks_label(ax=ax, ax_label=x_label, ax_type='x', valfmt="{x:.1f}", data=x_data, num=5)
    else:
        set_ticks_label(ax=ax, ax_label=x_label, ax_type='x', valfmt="{x:.1f}", data=x_data, ticks=x_ticks, num=5)
    if len(y2_data) != 0:
        if y2_ticks is None:
            y2_ticks = [np.min(y2_data), 0, np.max(y2_data)]
        else:
            pass
        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()
        axes.append(ax2)
        # make a plot with different y-axis using second axis object
        if y_scale[1] == 'log':
            ax2.semilogy(x_data, y2_data, color="green", marker="o", markersize=.5)
        else:
            for (y, cur_lab) in zip(y2_data, curve_labels_y2):
                ax2.plot(x_data, y, color="green",  marker='o',
                linewidth=3, markersize=4)
        set_ticks_label(ax=ax2, ax_label=y2_label, ax_type='y', data=y2_data, num=5,
                        ticks=y2_ticks, valfmt="{x:.1f}",
                        fontdict_label={'color':'green'})
    if title:
        plt.title(title)
    ax.grid()
    if len(y2_data) > 0:
        align_yaxis(ax, ax2)
    if len(y_data) > 1:
        set_legend(ax=ax)
        set_ticks_label(ax=ax, ax_label=y_label, ax_type='y', data=y_data, num=5,
                        ticks=y1_ticks, fontdict_label={'color': c})
    set_ticks_label(ax=ax, ax_label=y_label, ax_type='y', data=y_data, num=5,
                    ticks=y1_ticks, fontdict_label={'color': c})
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + figname)
    if show==True:
        plt.show()


# function for genetic algorithm to minimize (sum of squared error)
def sumOfSquaredError(parameterTuple):
    # warnings.filterwarnings("ignore")  # do not print warnings by genetic algorithm
    a = parameterTuple
    val = set_model(0, *a)
    return np.sum((I_to_fit - val) ** 2.0)


def generate_Initial_Parameters(bounds):
    parameterBounds = [[bounds[0][i], bounds[1][i]] for i in range(len(bounds[0]))]
    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x


def plot_(x_data, x_label, y_data, y_label, save_path=None, figname=None):
    fig, ax = plt.subplots()
    # make a plot
    ax.plot(x_data, y_data, color="red", marker="o", markersize=.5)
    ax.set_ylabel(y_label, color="red", fontsize=14)
    ax.set_xlabel(x_label, color="black", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + figname)
        plt.close()
    else:
        plt.show()

def load_data(dir_list, save_path):
    print('\nLoad data from folders:')
    for d in dir_list:
        print(d)

    I_list = list()
    V_list = list()
    time_list = list()
    sweep_vel_list = list()

    as_measured_I = list()
    as_measured_V = list()
    as_measured_time = list()

    for i, d in enumerate(dir_list[::-1]):
        file_name = join(d, "Data.txt")
        print('Load file:\n', file_name)
        data = np.loadtxt(file_name).transpose()
        time = data[0]  # secondi
        R = data[-1]
        V = data[1]
        I = V / R * 1e6

        if '1880' in file_name:
            outlier_value = .32
            # I = I[np.where((I > -.32) & (I < .32))]
            # V = V[np.where((I > -.32) & (I < .32))]
            # time = time[np.where((I > -.32) & (I < .32))]
            # pass
        if '1882' in file_name:
            outlier_value = .25
            # I = I[np.where((I > -.25) & (I < .25))]
            # V = V[np.where((I > -.25) & (I < .25))]
            # time = time[np.where((I > -.25) & (I < .25))]
        if '1878' in file_name:
            outlier_value = .25
            # I = I[np.where((I > -.25) & (I < .25))]
            # V = V[np.where((I > -.25) & (I < .25))]
            # time = time[np.where((I > -.25) & (I < .25))]

        index_to_remove = np.where((I > -outlier_value) & (I < outlier_value))[0]
        I = I[index_to_remove]
        V = V[index_to_remove]
        time = time[index_to_remove]
        time = time - time[0]
        #

        # Do interpolation to fixed length 15
        # Interpolation
        # t = np.linspace(0, time[-1], 159)
        # V = np.interp(t, time, V)
        # I = np.interp(t, time, I)
        # time = t

        # Data as measured
        as_measured_V.append(V)
        as_measured_time.append(time)
        as_measured_I.append(I)

        # Remove outliers (savgol) + take mean for the same voltage bias
        time, V, I = mean_over_same_v(time=time, V=V, I=I, take_only_ind=take_only_ind)
        I = savgol_filter(I, savgol_window, 3)

        # if '1878' in file_name:
        #     I[75] = .14
        #     I[76] = .14

        # window_size = 8
        # V = uniform_filter1d(V, size=window_size)
        # I = uniform_filter1d(I, size=window_size)
        # time = uniform_filter1d(time, size=window_size)

        sweep_vel = 2 * np.max(V) / (np.max(time) / 2)
        sweep_vel_list.append(sweep_vel)
        I_list.append(I)
        V_list.append(V)
        time_list.append(time)

    y_lim = (-.26, .26)
    plot_multiple_IV(V_list=as_measured_V, I_list=as_measured_I, time_list=as_measured_time,
                     save_path=save_path_fit,
                     # title='AsMeasured',
                     y_lim=y_lim,
                     color_list=color_list,
                     alpha_list=np.array(alpha_list)*.05,
                     name_fig='asMeasured',
                     ticks=[.21, .1, 0, -.1, -.21])
    plot_multiple_IV(V_list=V_list, I_list=I_list, time_list=time_list,
                     color_list=color_list,
                     y_lim=y_lim,
                     save_path=save_path_fit,
                     # title='AfterPreprocessing',
                     name_fig='AfterPreprocessing', ticks=[.21, .1, 0, -.1, -.21])

    # Save path
    utils.ensure_dir(save_path)
    print('Save path:\n{:s}\n'.format(save_path))
    np.save(save_path + '/I_list.npy', arr=I_list)
    np.save(save_path + '/V_list.npy', arr=V_list)
    np.save(save_path + '/time_list.npy', arr=time_list)
    np.save(save_path + '/sweep_vel_list.npy', arr=sweep_vel_list)

    return np.array(time_list), np.array(V_list), np.array(I_list)

def plot_multiple_IV(V_list, I_list, time_list, color_list=['orange', 'blue', 'green'][::-1], alpha_list=[1, .4, .4],
                     save_path='./', name_fig='', ticks=[.21, .1, 0, -.1, -.21], title=None,
                     y_lim=None):

    i_label = r'I [$\mathbf{\mu}$A]'
    fig, ax = plt.subplots()
    for i, V, I, time in zip(range(len(V_list)), V_list, I_list, time_list):
        sweep_vel = 2 * np.max(V) / (np.max(time) / 2)
        ax.plot(V, I, label='{:.1f} V/s'.format(sweep_vel), marker='o',  # linestyle='dashed',
                linewidth=2, markersize=4, color=color_list[i],
                alpha=alpha_list[i])
        name_fig = name_fig + '_{:.1f}Vel_'.format(sweep_vel)

    # alpha=.8)
    # ax.plot(V_to_plot, I_to_plot, markersize=50, label='{:.1f} V/s'.format(sweep_vel))
    # ax.set_ylabel(i_label, color="blue", fontsize=14)
    # ax.set_xlabel('V (V)', color="black", fontsize=14)
    set_ticks_label(ax=ax, ax_type='x', ax_label='V [V]', data=V, num=5, valfmt="{x:.1f}")
    set_ticks_label(ax=ax, ax_type='y', ax_label=i_label, data=I_list, num=5, ticks=ticks)
    ax.grid()
    if title:
        plt.title(title)
    set_legend(ax=ax, title='Sweep velocity')
    if y_lim:
        ax.set_ylim(y_lim)
    plt.tight_layout()
    plt.savefig('{:s}/IV_{:s}.svg'.format(save_path, name_fig))
    plt.show()

def plot_single_measurment(time_list, V_list, I_list,
                           save_path, name_fig='',
                           i_label=r'I [$\mathbf{\mu}$A]',
                           color_list=['red', 'blue']):

    for i, V, I, time in zip(range(len(V_list)), V_list, I_list, time_list):
        sweep_vel = 2 * np.max(V) / (np.max(time) / 2)
        fig, ax = plt.subplots()
        plot_twinx(y_data=I, y_label=i_label,
                   y2_data=V, y2_label='V [V]',
                   x_data=time, x_label="Time [s]",
                   y1_ticks=[.21, .1, 0, -.1, -.21], ax=ax,
                   title='{:.1f} V/s'.format(sweep_vel),
                   colors=[color_list[i]],
                   y2_ticks=[-20, -10, 0, 10, 20], y_scale=['linear', 'linear'],
                   save_path=save_path, figname=name_fig+'{:.1f}_meas_current.svg'.format(sweep_vel))
        plt.show()



    # plot_twinx(y_data=I, y_label=i_label, x_data=V, x_label="V [V]", save_path=save_path, title='Measured data',
    #            figname='meas_IV.png', y_scale=['linear'])


def set_model_multiple_sweep(i, *p):

    ptemp = np.array(p)

    # g0_list = [.55, .55, .6]
    # g0_list = [.4, .55, .55]
    # g0_list = [.1, .1, .55]

    vel_list = np.zeros(len(time_list))
    curr_list = np.zeros((len(time_list), len(time_list[0])))
    v_l = np.zeros((len(time_list), len(time_list[0])))

    for ind, V, I, time in zip(range(len(V_list)), V_list, I_list, time_list):
        vel_list[ind] = 2 * np.max(V) / (np.max(time) / 2)
        mem = Memristor(param=ptemp)
        mem.g = g0_list[ind]
        i_f = mem.run(v=V, time=time)
        curr_list[ind] = i_f

    return np.array(curr_list).reshape(-1)

def set_model_multiple_sweep_for_verification(p, save_path=None):

    ptemp = np.array(p)

    # g0_list = [.1, .1, .35]
    # g0_list = [.6, .55, .55]
    # g0_list = [p[-1], p[-1]]

    vel_list = np.zeros(len(time_list))
    curr_list = np.zeros((len(time_list), len(time_list[0])))
    v_l = np.zeros((len(time_list), len(time_list[0])))

    for ind, V, I, time in zip(range(len(V_list)), V_list, I_list, time_list):
        vel_list[ind] = 2 * np.max(V) / (np.max(time) / 2)
        mem = Memristor(param=ptemp)
        mem.g = g0_list[ind]
        i_f = mem.run(v=V, time=time)
        curr_list[ind] = i_f
        mem.plot_all(time=time, v=V, show=False)
        if save_path is not None:
            plt.savefig(save_path+'simulated_{:.1f}.svg'.format(vel_list[ind]))
            plt.show()
        mdic = {"time": time, "current": i_f, "voltage": V, "sweep_velocity": vel_list[ind], "units": 's_uA_V' }
        savemat(save_path+"matlab_sweep_{:.1f}.mat".format(vel_list[ind]), mdic)

    print(mem.par_state)
    print(mem.par_PF)
    print(mem.par_Schottky_OFF)

    with open(save_path+"par_state.json", "w") as outfile:
        json.dump(mem.par_state, outfile)
    with open(save_path+"par_PF.json", "w") as outfile:
        json.dump(mem.par_PF, outfile)
    with open(save_path+"par_Schottky_OFF.json", "w") as outfile:
        json.dump(mem.par_Schottky_OFF, outfile)

    return np.array(curr_list)

def take_firstquadrant_IV(time_list, V_list, I_list):
    new_I = list()
    new_time = list()
    new_V = list()

    for i, V in enumerate(V_list):
        index = V_list[i] >= 0
        new_time.append(time_list[i, index])
        new_V.append(V_list[i, index])
        new_I.append(I_list[i, index])
    return np.array(new_time), np.array(new_V), np.array(new_I)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_path', default='Output/AllMeasures', type=str)
    parser.add_argument('-diag', '--diagonals', default=0, type=int)
    parser.add_argument('-inp', '--input_type', default='I-V', type=str)
    parser.add_argument('-fit', '--do_fit', default=0, type=int)
    args = parser.parse_args()

    root = os.getcwd()
    save_path_ = join(root, '{:s}/{:s}/'.format(args.save_path, args.input_type))
    load_root = join(root + '/', 'Dati/{:s}/'.format(args.input_type))

    # load_root = join(root.rsplit('/', 1)[0]+'/', 'Dati/{:s}/'.format(args.input_type))
    # dir_list = sorted([d for d in glob.glob(load_root + "*/", recursive=True) if '.' not in d])
    # dir_list = sorted([d for d in glob.glob(load_root + "188*/", recursive=True) if '.' not in d])

    # dir_list = sorted([d for d in glob.glob(load_root + "188*/", recursive=True) if '.' not in d])[:2]
    # dir_list.append(glob.glob(load_root + "1878/", recursive=True)[0])

    # dir_list = list()
    # dir_list.append(glob.glob(load_root + "1882/", recursive=True)[0])
    # dir_list.append(glob.glob(load_root + "1878/", recursive=True)[0])
    # color_list = ['orange', 'blue', 'green', 'red']
    # alpha = [1, .4]

    dir_list = list()
    # dir_list.append(glob.glob(load_root + "1881/", recursive=True)[0])
    dir_list.append(glob.glob(load_root + "1878/", recursive=True)[0])
    dir_list.append(glob.glob(load_root + "1882/", recursive=True)[0])
    # dir_list.append(glob.glob(load_root + "1880/", recursive=True)[0])

    # color_list = ['orange', 'blue', 'green'][::-1]
    color_list = ['blue', 'orange']
    alpha_list = [1, .7, .7]

    save_path_ = join(root, '{:s}/{:s}/data_to_fit/'.format(args.save_path, args.input_type))

    save_path_fit = join(root,
                         '{:s}/{:s}/Fit2V_1V/FitMultipleSweep_g0={:.2f}_savgol{:d}_take_only_ind{:s}/'.format(args.save_path,
                                                                                                     args.input_type,
                                                                                                     g0_list[0],
                                                                                                     savgol_window,
                                                                                                     str(take_only_ind)))
    utils.ensure_dir(save_path_fit)

    time_list, V_list, I_list = load_data(dir_list=dir_list, save_path=save_path_)

    # Only V>0 I-V quadrant
    # time_list, V_list, I_list = take_firstquadrant_IV(time_list=time_list, V_list=V_list, I_list=I_list)


    plot_multiple_IV(V_list=V_list, I_list=I_list, time_list=time_list,
            save_path=save_path_, color_list=color_list, alpha_list=alpha_list)
    plot_single_measurment(V_list=V_list, I_list=I_list, time_list=time_list, save_path=save_path_,
                           color_list=color_list,
                           # alpha_list=alpha_list
                           )

    # # Param Schottky Off
    alpha_sc=1.9309504443539446e-06
    beta_sc=2.578400716579438

    # Param State
    lamb=0.0005466060834839386
    Vp=3.0907730590387272
    Vn=2.760988142749786

    # # Param Poole Frenkel
    A=0.0745656547187236
    eta=27.511761634849307
    beta=7.623614759607887
    s=0.1391070941337434
    q=703436.5293807065
    # Initial condition
    # g0 = 1

    # Fit

    if args.do_fit == 1:
        print('\nFitting...')
        p0 = [lamb, Vp, Vn,
              A, eta, beta, s, q,
              alpha_sc, beta_sc,
              ]
        bounds = ((
                   1e-20, 1e-20, 1e-20,
                   1e-20, 1e-20, 1e-20, 1e-20, 1,
                   1e-20, 1e-20,

                   ),
                  (
                   1e10, 18, 18,
                   1e10, 1e10, 1e10, 1, 1e10,
                   1e10, 1e10,

                  ))
        print(p0)
        print(bounds)

        best_val, cov = curve_fit(f=set_model_multiple_sweep,
                                  xdata=time_list.reshape(-1),
                                  ydata=I_list.reshape(-1),
                                  method='trf',
                                  p0=p0,
                                  bounds=bounds,
                                  # maxfev=maxfev, ftol=ftol
                                  )
        np.save(file='{:s}.npy'.format(save_path_fit + 'bounds'), arr=bounds)
        np.save(file='{:s}.npy'.format(save_path_fit + 'init_parameters'), arr=p0)
        np.save(file='{:s}.npy'.format(save_path_fit + 'best_param'), arr=best_val)
        np.save(file='{:s}.npy'.format(save_path_fit + 'g0'), arr=g0_list)
        print('\nFit concluded. Best param:\n')
        print(best_val)

    # Load param (and save in json to ease consultation)
    param = np.load('{:s}.npy'.format(save_path_fit + 'best_param'))
    print(param)
    with open(save_path_fit + "g0_list.json", "w") as outfile:
        json.dump({"g0V2": .35, "g0V1": .55}, outfile)

    # Plot output of the fit
    i_fit = set_model_multiple_sweep_for_verification(save_path=save_path_fit, p=param)

    plot_multiple_IV(V_list=V_list, I_list=i_fit, time_list=time_list,
            save_path=save_path_fit, color_list=color_list, alpha_list=alpha_list)
    plot_single_measurment(V_list=V_list, I_list=i_fit,
                           time_list=time_list,
                           save_path=save_path_fit,
                           name_fig='simulated_',
                           color_list=color_list,
                           # alpha_list=alpha_list
                           )

    # plot_multiple_IV(V_list=V_list, I_list=I_list, time_list=time_list,
    #                  save_path=save_path_fit,
    #                  name_fig='meas_',
    #                  color_list=color_list, alpha_list=alpha_list)
    # plot_single_measurment(V_list=V_list,
    #                        I_list=I_list,
    #                        time_list=time_list,
    #                        save_path=save_path_fit,
    #                        name_fig='meas_',
    #                        color_list=color_list,
    #                        # alpha_list=alpha_list
    #                        )

    a=0
