from scipy.integrate import odeint
from scipy.optimize import curve_fit
# from easydict import EasyDict as edict
import numpy as np
from math import isnan
# from tqdm import tqdm
#import tensorflow as tf
from matplotlib import pyplot as plt
from scipy import signal
# from IPython.display import display, clear_output
import sys
import copy
from os.path import isfile, join, abspath
from os import pardir
import time
import argparse
from collections import namedtuple
from copy import deepcopy



class Memristor():
    def __init__(self, param, mode='single'):
        self.__info = 'ZrAu Memristor'
        if mode == 'network':
            self.update = self.update_netw
        elif mode == 'single':
            self.update = self.update_single
            self.set_param(parameters=param)
            self.g = self.g0

    def set_param(self, parameters):
        Params_Schottky = namedtuple('Params_Schottky', 'alpha beta')
        Params_PooleFrenkel = namedtuple('Params_PooleFrenkel', 'A eta beta s q')
        Params_StateVar = namedtuple('Params_StateVar', 'lamb chi Vp Vn')

        p = deepcopy(parameters)
        if len(p)==1:
            p = p[0]
        # self.par_Schottky_OFF = Params_Schottky(alpha=0.00289528733038056/100,
        #                                         beta=6.291096853931629/np.sqrt(10))

        self.par_Schottky_OFF = Params_Schottky(alpha=p[8],
                                                beta=p[9])

        # self.par_state = Params_StateVar(eta1=.1096480065414791,
        #                                  eta2=.1096480065414791,
        #                                  lamb=0.07796863403595362, chi=-1,
        #                                  Vp=12.74999687980728,
        #                                  Vn=12.74999687980728)
        self.par_state = Params_StateVar(chi=-1,
                                         lamb=p[0],
                                         Vp=p[1],
                                         Vn=p[2])
        # self.k = p[5]
        self.par_PF = Params_PooleFrenkel(A=p[3], eta=p[4], beta=p[5], s=p[6], q=p[7])
        # 2nd Pool

        # 1 Pool
        # self.par_PF = Params_PooleFrenkel(A=0.0745656547187236,
        #                     eta=27.511761634849307,
        #                     beta=7.623614759607887,
        #                     s=0.1391070941337434,
        #                     q=703436.5293807065)
        # self.par_PF = Params_PooleFrenkel(A=17523.740659366234/10/100,
        #                                   eta=17.290306027126782,
        #                                   beta=14.689289935558998/np.sqrt(10),
        #                                   s=0.18586204566077572,
        #                                   q=1.0037699659219494)

        # 2nd
        # self.par_state = Params_StateVar(eta1=0.1096480065414791, eta2=23.774396615707275, lamb=0.0005466060834839386, chi=-1,
        #                 Vp=3.0907730590387272, Vn=2.760988142749786)
        # self.par_PF = Params_PooleFrenkel(A=0.0745656547187236,
        #                                   eta=27.511761634849307,
        #                                   beta=7.623614759607887,
        #                                   s=0.1391070941337434,
        #                                   q=703436.5293807065)
        # self.par_Schottky_OFF = Params_Schottky(alpha=1.9309504443539446e-06, beta=2.578400716579438)

        self.g0 = .6

    def dgdt(self, t, g, v):
        # return self.par_state.chi * self.par_state.lamb * \
        #        (np.exp(self.par_state.eta1 * v) - np.exp( - self.par_state.Vp*v)) * self.auxiliary_f(var1=v, var2=g)
        if v==0:
            return 0
        else:
            return self.par_state.chi * self.par_state.lamb * (np.exp(v / self.par_state.Vp) - np.exp(-v/self.par_state.Vn))

        # if v > self.par_state.Vp:
        #     return self.par_state.chi * self.par_state.lamb * \
        #            np.sinh(v/self.par_state.Vp) * \
        #            self.auxiliary_f(var1=v, var2=g)
        #         # (np.exp(self.par_state.eta1 * v) - np.exp(self.par_state.eta1*self.par_state.Vp)) * \
        # elif v < -self.par_state.Vn:
        #         # print('oooo')
        #         return -self.par_state.chi * self.par_state.lamb * \
        #                (np.exp(self.par_state.eta2 * np.abs(v)) - np.exp(self.par_state.eta2 * self.par_state.Vn)) * \
        #                self.auxiliary_f(var1=v, var2=g)
        # else:
        #     return 0

    def diode_Schottky_g(self, v, par):
        if v==0:
            return 0
        else:
            return par.alpha * (np.exp(par.beta * np.sqrt(v)) - 1)

    def PooleFrenkel(self, v, par):
        '''
        PF expression for larger voltage ranges, saturation to Ohmic for V>>1.
        '''
        alpha = par.beta * np.sqrt(v)
        a = (1 + par.s*np.exp(par.eta - alpha)/par.q)
        b = (np.sqrt(1 + 4*par.s*(par.q-1)*np.exp(par.eta - alpha)/(par.q*np.power(1 + par.s/par.q*np.exp(par.eta - alpha), 2))) - 1)
        return par.A * (.5*np.exp(-par.eta + alpha) * a * b) * v

    def i_v_func_(self, v):
        if v >= 0:
            self.Ioff = (1 - self.g) * self.diode_Schottky_g(v=np.abs(v), par=self.par_Schottky_OFF)
            self.Ion = self.g * self.PooleFrenkel(v=np.abs(v), par=self.par_PF)
            # self.Ion = self.PooleFrenkel(v=np.abs(v), par=self.par_PF)
            return np.sign(v)*(self.Ion + self.Ioff)
        else:
            # self.Ioff = (1 - self.g) * self.diode_Schottky_g(v=np.abs(v), par=self.par_Schottky_OFF)
            # self.Ion = self.g * self.PooleFrenkel(v=np.abs(v), par=self.par_PF)
            # return np.sign(v) * (self.Ion + self.Ioff)

            self.Ioff = self.g * self.diode_Schottky_g(v=np.abs(v), par=self.par_Schottky_OFF)
            self.Ion = (1-self.g) * self.PooleFrenkel(v=np.abs(v), par=self.par_PF)
            return np.sign(v)*(self.Ion + self.Ioff)

    def update_single(self, v, delta_t):
        result = odeint(self.dgdt, y0=self.g,
                        t=delta_t,
                        args=(v,), tfirst=True)
        self.g = np.clip(result[1], 0, 1)[0]
        i = self.i_v_func_(v)
        return i

    def run(self, v, time):
        self.curr = np.zeros(len(time))

        self.g_array = np.zeros(len(time))

        self.curr[0] = self.i_v_func_(v[0])
        self.g_array[0] = self.g

        self.Ion_array = np.zeros(len(time))
        self.Ioff_array = np.zeros(len(time))

        for i in range(1, len(time)):
            self.curr[i] = self.update(v=v[i], delta_t=[time[i-1], time[i]])
            self.g_array[i] = self.g
            self.Ion_array[i] = self.Ion
            self.Ioff_array[i] = self.Ioff

        self.memristance = v / self.curr
        self.memristance[v==0] = 0
        return self.curr

    def plot_all(self, time, v, show=True, figsize=(12, 10), title=None,
                 current_label=r'I [$\mathbf{\mu}$A]', voltage_label='V [V]'):
        from Visual.plot_line import plot_twinx, plot_twinx_scatter
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        if title:
            plt.suptitle(title)
        plot_twinx(y_data=[self.curr],
                   curve_labels_y=['Model'],
                   y_label=current_label,
                   y2_data=v, y2_label=voltage_label,
                   x_data=time, x_label="Time [s]",
                   y1_ticks=[.21, .1, 0, -.1, -.21],
                   y2_ticks=[-20, -10, 0, 10, 20],
                   # title='Positive Triangular',
                   # save_path=save_path_sim,
                   # figname='model_par_article_IV_pos_tri.png')
                   ax=axes[0, 0])

        plot_twinx(y_data=self.curr,
                   curve_labels_y=['Fit', 'Measure'],
                   y_label=current_label,
                   x_data=v, x_label=voltage_label,
                   y1_ticks=[.21, .1, 0, -.1, -.21],
                   x_ticks=[-20, -10, 0, 10, 20],
                   # title='Positive Triangular',
                   # save_path=save_path_sim, figname='model_par_article_IV_pos_tri.png'
                   ax=axes[0, 1])

        plot_twinx(y_data=self.g_array, y_label='w',
                   x_data=time, x_label='Time [s]',
                   y1_ticks=[0, .5, 1],
                   # title='Positive Triangular',
                   # save_path=save_path_sim, figname='model_par_article_IV_pos_tri.png'
                   ax=axes[1, 0])

        plot_twinx(y_data=self.g_array, y_label='w',
                   x_data=v, x_label=voltage_label,
                   x_ticks=[-20, -10, 0, 10, 20],
                   y1_ticks=[0, .5, 1],
                   # title='Positive Triangular',
                   # save_path=save_path_sim, figname='model_par_article_IV_pos_tri.png'
                   ax=axes[1, 1])
        if show:
            plt.show()


    # def Child(self, v, par):
    #     return par.alpha * (v ** 2)

    # def reversed_Schottky(self, v, par):
    #     return par.alpha * (1 - np.exp(-self.g * par.beta * v))

    # def FN_tnn(self, v, par):
    #     return par.alpha * v**2 * np.exp(-par.beta/v)

    # def Undirected_Tunneling(self, v, par):
    ##     return par.alpha * np.power((v - par.xc), self.c) * np.exp(-par.beta * np.power(v, self.j))
        # if v == 0:
        #     return 0
        # else:
        #     return par.alpha * np.sinh(v/np.sqrt(par.beta_1*v)) * np.exp(-par.beta_2 / np.power(v, par.j))

    def auxiliary_f(self, var1, var2, wmax=1, wmin=0):
        '''

        Parameters
        ----------
        var1: Voltage applied between
        var2: state variable
        wmax:
        wmin:

        Returns
        0 if V>0 and g>wmax
        0 if V<0 and g<wmin
        1 in all other cases
        -------
        '''

        def sign2(var):
            '''
            Parameters
            ----------
            var

            Returns
            -------
            1 if var>0
            0 if var<0
            '''
            return (np.sign(var) + 1) / 2

        return sign2(-var1) * sign2(wmax - var2) + sign2(var1) * sign2(var2 - wmin)
        # return sign2(var1) * sign2(var2-wmin)


