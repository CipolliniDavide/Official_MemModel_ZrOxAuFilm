#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:15:05 2021

@author: hp
"""
from Utils.utils import utils

import matplotlib.pyplot as plt    
import numpy as np
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform

class NorthPolarAxes(PolarAxes):
    '''
    A variant of PolarAxes where theta starts pointing north and goes
    clockwise.
    '''
    name = 'northpolar'

    class NorthPolarTransform(PolarAxes.PolarTransform):
        def transform(self, tr):
            xy   = np.zeros(tr.shape, np.float_)
            t    = tr[:, 0:1]
            r    = tr[:, 1:2]
            x    = xy[:, 0:1]
            y    = xy[:, 1:2]
            x[:] = r * np.sin(t)
            y[:] = r * np.cos(t)
            return xy

        transform_non_affine = transform

        def inverted(self):
            return NorthPolarAxes.InvertedNorthPolarTransform()

    class InvertedNorthPolarTransform(PolarAxes.InvertedPolarTransform):
        def transform(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:]
            r = np.sqrt(x*x + y*y)
            theta = np.arctan2(y, x)
            return np.concatenate((theta, r), 1)

        def inverted(self):
            return NorthPolarAxes.NorthPolarTransform()

        def _set_lim_and_transforms(self):
            PolarAxes._set_lim_and_transforms(self)
            self.transProjection = self.NorthPolarTransform()
            self.transData = (self.transScale + self.transProjection + (self.transProjectionAffine + self.transAxes))
            self._xaxis_transform = (self.transProjection + self.PolarAffine(IdentityTransform(), Bbox.unit()) + self.transAxes)
            self._xaxis_text1_transform = (self._theta_label1_position + self._xaxis_transform)
            self._yaxis_transform = (Affine2D().scale(np.pi * 2.0, 1.0) + self.transData)
            self._yaxis_text1_transform = (self._r_label1_position + Affine2D().scale(1.0 / 360.0, 1.0) + self._yaxis_transform)



class visualizer:
    def __init__(self, data):
        self.data= data
        
    def set_scale_bar(self, limits=None):
        range_scale= self.data.values.max() - self.data.values.min()
        center= (range_scale)/2
        new_max= center + range_scale/4
        new_min= center - range_scale/4
        return new_min, new_max
    
    def show(self, save_name=None, save_fold='./'):
        
        # Create figure and add axis
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)# Remove x and y ticks
        ax.xaxis.set_tick_params(size=0)
        ax.yaxis.set_tick_params(size=0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.axis('off') 
        # Show AFM image
        vmin=self.data.values.min(); vmax=self.data.values.max()
        img = ax.imshow(self.data.values, origin='lower', cmap='gray', 
                        extent=(0, self.data.scan_size_micron, 0, self.data.scan_size_micron), 
                        vmin=vmin, vmax=vmax)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        if save_name:
            utils.ensure_dir(save_fold)
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(save_fold+save_name+'.png')#, bbox_inches = extent)
        else: plt.show()
        
        
        
    def heatMap_withScale(self, save_name=None, save_fold='./', title='', scale_range= None):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        import numpy as np
        
        # Font parameters
        mpl.rcParams['font.family'] = 'Avenir'
        mpl.rcParams['font.size'] = 18# Edit axes parameters
        mpl.rcParams['axes.linewidth'] = 2# Tick properties
        mpl.rcParams['xtick.major.size'] = 10
        mpl.rcParams['xtick.major.width'] = 2
        mpl.rcParams['xtick.direction'] = 'out'
        mpl.rcParams['ytick.major.size'] = 10
        mpl.rcParams['ytick.major.width'] = 2
        mpl.rcParams['ytick.direction'] = 'out'
        
        # Create figure and add axis
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)# Remove x and y ticks
        ax.xaxis.set_tick_params(size=0)
        ax.yaxis.set_tick_params(size=0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Show AFM image
        if scale_range is not None:
            vmin, vmax= self.set_scale_bar(scale_range)
        else: vmin=self.data.values.min(); vmax=self.data.values.max()
        img = ax.imshow(self.data.values, origin='lower', cmap='gray',#cmap='YlGnBu_r', 
                        extent=(0, self.data.scan_size_micron, 0, self.data.scan_size_micron), 
                        vmin=vmin, vmax=vmax)
        
        # Create scale bar
        x_scale_start= self.data.scan_size_micron/5
        x_len= int(self.data.scan_size_micron/4)
        len_str= '%d' %x_len
        if x_len is 0:
            x_len= self.data.scan_size_micron/2
            len_str= '%.1f' %x_len
        ax.fill_between(x=[x_scale_start, x_scale_start+x_len], 
                        y1=[self.data.scan_size_micron/20, self.data.scan_size_micron/20], 
                        y2=[self.data.scan_size_micron/20 + self.data.scan_size_micron/100, self.data.scan_size_micron/20 + self.data.scan_size_micron/100], color='white')
        ax.text(x=x_scale_start+x_len/3, y=self.data.scan_size_micron/20 + self.data.scan_size_micron/50, 
                 s=r'%s $\mathregular{\mu}m$' %len_str, va='bottom', ha='center', color='white', size=20)        
        ax.set_xticks([]); ax.set_yticks([])
        
        # Create axis for colorbar
        cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='5%', pad=0.1)
        # Create colorbar
        cbar = fig.colorbar(mappable=img, cax=cbar_ax)
        # Edit colorbar ticks and labels
        cbar_ticks= np.linspace(vmin, vmax, 3)
        cbar.set_ticks(cbar_ticks)
        cbar_ticks_label= ['%.3f' %i for i in cbar_ticks]
        cbar_ticks_label[-1]= cbar_ticks_label[-1:][0] + ' ' + '%s' %self.data.unit 
        cbar.ax.set_yticklabels(cbar_ticks_label)
        #cbar.set_ticklabels()
        if save_name:
            utils.ensure_dir(save_fold)
            plt.savefig(save_fold+save_name+'_heatmap.png')
            plt.close()
        else: plt.show()
        
        
    def polar_plot(self, bins_number=100, mov_average=1, save_name=None, save_fold='./', title=''):
        '''
        source:
            https://stackoverflow.com/questions/2417794/how-to-make-the-angles-in-a-matplotlib-polar-plot-go-clockwise-with-0-at-the-to

        Parameters
        ----------
        bins_number : int, optional
            DESCRIPTION. The default is 100. Number of bins we divide the interval [0,360) 
        mov_average : int, optional
            DESCRIPTION. Size of moving avarage applied to the angles. The default is 1 (no moving avarage applied). 
        save_name : TYPE, optional
            DESCRIPTION. The default is None.
        save_fold : TYPE, optional
            DESCRIPTION. The default is './'.
        title : TYPE, optional
            DESCRIPTION. The default is ''.

        Returns
        -------
        None.

        '''
        from math import radians
        import matplotlib.pyplot as plt
        import numpy as np
        # Image
        fig = plt.figure(figsize=(14, 8))
        ax1 = plt.subplot((121))
        #ax2 = plt.subplot(122, projection='polar')
        #ax=[ax1,ax2]
        # afm
        vmin=self.data.values.min(); vmax=self.data.values.max()
        img = ax1.imshow(np.flipud(np.rot90(self.data.values,k=1, axes=(0,1))), origin='lower', cmap='YlGnBu_r', 
                        extent=(0, self.data.scan_size_micron, 0, self.data.scan_size_micron), 
                        vmin=vmin, vmax=vmax)
        
        # Create scale bar
        x_scale_start= self.data.scan_size_micron/5
        x_len= int(self.data.scan_size_micron/4)
        len_str= '%d' %x_len
        if x_len is 0:
            x_len= self.data.scan_size_micron/2
            len_str= '%.1f' %x_len
        ax1.fill_between(x=[x_scale_start, x_scale_start+x_len], 
                        y1=[self.data.scan_size_micron/20, self.data.scan_size_micron/20], 
                        y2=[self.data.scan_size_micron/20 + self.data.scan_size_micron/100, self.data.scan_size_micron/20 + self.data.scan_size_micron/100], color='white')
        ax1.text(x=x_scale_start+x_len/3, y=self.data.scan_size_micron/20 + self.data.scan_size_micron/50, 
                 s=r'%s $\mathregular{\mu}m$' %len_str, va='bottom', ha='center', color='white', size=20)        
        ax1.set_xticks([]); ax1.set_yticks([])
        
        # Polar Plot
        angle = np.linspace(0, 360, num=bins_number, dtype=float) * np.pi / 180.0
        data= utils.angles_with_sobel_filter(self.data.values) + 180
        data= np.convolve(sorted(list(data)), np.ones(mov_average)/mov_average, mode='valid')
        lux = [radians(a) for a in data]
        #h, b= np.histogram(lux, bins=angle)
        h, _, b= utils.empirical_pdf_and_cdf(lux, bins=angle)
        
        #plt.clf()
        sp = plt.subplot((122), projection='polar')
        sp.plot(angle[:-1], h, linewidth=6, alpha=.6)
        
        sp.set_theta_zero_location('N')
        sp.set_theta_direction(+1)
        
        font_size_xlabel=30
        font_size_ylabel=25
        yticks= np.linspace(h.min(), h.max(), 3)
        sp.set_yticks(yticks)
        sp.set_yticklabels(['%.3f'%s for s in yticks])
        sp.tick_params(axis='x', labelsize= font_size_xlabel)
        sp.tick_params(axis='y', labelsize= font_size_ylabel )
        
        plt.suptitle(title, fontweight='bold', fontsize=font_size_xlabel)
        if save_name:
            utils.ensure_dir(save_fold)
            plt.savefig(save_fold+save_name+'_polar_angle.png')
            plt.close()
        else: plt.show()
    
    
        
    '''    
    def polar_plot_angles_sobel(self, bins_number=100, save_name=None, save_fold='./', title=''):
        import numpy as np
        
        fig = plt.figure(figsize=(14, 8))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122, projection='polar')
        ax=[ax1,ax2]
        # afm
        vmin=self.data.values.min(); vmax=self.data.values.max()
        img = ax[0].imshow(self.data.values,  origin='lower', cmap='YlGnBu_r', 
                           extent=(0, self.data.scan_size_micron, 0, self.data.scan_size_micron), 
                           vmin=vmin, vmax=vmax)
        
        # Create scale bar
        x_scale_start= 3*self.data.scan_size_micron/5
        x_len= int(self.data.scan_size_micron/4)
        ax[0].fill_between(x=[x_scale_start, x_scale_start+x_len], 
                        y1=[0.3, 0.3], y2=[0.35, 0.35], color='white')
        ax[0].text(x=x_scale_start+x_len/3, y=0.37, s=r'%d $\mathregular{\mu}m$' %x_len, va='bottom', ha='center', color='white', size=20)        
        ax[0].set_xticks([]); ax[0].set_yticks([])
        
        # Polar plot
        angles= utils.angles_with_sobel_filter(self.data.values) 
        #N=5;a= np.convolve(angles, np.ones(N)/N, mode='valid')
        #bins_number = 80  # the [0, 360) interval will be subdivided into this
        # number of equal bins
        bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
        #print(bins)
        #hist, _, bins_edges= utils.empirical_pdf_and_cdf(angles, bins=bins)
        hist, _, _ = plt.hist(angles, bins_number)
        print(hist.shape)
        width = 2 * np.pi / bins_number
        bars = ax[1].bar(bins[:bins_number], hist, width=width, bottom=0.0)
        for bar in bars:
            bar.set_alpha(0.5)
        font_size_label=40
        fontsize_ticks= 30
        ax[1].tick_params(axis='x', labelsize= font_size_label)
        ax[1].tick_params(axis='y', labelsize= font_size_label )
        
        plt.suptitle(title, fontweight='bold', fontsize=font_size_label)
        if save_name:
            utils.ensure_dir(save_fold)
            plt.savefig(save_fold+save_name+'polar_hist.png')
        else: plt.show()
    '''    
    
        
    '''    
    def polar_plot_zero_est(self, bins_number=100, save_name=None, save_fold='./', title=''):
        from math import radians
        
        register_projection(NorthPolarAxes)
        
        angle = np.linspace(0, 360, num=bins_number, dtype=float) * np.pi / 180.0
        #arbitrary_data = (np.abs(np.sin(angle)) + 0.1 * 
        #    (np.random.random_sample(size=angle.shape) - 0.5))
        #arbitrary_data= utils.scale(utils.angles_with_sobel_filter(self.data.values), (0,360))
        temp= utils.angles_with_sobel_filter(self.data.values) + 180
        data= [radians(a) for a in temp]
        arbitrary_data, b= np.histogram(data, bins=angle)
        print(b)
        print(b.shape)
        
        plt.clf()
        plt.subplot(1, 1, 1, projection='northpolar')
        plt.plot(angle[1:], arbitrary_data)
        plt.show()
    '''

