
#cal delta rmse

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from scipy.stats import linregress, pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.colors as colors
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import namedtuple
import xskillscore as xs
#read data

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
NorCPM_grid = xr.open_dataset('H:/NorCPM/data/grid/ocn_grid.nc')
plon,plat,parea,pdepth = NorCPM_grid['plon'].values,NorCPM_grid['plat'].values,NorCPM_grid['parea'].values,NorCPM_grid['pdepth'].values
region_dir = 'H:/NorCPM/data/grid/Region.nc'
region_nc  = xr.open_dataset(region_dir)
region = region_nc['region']
#oice = np.load('SIE_Hindcast_0321.npz')['oice']
pice = np.load('H:/Hybrid_model/evaluate/evaluate_v4/data/SIE_Hindcast_case10.npz')['hyice']
MLice = np.load('H:/Hybrid_model/evaluate/evaluate_v4/data/SIE_offline_ML_case10.npz')['hyice']
hyice = np.load('H:/Hybrid_model/evaluate/evaluate_v4/data/SIE_Hybrid_ML_case10.npz')['hyice']
#SIE_boottrap_base_hybrid_07.npz
anaice = np.load('H:/Hybrid_model/evaluate/evaluate_v4/data/SIE_reanalysis.npz')['anaice']

pice_new = np.zeros((7,10,13,12,19))
hyice_new = np.zeros((7,10,13,12,19))
MLice_new = np.zeros((7,10,13,12,19))
anaice_new = np.zeros((7,10,13,12,19))
off_base_new = np.zeros((7,10,13,12,19))
pice_new[0]=pice[0]
pice_new[1]=pice[1]
pice_new[2]=np.sum(pice[2:5],axis=0)
pice_new[3]=np.sum(pice[5:7],axis=0)
pice_new[4]=pice[7]+pice[10]
pice_new[5]=np.sum(pice[11:15],axis=0)
pice_new[6]=np.sum(pice[8:10],axis=0)

hyice_new[0]=hyice[0]
hyice_new[1]=hyice[1]
hyice_new[2]=np.sum(hyice[2:5],axis=0)
hyice_new[3]=np.sum(hyice[5:7],axis=0)
hyice_new[4]=hyice[7]+hyice[10]
hyice_new[5]=np.sum(hyice[11:15],axis=0)
hyice_new[6]=np.sum(hyice[8:10],axis=0)

MLice_new[0]=MLice[0]
MLice_new[1]=MLice[1]
MLice_new[2]=np.sum(MLice[2:5],axis=0)
MLice_new[3]=np.sum(MLice[5:7],axis=0)
MLice_new[4]=MLice[7]+MLice[10]
MLice_new[5]=np.sum(MLice[11:15],axis=0)
MLice_new[6]=np.sum(MLice[8:10],axis=0)

anaice_new[0]=anaice[0]
anaice_new[1]=anaice[1]
anaice_new[2]=np.sum(anaice[2:5],axis=0)
anaice_new[3]=np.sum(anaice[5:7],axis=0)
anaice_new[4]=anaice[7]+anaice[10]
anaice_new[5]=np.sum(anaice[11:15],axis=0)
anaice_new[6]=np.sum(anaice[8:10],axis=0)

SIE_hi = pice_new.transpose(0,3,4,1,2)
SIE_ML_off = MLice_new.transpose(0,3,4,1,2)
SIE_ML_on = hyice_new.transpose(0,3,4,1,2)
SIE_ana = anaice_new.transpose(0,3,4,1,2)

from scipy.stats import ttest_ind

def cal_rmse(ana_detrend,hind_detrend):
    SE = np.square(ana_detrend-hind_detrend)
    RMSE= np.sqrt(np.mean(SE,axis=0))
    return np.array([RMSE])

p_ml_of = np.zeros((11,12,7))
p_ml_on  = np.zeros((11,12,7))
p_d  = np.zeros((11,12,7))
rmse_hi = np.zeros((11,12,7))
rmse_on = np.zeros((11,12,7))
rmse_of = np.zeros((11,12,7))
for lo_ind in range(7):
    for start_m in [1,4,7,10]:
        for le_m in range(1,12):
            #if le_m==5:
                #if lo_ind==6:
            
                    mm_t=np.mod((start_m+le_m-1),12)
                    
                    le_m1 =le_m-1
                    di=np.mean(SIE_hi[lo_ind,start_m-1,:,:,le_m],axis=1)
                    dana=np.mean(SIE_ana[lo_ind,start_m-1,:,:,le_m],axis=1)
                    don=np.mean(SIE_ML_on[lo_ind,start_m-1,:,:,le_m],axis=1)
                    dof=np.mean(SIE_ML_off[lo_ind,start_m-1,:,:,le_m],axis=1)
                    rmse_d = cal_rmse(dana,dof)
                    rmse_of[le_m1,mm_t,lo_ind] = np.copy(rmse_d)
                    rmse_d = cal_rmse(dana,di)
                    rmse_hi[le_m1,mm_t,lo_ind] = np.copy(rmse_d)
                    rmse_on[le_m1,mm_t,lo_ind] = cal_rmse(dana,don)
                    
                    statistic, p_value = ttest_ind(np.square(di-dana),np.square(don-dana))
                    p_ml_on[le_m1,mm_t,lo_ind]=p_value
                    statistic, p_value = ttest_ind(np.square(di-dana),np.square(dof-dana))
                    p_ml_of[le_m1,mm_t,lo_ind]=p_value
                    statistic, p_value = ttest_ind(np.square(di-don),np.square(dof-di))
                    p_d[le_m1,mm_t,lo_ind]=p_value





#%%

import numpy as np
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


projection = ccrs.NorthPolarStereo()
x, y, _ = projection.transform_points(ccrs.PlateCarree(), plon, plat).T
rows = 1
cols = 3
fig = plt.figure(figsize=(30, 10))
gs = gridspec.GridSpec(rows, cols , figure=fig)
import numpy as np
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
lim=0.3
ia = 19
levels = np.linspace(-lim, lim,ia)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(levels)))
colors = np.insert(colors, int((ia-1)/2), colors[int((ia-1)/2)], axis=0)

color = np.array(
    [
            [148,17,0],
            [232,111,15],
            [250,198,0],
            [241,232,78],
            [243,244,194],
            [200,226,197],
            [138,200,163],
            [110,196,228],
            [69,77,157],
            [40,61,147]
    ]
)/255
#color = np.insert(color, int((ia-1)/2), colors[int((ia-1)/2)], axis=0)

cmap = ListedColormap(color[::-1,:], name='mycmap')
lim=0.3
cmap = LinearSegmentedColormap.from_list('mycmap', color[::-1, :], N=8)
colors = np.insert(colors, int((ia-1)/2), colors[int((ia-1)/2)], axis=0)
#cmap = ListedColormap(colors)
norm = BoundaryNorm(levels, len(colors))
# 假设已经定义了 rmse_on_m, rmse_hi_m, cmap, lim, cbar_kw 和 p_ml_on
p_ml_on[p_ml_on==0]=np.nan
ss_on =  - rmse_on + rmse_hi
ss_on[np.isinf(ss_on)] = np.nan
ss_on[np.isnan(ss_on)] = 0
ss_on[np.isnan(p_ml_on)] = np.nan

p_ml_of[p_ml_of==0]=np.nan
p_d[p_d==0]=np.nan
ss_of =  - rmse_of + rmse_hi
ss_of[np.isinf(ss_of)] = np.nan
ss_of[np.isnan(ss_of)] = 0
ss_of[np.isnan(p_ml_of)] = np.nan
fs=28
cbar_kw = {"shrink": 0.65}
for i in range(rows * cols):
        print(i)
        if i ==0:
            ax = fig.add_subplot(gs[i // cols, i % cols])
            ax = sns.heatmap(ss_on[:,:,0], cmap=cmap,vmin=-lim,vmax=lim,cbar=False, square=True, linecolor='black', linewidths=0, cbar_kws=cbar_kw)
            indices = np.where(p_ml_on[:,:,0] > 0.05)
            if indices[0].size > 0:
                ax.scatter(indices[1]+0.5, indices[0]+0.5, marker='o', color='black', s=40)
            ax.tick_params(axis='both', labelsize=fs)
            ax.set_yticks(np.arange(11)+0.5) 
            ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], fontsize=fs)
            
            
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['top'].set_visible(True)
    
            ax.invert_yaxis()
            ax.set_xticks(np.arange(12)+0.5)
            ax.set_title('Reference-OnlineML', fontsize=fs+2)
            ax.text(0.02, 1.08, f'({chr(97 + i)})', transform=ax.transAxes, fontsize=fs+2,
                   verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 5})
            ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=fs)
            plt.xlabel('Target month',fontsize=fs)
            plt.ylabel('Lead month',fontsize=fs)

        if i ==1:
            ax = fig.add_subplot(gs[i // cols, i % cols])
            ax = sns.heatmap(ss_of[:,:,0], cmap=cmap,vmin=-lim,vmax=lim,cbar=False, square=True, linecolor='black', linewidths=0, cbar_kws=cbar_kw)
            indices = np.where(p_ml_of[:,:,0] > 0.05)
            ax.scatter(indices[1]+0.5, indices[0]+0.5, marker='o', color='black', s=40)
            ax.tick_params(axis='both', labelsize=fs)
            ax.set_yticks(np.arange(11)+0.5) 
            ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], fontsize=fs)
            
            
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['top'].set_visible(True)
    
            ax.invert_yaxis()
            ax.set_xticks(np.arange(12)+0.5)
            ax.set_title('Reference-OfflineML', fontsize=fs+2)
            ax.text(0.02, 1.08, f'({chr(97 + i)})', transform=ax.transAxes, fontsize=fs+2,
                   verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 5})
            ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=fs)
            
            plt.xlabel('Target month',fontsize=fs)
            plt.ylabel('Lead month',fontsize=fs)

            
        if i ==2:
            ax = fig.add_subplot(gs[i // cols, i % cols])
            ax = sns.heatmap(ss_on[:,:,0]-ss_of[:,:,0], cmap=cmap,vmin=-lim,vmax=lim,cbar=False, square=True, linecolor='black', linewidths=0, cbar_kws=cbar_kw)
            indices = np.where(p_d[:,:,0] > 0.05)
            ax.scatter(indices[1]+0.5, indices[0]+0.5, marker='o', color='black', s=40)
            ax.tick_params(axis='both', labelsize=fs)
            ax.set_yticks(np.arange(11)+0.5) 
            ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], fontsize=fs)
            
            
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['top'].set_visible(True)
            plt.xlabel('Target month',fontsize=fs)
            plt.ylabel('Lead month',fontsize=fs)
    
            ax.invert_yaxis()
            ax.set_xticks(np.arange(12)+0.5)
            ax.set_title('OfflineML-OnlineML', fontsize=fs+2)
            ax.text(0.02, 1.08, f'({chr(97 + i)})', transform=ax.transAxes, fontsize=fs+2,
                   verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 5})
            ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=fs)

colorbar_ax = fig.add_axes([0.94, 0.20, 0.01, 0.60])  # 调整参数以适应您的布局
colorbar = plt.colorbar(ax.collections[0], cax=colorbar_ax, shrink=0.9, ticks=np.arange(-2, 2, 0.1))
colorbar.ax.tick_params(labelsize=38)
colorbar.set_label('$\Delta$RMSE ($\\times 10^6$ km$^2$)', fontsize=38)
plt.savefig('SIE_pan_seasonal.png',bbox_inches='tight')         
plt.show()
#%%
cmap = LinearSegmentedColormap.from_list('mycmap', color[::-1, :], N=8)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec

rows = 4
cols = 3
lim=0.2
p_ml_on[p_ml_on==0]=np.nan
ss_on =   rmse_hi - rmse_on
ss_on[np.isinf(ss_on)] = np.nan
ss_on[np.isnan(ss_on)] = 0
ss_on[np.isnan(p_ml_on)] = np.nan

p_ml_of[p_ml_of==0]=np.nan
p_d[p_d==0]=np.nan
ss_of = rmse_hi - rmse_of
ss_of[np.isinf(ss_of)] = np.nan
ss_of[np.isnan(ss_of)] = 0
ss_of[np.isnan(p_ml_of)] = np.nan

fig = plt.figure(figsize=(30, 40))
gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.4)
lo_idx = ['Pan Arctic', 'Central Arctic', 'Atlantic', 'Siberian', 'Alaskan', 'Candian', 'Pacific']

for i in range(rows * cols):
    if i<=2:
        ax = fig.add_subplot(gs[i // cols, i % cols])
        i=i+1
        ax = sns.heatmap(ss_on[:, :, i], cmap=cmap, vmin=-lim, vmax=lim, cbar=False, square=True, linecolor='black', linewidths=0, cbar_kws=cbar_kw)
        
        ax.set_yticks(np.arange(11)+0.5) 
        ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], fontsize=30)
        indices = np.where(p_ml_on[:, :, i] > 0.05)
        if indices[0].size > 0:
            ax.scatter(indices[1]+0.5, indices[0]+0.5, marker='o', color='black', s=60)
        
        ax.tick_params(axis='both', labelsize=30)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(12)+0.5)
        ax.set_title(lo_idx[i], fontsize=36)
        ax.text(0.01, 1.08, f'({chr(96 + i)})', transform=ax.transAxes, fontsize=38, verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 5})
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=30)
        plt.xlabel('Target month', fontsize=38)
        plt.ylabel('Lead month', fontsize=38)
        if i==1:
            ax.text(-0.9, 0.6, 'OnlineML', transform=ax.transAxes, fontsize=50, verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 5}, weight='bold')
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=30)
        plt.xlabel('Target month', fontsize=38)
        plt.ylabel('Lead month', fontsize=38)
    if 2<i<=5:
        ax = fig.add_subplot(gs[i // cols, i % cols])
        i=i-2
        ax = sns.heatmap(ss_of[:, :, i], cmap=cmap, vmin=-lim, vmax=lim, cbar=False, square=True, linecolor='black', linewidths=0, cbar_kws=cbar_kw)
        ax.set_yticks(np.arange(11)+0.5) 
        ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], fontsize=30)
        indices = np.where(p_ml_of[:, :, i] > 0.05)
        ax.scatter(indices[1]+0.5, indices[0]+0.5, marker='o', color='black', s=60)
        if i==1:
            ax.text(-0.9, 0.6, 'OfflineML', transform=ax.transAxes, fontsize=50, verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 5}, weight='bold')        
        ax.tick_params(axis='both', labelsize=30)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(12)+0.5)
        ax.set_title(lo_idx[i], fontsize=36)
        ax.text(0.01, 1.08, f'({chr(96+3 + i)})', transform=ax.transAxes, fontsize=38, verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 5})
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=30)
        plt.xlabel('Target month', fontsize=38)
        plt.ylabel('Lead month', fontsize=38)

    if 5<i<=8:
        ax = fig.add_subplot(gs[i // cols, i % cols])
        i=i-2
        ax = sns.heatmap(ss_on[:, :, i], cmap=cmap, vmin=-lim, vmax=lim, cbar=False, square=True, linecolor='black', linewidths=0, cbar_kws=cbar_kw)
        ax.set_yticks(np.arange(11)+0.5) 
        ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], fontsize=30)
        indices = np.where(p_ml_on[:, :, i] > 0.05)
        ax.scatter(indices[1]+0.5, indices[0]+0.5, marker='o', color='black', s=60)
        
        ax.tick_params(axis='both', labelsize=30)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(12)+0.5)
        ax.set_title(lo_idx[i], fontsize=36)
        ax.text(0.01, 1.08, f'({chr(96+3 + i)})', transform=ax.transAxes, fontsize=38, verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 5})
        if i==4:
            ax.text(-0.9, 0.6, 'OnlineML', transform=ax.transAxes, fontsize=50, verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 5}, weight='bold')
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=30)
        plt.xlabel('Target month', fontsize=38)
        plt.ylabel('Lead month', fontsize=38)
        
    if 8<i:
        ax = fig.add_subplot(gs[i // cols, i % cols])
        i=i-5
        print(i)
        ax = sns.heatmap(ss_of[:, :, i], cmap=cmap, vmin=-lim, vmax=lim, cbar=False, square=True, linecolor='black', linewidths=0, cbar_kws=cbar_kw)
        ax.set_yticks(np.arange(11)+0.5) 
        ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], fontsize=30)
        indices = np.where(p_ml_of[:, :, i] > 0.05)
        ax.scatter(indices[1]+0.5, indices[0]+0.5, marker='o', color='black', s=60)
        if i==4:
            ax.text(-0.9, 0.6, 'OfflineML', transform=ax.transAxes, fontsize=50, verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 5}, weight='bold')        
       
        ax.tick_params(axis='both', labelsize=30)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(12)+0.5)
        ax.set_title(lo_idx[i], fontsize=36)
        ax.text(0.01, 1.08, f'({chr(96+6 + i)})', transform=ax.transAxes, fontsize=38, verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 5})
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=30)
        plt.xlabel('Target month', fontsize=38)
        plt.ylabel('Lead month', fontsize=38)


colorbar_ax = fig.add_axes([0.94, 0.20, 0.02, 0.60])  # 调整参数以适应您的布局
colorbar = plt.colorbar(ax.collections[0], cax=colorbar_ax, shrink=0.9, ticks=np.arange(-2, 2, 0.05))
colorbar.ax.tick_params(labelsize=38)
colorbar.set_label('$\Delta$RMSE ($\\times 10^6$ km$^2$)', fontsize=38)

#plt.tight_layout() 

plt.savefig('FigureS1.png', bbox_inches='tight',dpi=300)
plt.show()

