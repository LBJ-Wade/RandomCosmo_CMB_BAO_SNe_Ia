# import libraries
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pylab as pil
import seaborn as sns
from matplotlib import gridspec

sns.set()

fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2,2, height_ratios=[1,0.5],width_ratios=[0.5,1.5], hspace=0.02, wspace=0)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[0, 1])
ax4 = plt.subplot(gs[1, 1])

# load some data
#Planck_CMB_TT = np.loadtxt('Planck_CMB_TT_spectra.dat')
Planck_CMB_TT = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt')
#GR_cl  = np.loadtxt('testGR_scalCls.dat')
GR_cl  = np.loadtxt('COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt')


#cls = np.loadtxt('/scratch/l/levon/azucca/sampler_IC_GBD_fit_weight2_IC_cls_TT.dat')
#cls = np.loadtxt('/scratch/l/levon/azucca/sampler_IC_GBD_fit_weight2_IC_cls_TT.dat')
cls = np.loadtxt('/scratch/l/levon/azucca/poly_wei3_IC_cls_TT.dat')
#cls = np.loadtxt('powlaw_wei_tanh_IC_cls_TT.dat')

# compute average cls
#avg_cls = np.zeros(799)
#for i in range(799):
#    avg_cls[i] = 0
#    for j in range(cls.shape[0]):
#        avg_cls[i]+=cls[j,i]/cls.shape[0]

ax1.semilogx(GR_cl[:,0], GR_cl[:,1], label=r'$\Lambda$CDM', linestyle='-', linewidth=1 ,color = 'black', zorder=4)
ax3.plot(GR_cl[:,0], GR_cl[:,1], linestyle='-', linewidth=1 ,color = 'black', zorder=4, label=r'$\Lambda$CDM')
#ax1.errorbar(Planck_CMB_TT[:28,0], Planck_CMB_TT[:28,1], yerr=[Planck_CMB_TT[:28,2], Planck_CMB_TT[:28,3]], linewidth=1.0, fmt='none', color = 'red', alpha=0.5,zorder=2)
#ax1.scatter(Planck_CMB_TT[:28,0],  Planck_CMB_TT[:28,1], color='red', alpha=0.5, s=10.0, zorder=3,label='Planck TT')
#ax3.errorbar(Planck_CMB_TT[28:,0], Planck_CMB_TT[28:,1], yerr=[Planck_CMB_TT[28:,2], Planck_CMB_TT[28:,3]], linewidth=1.0, fmt='none', color = 'red', alpha=0.5,zorder=2)
#ax3.scatter( Planck_CMB_TT[28:,0], Planck_CMB_TT[28:,1], color='red', alpha=0.5, s=10.0, zorder=3,label='Planck TT')


GR_cl_new = GR_cl[:799,:]


for i in range(cls.shape[0]):   
#for i in range(100):   
    if cls[i,798] < 3000:
        ax1.semilogx(range(2,cls.shape[1]+2), cls[i,:], linewidth=1, color='C0', alpha=0.01, zorder=1)
        ax3.plot(range(2,cls.shape[1]+2), cls[i,:], linewidth=1, color='C0', alpha=0.01, zorder=1)
        ax2.semilogx(range(2,cls.shape[1]+2),  (cls[i,:]-GR_cl_new[:,1]), linewidth=1, color='C0', alpha=0.01, zorder=1)
        ax4.plot(range(2,cls.shape[1]+2),  (cls[i,:]-GR_cl_new[:,1]), linewidth=1, color='C0', alpha=0.01, zorder=1)


ax1.grid(b=True, which='minor', color='w', linewidth=0.5)
ax2.grid(b=True, which='minor', color='w', linewidth=0.5)

ax1.axvline(30, linestyle='--', linewidth=1, alpha=0.5)
ax2.axvline(30, linestyle='--', linewidth=1, alpha=0.5)

ax1.set_xlim(1.7,30)
ax1.set_ylim(0,6.5e3)
ax3.set_xlim(30,800)
ax3.set_ylim(0,6.5e3)
ax3.set_yticklabels([])
ax1.set_xticklabels([])
ax3.set_xticklabels([])

ax3.legend(loc='upper right', fontsize= 13)
#ax1.set_xlabel(r'$\ell$', fontsize= 13)
ax1.set_ylabel(r'$\ell (\ell+1) / (2 \pi) \, C_{\ell}^{\rm TT}$ $[\mu {\rm K}^2]$', fontsize= 13)

ax2.set_xlim(1.7,30)
ax4.set_xlabel(r'$\ell$           	', fontsize= 13)
ax2.set_ylim(-650,650)
ax4.set_xlim(30,800)
ax4.set_ylim(-650,650)
ax2.set_ylabel(r'$\Delta C_{\ell}^{\rm TT}$', fontsize= 13)

ax4.set_yticklabels([])

#pil.savefig('powlaw_wei_tanh_IC_cls_TT.pdf', bbox_inches='tight')
pil.savefig('poly_wei3_IC_cls_TT.pdf', bbox_inches='tight')
