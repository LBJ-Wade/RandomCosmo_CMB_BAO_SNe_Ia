 import libraries
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pylab as pil
import seaborn as sns

sns.set()

# load some data
Planck_CMB_TT = np.loadtxt('Planck_CMB_TT_spectra.dat')
GR_cl  = np.loadtxt('testGR_scalCls.dat')
GR_pk  = np.loadtxt('testGR_matterpower.dat')


#cls = np.loadtxt('/scratch/l/levon/azucca/sampler_IC_GBD_fit_weight2_IC_cls_TT.dat')
cls = np.loadtxt('poly_std_IC_cls_TT.dat')
#cls = np.loadtxt('/scratch/l/levon/azucca/poly_std_IC_cls_TT.dat')
#cls = np.loadtxt('/scratch/l/levon/azucca/poly_std_tanh_IC_cls_TT.dat')
#cls = np.loadtxt('/scratch/l/levon/azucca/poly_wei_tanh_IC_cls_TT.dat')
#cls = np.loadtxt('/scratch/l/levon/azucca/powlaw_wei_IC_cls_TT.dat')
#cls = np.loadtxt('/scratch/l/levon/azucca/powlaw_std_IC_cls_TT.dat')
#cls = np.loadtxt('/scratch/l/levon/azucca/powlaw_std_tanh_IC_cls_TT.dat')
#cls = np.loadtxt('/scratch/l/levon/azucca/powlaw_wei_tanh_IC_cls_TT.dat')


# compute average cls
avg_cls = np.zeros(799)
for i in range(799):
    avg_cls[i] = 0
    for j in range(cls.shape[0]):
        avg_cls[i]+=cls[j,i]/cls.shape[0]

plt.semilogx(GR_cl[:,0], GR_cl[:,1], label=r'$\Lambda$CDM', linestyle='--', linewidth=1 ,color = 'black', zorder=4)
plt.errorbar(Planck_CMB_TT[:,0], Planck_CMB_TT[:,1], yerr=[Planck_CMB_TT[:,2], Planck_CMB_TT[:,3]], linewidth=0.5, fmt='none', color = 'red', alpha=0.5,zorder=2)
plt.scatter(Planck_CMB_TT[:,0], Planck_CMB_TT[:,1], color='red', alpha=0.5, s=15.0, zorder=3,label='Planck TT')
plt.semilogx(range(2,cls.shape[1]+2), avg_cls, linewidth=1, color='white', zorder=2)

for i in range(cls.shape[0]):
    if cls[i,798] < 3000:
        plt.semilogx(range(2,cls.shape[1]+2), cls[i,:], linewidth=1, color='C0', alpha=0.01, zorder=1)
        #plt.fill_between(range(2,cls.shape[1]+2), cls[0,:], cls[i,:],  color='C0', alpha=0.01)

plt.xlim(2,2000)
plt.ylim(0,7e3)
plt.legend(loc='upper left', fontsize= 20)
plt.xlabel(r'$\ell$', fontsize= 20)
plt.ylabel(r'$\ell (\ell+1) / (2 \pi) \, C_{\ell}^{\rm TT}$', fontsize= 20)
pil.savefig('poly_std_IC_cls_TT.pdf', bbox_inches='tight')
#plt.show()

