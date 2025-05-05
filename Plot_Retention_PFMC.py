import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#data = pd.read_excel('Retention_PFMC.xlsx')

t = np.linspace(0,1.56e5,500)
Ret1 = 1e15*t**0.558
Ret2 = 1e15*t**0.582

plt.figure(1,dpi=100)
plt.plot(t, Ret1, label='Ep($\\theta_{Ti}$)')
plt.plot(t, Ret2, label='Ep($\\theta_{T1}$)')
# plt.plot(data.t, data.QH, label='Ep($\\theta_{Ti}$')
# plt.plot(data.t2, data.QH2, label='Ep($\\theta_{T1}$')

plt.legend()

plt.xlabel('Time [s]',fontsize=12)
plt.ylabel('Hydrogen atoms per monoblock',fontsize=12)
plt.ticklabel_format(axis="x", style="scientific", scilimits=(0,0))
plt.legend(ncol=1, loc='lower right',fontsize=12)
plt.ylim([0,1.2e18])
plt.xlim([0,1.6e5])
#plt.yscale('log')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.savefig("Retention_PFMC.png",bbox_inches='tight')
