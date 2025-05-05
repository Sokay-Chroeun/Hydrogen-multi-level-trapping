import matplotlib.pyplot as plt
import numpy as np

thetaT = np.linspace(0,6,600)

theta = [1, 2, 3, 4, 5, 6]
Epi = [1.31, 1.3, 1.19, 1.17, 1.06, 0.85]


x = np.linspace(1,6,20)
xEpi = -0.0179*x**2 + 0.0381*x + 1.284     # parabolique

plt.figure(2,dpi=100)

plt.scatter(theta, Epi, label='[Hodille 2016]', marker='D')
plt.plot(x, xEpi, label='This current work', color = 'r')

plt.legend(ncol=1)
# plt.grid()
plt.xlabel('Number of trapped H atoms',fontsize=12)
plt.ylabel('Detrapping energy [eV]',fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylim([0.4,1.7])
plt.savefig("Ep_TDS-0D.png",bbox_inches='tight')
