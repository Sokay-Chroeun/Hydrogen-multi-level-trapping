import matplotlib.pyplot as plt

def readData(FileName):
    txt = [];X = [];Y = [];
    for Line in open(FileName,'r'):
        t = Line.split()
        txt.append(t)
    for i in range(len(txt)-1):
        X.append(float(txt[i+1][0]))
        Y.append(float(txt[i+1][1]))
    return (X,Y)

x1, y1 = readData('TDS_Hodille.txt')
x2, y2 = readData('TDS_Abaqus2.txt')
x3, y3 = readData('TDS_Hodille2.txt')
Y2 = []
Y3 = []
for i in range(len(y2)):
    Y2.append(y2[i]*6.32e25) # Convertion de l'unité de flux 

plt.figure(1, figsize=(7,4))
plt.scatter(x3,y3, marker = '+',label = 'Experiment [6]', color = 'black',s=50)
plt.plot(x1,y1, label = 'Sequential kinetics [3]', linestyle='--')
plt.plot(x2,Y2, label='Simultaneous kinetics',color='Tab:red')
#plt.ylim([-0.1e19, 1.75e19])
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Temperature [K]',fontsize=12)
plt.ylabel('Desorption rate [D/m$^2$.s]',fontsize=12)
plt.legend(loc='upper left')
# plt.grid()
plt.savefig("Hodille_TDS.png",bbox_inches='tight',dpi=500)
