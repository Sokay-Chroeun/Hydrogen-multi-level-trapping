import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

E_constant = False
TDS = True
E = 1.3

def MacNabb1(C, t):
    T, cL, CT = C
    LatticeP = 3.16E-7 * 3.16E-7
    Nt = 3.62e-3
    NL = 6.32e19
    thetaT = CT/Nt
    phi = 0.0059*thetaT**3 -0.0083*thetaT**2 
    phi2 = 0.0059*(6-thetaT)**3 - 0.0085*(6-thetaT)**2 + 0.0023*(6-thetaT)
    phi=1
    phi2=1
    xEpi = -0.0179*thetaT**2 + 0.0381*thetaT + 1.284     # parabolique
    k = (1.38e-1/LatticeP)*np.exp(-0.39/(8.62e-5*T))
    p = 1e13*np.exp(-xEpi/(8.62e-5*T)) 
    dCTdt = k/NL*cL*(Nt - CT/6) - p*CT
    dCLdt = 0.
    dTdt = 0.
    return [dTdt, dCLdt, dCTdt]

def MacNabb2(C, t):
    T, cL, CT = C
    LatticeP = 3.16E-7 * 3.16E-7
    Nt = 3.62e-3
    NL = 6.32e19
    thetaT = CT/Nt
    phi = 0.0059*thetaT**3 -0.0083*thetaT**2 
    phi2 = 0.0059*(6-thetaT)**3 - 0.0085*(6-thetaT)**2 + 0.0023*(6-thetaT)
    phi=1
    phi2=1
    xEpi = -0.0179*thetaT**2 + 0.0381*thetaT + 1.284     # parabolique
    k = (1.38e-1/LatticeP)*np.exp(-0.39/(8.62e-5*T))
    p = 1e13*np.exp(-xEpi/(8.62e-5*T)) 
    #dCTdt = k/NL*cL*(Nt - CT/6*phi) - p*Nt*(1 - phi2)
    dCTdt = k/NL*cL*(Nt - CT/6) - p*CT
    dCLdt = 0.
    dTdt = 0.
    return [dTdt, dCLdt, dCTdt]

def MacNabb3(C, t):
    T, cL, CT = C
    LatticeP = 3.16E-7 * 3.16E-7
    Nt = 3.62e-3
    NL = 6.32e19
    thetaT = CT/Nt
    phi = 0.0059*thetaT**3 -0.0083*thetaT**2 
    phi2 = 0.0059*(6-thetaT)**3 - 0.0085*(6-thetaT)**2 + 0.0023*(6-thetaT)
    xEpi = -0.0179*thetaT**2 + 0.0381*thetaT + 1.284     # parabolique
    phi=1
    phi2=1
    xEpi = -0.0179*thetaT**2 + 0.0381*thetaT + 1.284     # parabolique
    k = (1.38e-1/LatticeP)*np.exp(-0.39/(8.62e-5*T))
    p = 1e13*np.exp(-xEpi/(8.62e-5*T)) 
    #dCTdt = k/NL*cL*(Nt - CT/6*phi) - p*Nt*(1 - phi2)
    dCTdt = k/NL*cL*(Nt - CT/6) - p*CT
    dCLdt = 0.
    dTdt = 5.1
    return [dTdt, dCLdt, dCTdt]

def Hodille1(C, t):
    T, cL, N0, N1, N2, N3, N4, N5, N6, Ct  = C
    LatticeP = 3.16E-7 * 3.16E-7
    NL = 6.32e19
    xev = 1.602176634e-19*6.022e23
    k = (1.38e-1/LatticeP)*np.exp(-0.39*xev/(8.31*T))
    Epi = [1.3042, 1.2886, 1.2372, 1.1500, 1.0270, 0.8682]   # parabolique
    #Epi = [1.31, 1.3, 1.19, 1.17, 1.06, 0.85]               # Hodille
    #Epi = [1.3099, 1.218, 1.126, 1.03399, 0.942, 0.84999]    # Lineaire
    p1 = 1e13*np.exp(-Epi[0]/(8.62e-5*T))
    p2 = 1e13*np.exp(-Epi[1]/(8.62e-5*T))
    p3 = 1e13*np.exp(-Epi[2]/(8.62e-5*T))
    p4 = 1e13*np.exp(-Epi[3]/(8.62e-5*T))
    p5 = 1e13*np.exp(-Epi[4]/(8.62e-5*T))
    p6 = 1e13*np.exp(-Epi[5]/(8.62e-5*T))
    if E_constant:
        p1 = 1e13*np.exp(-E/(8.62e-5*T))
        p2 = 1e13*np.exp(-E/(8.62e-5*T))
        p3 = 1e13*np.exp(-E/(8.62e-5*T))
        p4 = 1e13*np.exp(-E/(8.62e-5*T))
        p5 = 1e13*np.exp(-E/(8.62e-5*T))
        p6 = 1e13*np.exp(-E/(8.62e-5*T))
    dN0dt = -k*cL/NL*N0 + p1*N1
    dN1dt = -k*cL/NL*N1 + k*cL/NL*N0 - p1*N1 + p2*N2
    dN2dt = -k*cL/NL*N2 + k*cL/NL*N1 - p2*N2 + p3*N3
    dN3dt = -k*cL/NL*N3 + k*cL/NL*N2 - p3*N3 + p4*N4
    dN4dt = -k*cL/NL*N4 + k*cL/NL*N3 - p4*N4 + p5*N5
    dN5dt = -k*cL/NL*N5 + k*cL/NL*N4 - p5*N5 + p6*N6
    dN6dt = k*cL/NL*N5 - p6*N6
    #dCt = k*cL/NL*(Nt - N6) - (p1*N1 + p2*N2 + p3*N3 + p4*N4 + p5*N5 + p6*N6)
    dCt = dN1dt + 2*dN2dt + 3*dN3dt + 4*dN4dt + 5*dN5dt + 6*dN6dt
    #dCt = k*cL/NL*(Nt - Ct) - (p1*N1 + p2*N2 + p3*N3 + p4*N4 + p5*N5 + p6*N6)
    dCLdt = 0.
    dTdt = 0.
    return [dTdt, dCLdt, dN0dt, dN1dt, dN2dt, dN3dt, dN4dt, dN5dt, dN6dt, dCt]

def Hodille2(C, t):
    T, cL, N0, N1, N2, N3, N4, N5, N6, Ct  = C
    LatticeP = 3.16E-7 * 3.16E-7
    NL = 6.32e19
    xev = 1.602176634e-19*6.022e23
    k = (1.38e-1/LatticeP)*np.exp(-0.39*xev/(8.31*T))
    Epi = [1.3042, 1.2886, 1.2372, 1.1500, 1.0270, 0.8682]   # parabolique
    #Epi = [1.31, 1.3, 1.19, 1.17, 1.06, 0.85]               # Hodille
    #Epi = [1.3099, 1.218, 1.126, 1.03399, 0.942, 0.84999]    # Lineaire
    p1 = 1e13*np.exp(-Epi[0]/(8.62e-5*T))
    p2 = 1e13*np.exp(-Epi[1]/(8.62e-5*T))
    p3 = 1e13*np.exp(-Epi[2]/(8.62e-5*T))
    p4 = 1e13*np.exp(-Epi[3]/(8.62e-5*T))
    p5 = 1e13*np.exp(-Epi[4]/(8.62e-5*T))
    p6 = 1e13*np.exp(-Epi[5]/(8.62e-5*T))
    if E_constant:
        p1 = 1e13*np.exp(-E/(8.62e-5*T))
        p2 = 1e13*np.exp(-E/(8.62e-5*T))
        p3 = 1e13*np.exp(-E/(8.62e-5*T))
        p4 = 1e13*np.exp(-E/(8.62e-5*T))
        p5 = 1e13*np.exp(-E/(8.62e-5*T))
        p6 = 1e13*np.exp(-E/(8.62e-5*T))
    dN0dt = -k*cL/NL*N0 + p1*N1
    dN1dt = -k*cL/NL*N1 + k*cL/NL*N0 - p1*N1 + p2*N2
    dN2dt = -k*cL/NL*N2 + k*cL/NL*N1 - p2*N2 + p3*N3
    dN3dt = -k*cL/NL*N3 + k*cL/NL*N2 - p3*N3 + p4*N4
    dN4dt = -k*cL/NL*N4 + k*cL/NL*N3 - p4*N4 + p5*N5
    dN5dt = -k*cL/NL*N5 + k*cL/NL*N4 - p5*N5 + p6*N6
    dN6dt = k*cL/NL*N5 - p6*N6
    #dCt = k*cL/NL*(Nt - N6) - (p1*N1 + p2*N2 + p3*N3 + p4*N4 + p5*N5 + p6*N6)
    dCt = dN1dt + 2*dN2dt + 3*dN3dt + 4*dN4dt + 5*dN5dt + 6*dN6dt
    #dCt = k*cL/NL*(Nt - Ct) - (p1*N1 + p2*N2 + p3*N3 + p4*N4 + p5*N5 + p6*N6)
    dCLdt = 0.
    dTdt = 0.
    return [dTdt, dCLdt, dN0dt, dN1dt, dN2dt, dN3dt, dN4dt, dN5dt, dN6dt, dCt]

def Hodille3(C, t):
    T, cL, N0, N1, N2, N3, N4, N5, N6, Ct  = C
    LatticeP = 3.16E-7 * 3.16E-7
    NL = 6.32e19
    k = (1.38e-1/LatticeP)*np.exp(-0.39/(8.62e-5*T))
    Epi = [1.3042, 1.2886, 1.2372, 1.1500, 1.0270, 0.8682]   # parabolique
    #Epi = [1.31, 1.3, 1.19, 1.17, 1.06, 0.85]               # Hodille
    #Epi = [1.3099, 1.218, 1.126, 1.03399, 0.942, 0.84999]    # Lineaire
    p1 = 1e13*np.exp(-Epi[0]/(8.62e-5*T))
    p2 = 1e13*np.exp(-Epi[1]/(8.62e-5*T))
    p3 = 1e13*np.exp(-Epi[2]/(8.62e-5*T))
    p4 = 1e13*np.exp(-Epi[3]/(8.62e-5*T))
    p5 = 1e13*np.exp(-Epi[4]/(8.62e-5*T))
    p6 = 1e13*np.exp(-Epi[5]/(8.62e-5*T))
    if E_constant:
        p1 = 1e13*np.exp(-E/(8.62e-5*T))
        p2 = 1e13*np.exp(-E/(8.62e-5*T))
        p3 = 1e13*np.exp(-E/(8.62e-5*T))
        p4 = 1e13*np.exp(-E/(8.62e-5*T))
        p5 = 1e13*np.exp(-E/(8.62e-5*T))
        p6 = 1e13*np.exp(-E/(8.62e-5*T))
    dN0dt = -k*cL/NL*N0 + p1*N1
    dN1dt = -k*cL/NL*N1 + k*cL/NL*N0 - p1*N1 + p2*N2
    dN2dt = -k*cL/NL*N2 + k*cL/NL*N1 - p2*N2 + p3*N3
    dN3dt = -k*cL/NL*N3 + k*cL/NL*N2 - p3*N3 + p4*N4
    dN4dt = -k*cL/NL*N4 + k*cL/NL*N3 - p4*N4 + p5*N5
    dN5dt = -k*cL/NL*N5 + k*cL/NL*N4 - p5*N5 + p6*N6
    dN6dt = k*cL/NL*N5 - p6*N6
    #dCt = k*cL/NL*(Nt - N6) - (p1*N1 + p2*N2 + p3*N3 + p4*N4 + p5*N5 + p6*N6)
    dCt = dN1dt + 2*dN2dt + 3*dN3dt + 4*dN4dt + 5*dN5dt + 6*dN6dt
    #dCt = k*cL/NL*(Nt - Ct) - (p1*N1 + p2*N2 + p3*N3 + p4*N4 + p5*N5 + p6*N6)
    dCLdt = 0.
    dTdt = 5.1
    return [dTdt, dCLdt, dN0dt, dN1dt, dN2dt, dN3dt, dN4dt, dN5dt, dN6dt, dCt]

# solve ode python MacNabb-------------------

# Step-1 : loading ---------------------
CL0 = 5e12
CT0 = 0.0
T0 = 300.
C_0 = [T0, CL0, CT0]
time1 = np.linspace(0, 1000, 1000)
sol = odeint(MacNabb1, C_0, t=time1)
T_sol1 = sol.T[0]
CL_macNabb1 = sol.T[1]
CT_macNabb1 = sol.T[2]

C_0 = [T0, CL0, 3.62e-3, 0., 0., 0., 0., 0., 0., 0.]
sol = odeint(Hodille1, C_0, t=time1)
N0_Hodille = sol.T[2]
N1_Hodille = sol.T[3]
N2_Hodille = sol.T[4]
N3_Hodille = sol.T[5]
N4_Hodille = sol.T[6]
N5_Hodille = sol.T[7]
N6_Hodille = sol.T[8]
Ct_Hodille1 = sol.T[9]

plt.figure(1)
plt.plot(time1, CT_macNabb1, label='E Cont')
plt.plot(time1, Ct_Hodille1, linestyle='--', label='E discont', color='r')
plt.legend(loc='lower center', ncol = 2)

plt.figure(2)
plt.plot(time1, T_sol1, label = 'Loading')

# Step-2 : Resting ---------------------
CL0 = 0
CT0 = CT_macNabb1[-1]
T0 = 300.
C_0 = [T0, CL0, CT0]
time2 = np.linspace(1000.5, 2332, 1000)
sol = odeint(MacNabb2, C_0, t=time2)
T_sol2 = sol.T[0]
CL_macNabb2 = sol.T[1]
CT_macNabb2 = sol.T[2]

C_0 = [T0, CL0, N0_Hodille[-1], N1_Hodille[-1], N2_Hodille[-1], N3_Hodille[-1], N4_Hodille[-1], N5_Hodille[-1], N6_Hodille[-1] , Ct_Hodille1[-1]]
sol = odeint(Hodille2, C_0, t=time2)
N0_Hodille = sol.T[2]
N1_Hodille = sol.T[3]
N2_Hodille = sol.T[4]
N3_Hodille = sol.T[5]
N4_Hodille = sol.T[6]
N5_Hodille = sol.T[7]
N6_Hodille = sol.T[8]
Ct_Hodille2 = sol.T[9]

plt.figure(1)
plt.plot(time2, CT_macNabb2)
plt.plot(time2, Ct_Hodille2, linestyle='--', color='r')

plt.figure(2)
plt.plot(time2, T_sol2, label='Resting')



# Step-3 : Heating ---------------------
CL0 = 0
CT0 = CT_macNabb2[-1]
T0 = 300.
C_0 = [T0, CL0, CT0]
time3 = np.linspace(2332.5, 2430., 300)
sol = odeint(MacNabb3, C_0, t=time3)
T_sol3 = sol.T[0]
CL_macNabb3 = sol.T[1]
CT_macNabb3 = sol.T[2]

C_0 = [T0, CL0, N0_Hodille[-1], N1_Hodille[-1], N2_Hodille[-1], N3_Hodille[-1], N4_Hodille[-1], N5_Hodille[-1], N6_Hodille[-1] , Ct_Hodille2[-1]]
sol = odeint(Hodille3, C_0, t=time3)
Ct_Hodille3 = sol.T[9]

plt.figure(1)
plt.plot(time3, CT_macNabb3)
plt.plot(time3, Ct_Hodille3, linestyle='--', color='r')

plt.figure(2)
plt.plot(time3, T_sol3, label='Heating')
plt.legend()


time = np.linspace(0,1098,1098)
time = time1.tolist() + time2.tolist() + time3.tolist()
T = T_sol1.tolist() + T_sol2.tolist() + T_sol3.tolist()
CT_macNabb = CT_macNabb1.tolist()+ CT_macNabb2.tolist() + CT_macNabb3.tolist()
Ct_Hodille = Ct_Hodille1.tolist()+ Ct_Hodille2.tolist() + Ct_Hodille3.tolist()

# TDS ------------------------------------------

tm = np.zeros(len(time)-1)
Des1 = np.zeros(len(tm))
Des2 = np.zeros(len(tm))

for i in range(len(tm)):
    tm[i] = (T[i+1] + T[i])/2
    Des1[i] = (Ct_Hodille[i+1] - Ct_Hodille[i])/(time[i+1] - time[i])
    Des2[i] = (CT_macNabb[i+1] - CT_macNabb[i])/(time[i+1] - time[i])

plt.figure(5)
plt.plot(tm[2000:], abs(Des1[2000:]), label= 'E Discont')
plt.plot(tm[2000:], abs(Des2[2000:]), label= 'E Cont')
plt.xlabel('Temperature [K]')
plt.ylabel('Desorption rate [at.%.m/s]')
plt.xlim([290., 550])
#plt.ylim([-7e-5, 1.4e-3])
plt.legend(loc = 'upper left')

#plt.plot(tm, Des2)





















