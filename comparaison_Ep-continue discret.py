import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

E_constant = False
TDS = True
Lam = 30

def MacNabb(C, t):
    T, cL, CT = C
    LatticeP = 3.16E-7 * 3.16E-7
    NT = 6*3.62e-3
    NL = 6.32e19
    xev = 1.602176634e-19*6.022e23
    thetaT = CT/NT
    p0 = 5/6
    #phi =  1/(1+np.exp(-Lam*(thetaT-p0)))
    phi = 1
    xEpi = -0.45*thetaT**2 - 0.0476*thetaT + 1.374
    xEpi = xEpi + 0.2
    k = (1.38e-1/LatticeP)*np.exp(-0.2*xev/(8.31*T))
    p = 1e13*np.exp(-xEpi*xev/(8.31*T))
    if E_constant :
        p = 1e13*np.exp(-1.51*xev/(8.31*T))
    #dCTdt = k/NL*cL/6*(NT - (6*CT - 5*NT)*phi) - p*CT
    dCTdt = k/NL*cL/6*(NT - CT*phi) - p*CT
    dCLdt = 0.
    dTdt = 0.
    if TDS:
        dTdt = 1e-2
    return [dTdt, dCLdt, dCTdt]

def MacNabb2(C, t):
    T, cL, CT = C
    LatticeP = 3.16E-7 * 3.16E-7
    NT = 6*3.62e-3
    NL = 6.32e19
    xev = 1.602176634e-19*6.022e23
    thetaT = CT/NT
    p0 = 5/6
    phi =  1/(1+np.exp(-Lam*(thetaT-p0)))
    xEpi = -0.45*thetaT**2 - 0.0476*thetaT + 1.374
    xEpi = xEpi + 0.2
    k = (1.38e-1/LatticeP)*np.exp(-0.2*xev/(8.31*T))
    p = 1e13*np.exp(-xEpi*xev/(8.31*T))
    if E_constant :
        p = 1e13*np.exp(-1.51*xev/(8.31*T))
    #dCTdt = k/NL*cL/6*(NT - (6*CT - 5*NT)*phi) - p*CT
    dCTdt = k/NL*cL/6*(NT - CT*phi) - p*CT
    dCLdt = 0.
    dTdt = 0.
    if TDS:
        dTdt = 1e-2
    return [dTdt, dCLdt, dCTdt]

def Hodille(C, t):
    T, cL, N0, N1, N2, N3, N4, N5, N6, Ct  = C
    LatticeP = 3.16E-7 * 3.16E-7
    NL = 6.32e19
    Nt = 3.62e-3
    xev = 1.602176634e-19*6.022e23
    k = (1.38e-1/LatticeP)*np.exp(-0.2*xev/(8.31*T))
    p1 = 1e13*np.exp(-1.51*xev/(8.31*T))
    p2 = 1e13*np.exp(-1.50*xev/(8.31*T))
    p3 = 1e13*np.exp(-1.39*xev/(8.31*T))
    p4 = 1e13*np.exp(-1.37*xev/(8.31*T))
    p5 = 1e13*np.exp(-1.26*xev/(8.31*T))
    p6 = 1e13*np.exp(-1.05*xev/(8.31*T))
    if E_constant:
        p1 = 1e13*np.exp(-1.51*xev/(8.31*T))
        p2 = 1e13*np.exp(-1.51*xev/(8.31*T))
        p3 = 1e13*np.exp(-1.51*xev/(8.31*T))
        p4 = 1e13*np.exp(-1.51*xev/(8.31*T))
        p5 = 1e13*np.exp(-1.51*xev/(8.31*T))
        p6 = 1e13*np.exp(-1.51*xev/(8.31*T))
    dN0dt = -k*cL/NL*N0 + p1*N1
    dN1dt = -k*cL/NL*N1 + k*cL/NL*N0 - p1*N1 + p2*N2
    dN2dt = -k*cL/NL*N2 + k*cL/NL*N1 - p2*N2 + p3*N3
    dN3dt = -k*cL/NL*N3 + k*cL/NL*N2 - p3*N3 + p4*N4
    dN4dt = -k*cL/NL*N4 + k*cL/NL*N3 - p4*N4 + p5*N5
    dN5dt = -k*cL/NL*N5 + k*cL/NL*N4 - p5*N5 + p6*N6
    dN6dt = k*cL/NL*N5 - p6*N6
    dCt = k*cL/NL*(Nt - N6) - (p1*N1 + p2*N2 + p3*N3 + p4*N4 + p5*N5 + p6*N6)
    dCLdt = 0.
    dTdt = 0.
    if TDS:
        dTdt = 1e-2
    return [dTdt, dCLdt, dN0dt, dN1dt, dN2dt, dN3dt, dN4dt, dN5dt, dN6dt, dCt]

# solve ode python MacNabb-------------------
CL0 = 6e9
CT0 = 0
T0 = 300.
C_0 = [T0, CL0, CT0]
time = np.linspace(0, 50000, 800)
sol = odeint(MacNabb, C_0, t=time)
T_sol = sol.T[0]
CL_macNabb = sol.T[1]
CT_macNabb = sol.T[2]

sol2 = odeint(MacNabb2, C_0, t=time)
T_sol2 = sol2.T[0]
CL_macNabb2 = sol2.T[1]
CT_macNabb2 = sol2.T[2]



# solve ode python Hodille -------------------
CT10 = 3.62e-3
C_0 = [T0, CL0, CT10, 0., 0., 0., 0., 0., 0., 0.]
sol = odeint(Hodille, C_0, t=time)
CL_Hodille = sol.T[1]
N0_Hodille = sol.T[2]
N1_Hodille = sol.T[3]
N2_Hodille = sol.T[4]
N3_Hodille = sol.T[5]
N4_Hodille = sol.T[6]
N5_Hodille = sol.T[7]
N6_Hodille = sol.T[8]
Ct_Hodille = sol.T[9]
CT_Hodille = N1_Hodille + N2_Hodille*2 + N3_Hodille*3 + N4_Hodille*4 + N5_Hodille*5 + N6_Hodille*6

NT = 6*3.62e-3
thetaT_H = CT_Hodille/NT
thetaT = CT_macNabb/NT
thetaT2 = CT_macNabb/NT

# visualisation ------------------------
plt.figure(1)
if TDS == False:
    T_sol = time
plt.plot(T_sol,N1_Hodille, label = 'CT1')
plt.plot(T_sol,N2_Hodille*2, label = 'CT2')
plt.plot(T_sol,N3_Hodille*3, label = 'CT3')
plt.plot(T_sol,N4_Hodille*4, label = 'CT4')
plt.plot(T_sol,N5_Hodille*5, label = 'CT5')
plt.plot(T_sol,N6_Hodille*6, label = 'CT6')
plt.plot(T_sol,Ct_Hodille, label = 'CT : Sequential',color='k')
plt.plot(T_sol,CT_macNabb, label = 'CT : Simultaneous',color='k', linestyle = '--')
plt.plot(T_sol,CT_macNabb2, label = 'CT : Simultaneous (with $\\varphi$)', color='k', linestyle = '-.')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol = 5)

plt.xlabel('Temperature [K]')
if TDS == False:
    plt.xlabel('Time [s]')
plt.ylabel('$\\theta_{V_1}$')
plt.grid()
plt.savefig("ECont_vs_EDiscont.png",bbox_inches='tight',dpi=500)

plt.figure(2)
plt.plot(time,thetaT_H, label = 'E discontinued')
plt.plot(time,thetaT, label = 'E continued', linestyle = '--')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('$\\theta_T$ [.]')
plt.grid()



p0 = 5/6
phi = 1/(1+np.exp(-Lam*(thetaT-p0)))
DeltaCT1 = NT/6 - N6_Hodille
DeltaCT2 = NT - (6*CT_macNabb - 5*NT)*phi
DeltaCT2 = NT - CT_macNabb*phi

plt.figure(3)
plt.plot(time,DeltaCT1, label = '$\\Delta$CT_Hodille')
plt.plot(time,DeltaCT2/6, label = '$\\Delta$CT_MacNabb')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('$\\Delta C_T$')
plt.grid()


plt.figure(4)
theta4 = np.linspace(0, 1, 100)
phi4 = 1/(1+np.exp(-Lam*(theta4 -p0)))
plt.plot(theta4 ,phi4, label = 'with $\\lambda$=30')
plt.legend()
plt.xlabel('$\\theta_{V_1}$')
plt.ylabel('$\\varphi$')
plt.grid()
plt.savefig("lambda.png",bbox_inches='tight',dpi=500)


























