import scipy.special as sp
import scipy.integrate as integ

from scipy.integrate import solve_ivp
from astropy.modeling.blackbody import blackbody_nu

from global_initCond import *

# CONSTANTS [CGS]

boltzmann = const.k * 1.0e7
cLight = const.speed_of_light * 1.0e2
cLight2 = cLight*cLight
m_e = const.electron_mass*1.0e3
m_p = const.proton_mass*1.0e3
mu = const.physical_constants['atomic mass constant'][0]*1.0e3
G = const.gravitational_constant * 1.0e3
thomson = const.physical_constants['Thomson cross section'][0]*1.0e4
planck = const.h*1.0e7
eCharge = 4.80320425e-10
re = const.physical_constants['classical electron radius'][0]*1.0e2
solarMass = 1.98847e33
eMMW = 1.14
iMMW = 1.23

# ADAF EQS

schwRadius = 2.0*G*blackHoleMass*solarMass / cLight2

def a_aux(x):
    k1 = sp.kn(1,x) 
    k2 = sp.kn(2,x)
    k3 = sp.kn(3,x)
    result = np.where( k2 > 0.0 , x*(0.25*(3.0*k3+k1)/k2-1.0) , 1.5053 )
    return result

def da_aux(x):
    k0 = sp.kn(0,x)
    k1 = sp.kn(1,x)
    k2 = sp.kn(2,x)
    k3 = sp.kn(3,x)
    k4 = sp.kn(4,x)
    
    aux = 0.125*( 3.0*k4+4.0*k2+k0 )
    result_aux = a_aux(x)/x - x*( aux - 0.5*(k1+k3)*(a_aux(x)/x+1.0))/k2
    return np.where(k2 > 0.0, result_aux, a_aux(x)/x )
    
def aux_der(x):
    return - da_aux(1.0/x)/ (x*x)

def keplAngMom(rnorm):
    return np.power(rnorm,1.5)/(rnorm-1.0) / np.sqrt(2.0)                 # [rS c]

def keplAngVel(r):
    return np.sqrt(G*blackHoleMass*solarMass/r) / (r-schwRadius)

def sqrdSoundVel(temp_i,temp_e):
    return boltzmann/(beta*mu) * ( temp_i/iMMW + temp_e/eMMW )

def height(r,temp_i,temp_e):
    return np.sqrt(sqrdSoundVel(temp_i,temp_e))/keplAngVel(r)

def massDensity(r,temp_i,temp_e,radialVel):
    return accRateOut*np.power(r/schwRadius/rOut,s) / (4.0*np.pi*r*height(r,temp_i,temp_e)*(-radialVel))

def magField(r,temp_i,temp_e,radialVel):
    return np.sqrt((1.0-beta)*massDensity(r,temp_i,temp_e,radialVel)*sqrdSoundVel(temp_i,temp_e)*8.0*np.pi)

def iDens(r,temp_i,temp_e,radialVel):
    return massDensity(r,temp_i,temp_e,radialVel)/(iMMW*mu)

def eDens(r,temp_i,temp_e,radialVel):
    return massDensity(r,temp_i,temp_e,radialVel)/(eMMW*mu)
   
def qie(r,temp_i,temp_e,v):
    lnLambda = 20.0
    theta_e = boltzmann*temp_e/(m_e*cLight2)
    theta_i = boltzmann*temp_i/(m_p*cLight2)
    xe = 1.0/theta_e
    xi = 1.0/theta_i
    xei = xe+xi
        
    k2i = sp.kn(2,xi)
    k2e = sp.kn(2,xe)
    k1ei = sp.kn(1,xei)
    k0ei = sp.kn(0,xei)
    
    sumtheta = theta_i + theta_e
    aux1 = 1.875*thomson*(m_e/m_p)*cLight *lnLambda *eDens(r,temp_i,temp_e,v)*iDens(r,temp_i,temp_e,v) *\
            boltzmann*(temp_i-temp_e)
    aux2 = (2.0*sumtheta*sumtheta+1.0)/sumtheta
    return np.where(xei > 300.0 , 
                    np.where(xi > 150.0 , 
                             np.where(xe > 150.0 , 
                                      aux1*np.sqrt(2.0*xei/(np.pi*xe*xi))*(aux2+2.0) , 
                                      aux1*np.sqrt(xi/xei)*(aux2+2.0)*np.exp(-xe)/k2e) , 
                             aux1*np.sqrt(xe/xei)*(aux2+2.0)*np.exp(-xi)/k2i),
                    aux1*(aux2 * k1ei/k2i + 2.0*k0ei/k2i)/k2e)

def qie2(r,temp_i,temp_e,v):
    theta_e = boltzmann*temp_e / (m_e*cLight2)
    theta_i = boltzmann*temp_i / (m_p*cLight2) 
    lnLambda = 20.0
    diftemps = (boltzmann*temp_i - boltzmann*temp_e)
    aux = 1.5*m_e/m_p * eDens(r,temp_i,temp_e,v)*iDens(r,temp_i,temp_e,v)*thomson*cLight*lnLambda*diftemps
    aux2 = (np.sqrt(2.0/np.pi)+np.sqrt(theta_e+theta_i))/np.power(theta_e+theta_i,1.5)
    return aux*aux2

def Qie3(r,temp_i,temp_e,v):
    theta_e = boltzmann*temp_e / (m_e*cLight2)
    theta_i = boltzmann*temp_i / (m_p*cLight2)
    lnLambda= 20.0
    FactorTemps = (temp_i-temp_e)/(eMMW*temp_i+iMMW*temp_e)
    aux = 1.5 / 1836.15 * massDensity(r,temp_i,temp_e,v)/mu * thomson * lnLambda * FactorTemps
    aux2 = (np.sqrt(2.0/np.pi)+np.sqrt(theta_e+theta_i))/np.power(theta_e+theta_i,1.5)
    
    return aux*aux2     # [s^-1 cm^-1]

def gaunt(nu,temp_e):
    aux = boltzmann*temp_e/(planck*nu)
    zeda = 0.57695
    return np.where(aux < 1.0,np.sqrt(3.0/np.pi * aux), np.sqrt(3)/np.pi*np.log(4.0/zeda * aux))

def qei(r,temp_i,temp_e,v):
    theta = boltzmann*temp_e/(m_e*cLight2)
    ne = eDens(r,temp_i,temp_e,v)
    Fei = np.where(theta<1.0, 
                   4.0 * np.sqrt(2.0*theta/(np.pi*np.pi*np.pi)) * 
                   (1.0+1.781*np.power(theta,1.34)),
                   4.5*theta/np.pi * (np.log(1.123*theta+0.48)+1.5))
    
    return 1.25*ne*ne*thomson*cLight*const.alpha*m_e*cLight2*Fei

def qee(r,temp_i,temp_e,v):
    theta = boltzmann*temp_e/(m_e*cLight2)
    ne = eDens(r,temp_i,temp_e,v)
    return ne*ne*cLight*re*re*const.alpha*m_e*cLight2 * \
            np.where(theta<1.0, 20.0/(9.0*np.sqrt(np.pi)) * (44.0-3.0*np.pi*np.pi) * \
                     np.power(theta,1.5)*(1.0+1.1*theta+theta*theta-
                                            1.25*np.power(theta,2.5)),
                     24.0*theta*(np.log(1.1232*theta)+1.28))

def xiBremss(nu,r,temp_i,temp_e,v):
    qBremss = qee(r,temp_i,temp_e,v)+qei(r,temp_i,temp_e,v)
    return qBremss*np.exp(-planck*nu/(boltzmann*temp_e))*(planck/(boltzmann*temp_e))*gaunt(nu,temp_e)

def Iprim(x):
    return 4.0505/np.power(x,1.0/6.0) * (1.0+0.4/np.power(x,0.25)+0.5316/np.sqrt(x)) * \
            np.exp(-1.8899*np.power(x,1.0/3.0))

def xiSync(nu,r,temp_i,temp_e,v):
    theta = boltzmann*temp_e/(m_e*cLight2)
    nu0 = eCharge*magField(r,temp_i,temp_e,v)/(2.0*np.pi*m_e*cLight)
    xM = 2.0*nu/(3.0*nu0*theta*theta)
    return np.where(temp_e>1.0e8,4.43e-30*4.0*np.pi*eDens(r,temp_i,temp_e,v)*nu / sp.kn(2,1.0/theta) * \
            Iprim(xM),0.0)

def eta(nu,r,temp_i,temp_e,v):
    theta = boltzmann*temp_e/(m_e*cLight2)
    #tau_eff = tau(nu,r,temp_i,temp_e)*np.sqrt(1.0+eDens(r,temp_i,temp_e)*thomson/kappa(nu,r,temp_i,temp_e))
    #tau_es = 2.0*eDens(r,temp_i,temp_e)*thomson*height(r,temp_i,temp_e)*np.where(1.0>1.0/tau_eff,1.0,1.0/tau_eff)
    tau_es = 2.0*eDens(r,temp_i,temp_e,v)*thomson*height(r,temp_i,temp_e)
    s_exp = tau_es*(tau_es+1.0)
    A = 1.0+4.0*theta+16.0*theta*theta
    etaMax = 3.0*boltzmann*temp_e/(planck*nu)
    jm = np.log(etaMax)/np.log(A)
    
    gamma1 = sp.gammainc(jm+1.0,A*s_exp)
    gamma2 = sp.gammainc(jm+1.0,s_exp)
    aux2 = s_exp*(A-1.0)
    
    result = np.where(aux2<200.0,np.exp(aux2)*(1.0-gamma1)+etaMax*gamma2,etaMax*gamma2)
    return result

def eta2(nu,r,tempi,tempe,v):
    
    theta = boltzmann*tempe/(m_e*cLight2)
    tau_abs = tau(nu,r,tempi,tempe,v)
    ne = eDens(r,tempi,tempe,v)
    h = height(r,tempi,tempe)
    tau_es1 = ne*thomson*h
    tau_eff = tau_abs*np.sqrt(1.0 + ne*thomson/kappa(nu,r,tempi,tempe,v))
    #tau_es2 = 2.0*ne*thomson*h*np.min(1.0,1.0/tau_eff)
    tau_es2 = tau_es1
    P = 1.0-np.exp(-tau_es2)
    x = planck*nu/(m_e*cLight2)
    A = 1.0 + 4.0*theta+16.0*theta*theta
    kap = P*(A-1.0)/(1.0-P*A)
    phi = -np.log(P)/np.log(A)
    x = x/(3.0*theta)
    result = 1.0 + kap*(1.0-np.power(x,phi-1.0))
    
    return np.where(result>1.0,result,1.0)

def kappa(nu,r,temp_i,temp_e,v):
    bnu = 4.0*np.pi*blackbody_nu(nu,temp_e).value
    xi = xiBremss(nu,r,temp_i,temp_e,v)+xiSync(nu,r,temp_i,temp_e,v)
    return np.where(bnu > 1.0e-2*xi*height(r,temp_i,temp_e),xi/bnu,50.0)

def tau(nu,r,temp_i,temp_e,v):
    return np.sqrt(np.pi)/2.0 * kappa(nu,r,temp_i,temp_e,v)*height(r,temp_i,temp_e)

def flux(nu,r,temp_i,temp_e,v):
    bnu = blackbody_nu(nu,temp_e).value
    return 2.0*np.pi/np.sqrt(3.0) * bnu * (1.0-np.exp(-2.0*np.sqrt(3.0)*tau(nu,r,temp_i,temp_e,v)))

def integrand(nu,r,temp_i,temp_e,v):
    return flux(nu,r,temp_i,temp_e,v)*eta2(nu,r,temp_i,temp_e,v)

def qminus(r,temp_i,temp_e,v):
    result = integ.quad(integrand,0.0,np.infty,args=(r,temp_i,temp_e,v))[0] / height(r,temp_i,temp_e)
    return result

def qminus2(r,temp_i,temp_e,v):
    nuMin = 1.0e6
    nuMax = 1.0e21
    iter = 20
    paso = np.power(nuMax/nuMin,1.0/float(iter-1))
    sum = 0.0
    nu = nuMin
    for i in range(iter):
        dnu = nu*(np.sqrt(paso)-1.0/np.sqrt(paso))
        sum += integrand(nu,r,temp_i,temp_e,v)*dnu
        nu = nu*paso
        
    return sum / height(r,temp_i,temp_e)

def Den_r(logr,logTi,logTe,logv,j): 
    
    Ti = np.exp(logTi)
    TiR = Ti*iMMW
    Te = np.exp(logTe)
    TeR = Te*eMMW
    te = boltzmann*TeR/(m_e*cLight2)
    v  = -np.exp(logv)
    
    cs2 = sqrdSoundVel(TiR,TeR)
    
    vnorm = v/cLight
    cs2norm = cs2/cLight2
    
    machNumber2 = vnorm*vnorm / cs2norm
    invM2 = alpha*alpha/machNumber2
    invM2e = delta*invM2
    invM2i = (1.0-delta)*invM2
    Qvcs = machNumber2-1.0
    
    etai = Ti/(Ti+Te)
    etae = 1.0-etai

    aaux = a_aux(1.0/te) + te*aux_der(te)
    
    A = etai/2.0
    B = etae/2.0
    C = Qvcs
    E = etai*(0.5*beta*(3.0+etai)-invM2i)
    F = etae*(beta*0.5*etai-invM2i)
    G = beta*etai+invM2i
    I = etai*(0.5*beta*etae-invM2e)
    J = etae*(beta*(aaux+0.5*etae)-invM2e)
    K = beta*etae+invM2e
    
    return A*F*K - A*G*J + C*E*J - C*F*I + B*G*I - B*E*K

def Num1_r(logr,logTi,logTe,logv,j):
    
    Ti = np.exp(logTi)
    TiR = Ti*iMMW
    Te = np.exp(logTe)
    TeR = Te*eMMW
    te = boltzmann*TeR/(m_e*cLight2)
    v  = -np.exp(logv)
    
    rnorm = np.exp(logr)
    r = schwRadius*rnorm
    
    rho = massDensity(r,TiR,TeR,v)
    cs2 = sqrdSoundVel(TiR,TeR)

    vnorm = v/cLight
    cs2norm = cs2/cLight2
    
    l = -alpha * cs2norm/vnorm * rnorm + j
    lK = keplAngMom(rnorm)
    machNumber2 = vnorm*vnorm / cs2norm
    invM2 = alpha*alpha/machNumber2
    invM2e = delta*invM2
    invM2i = (1.0-delta)*invM2
    Qvcs = machNumber2-1.0
    f1 = 1.5-s + 1.0/(1-1.0/rnorm)
    f2 = (l*l-lK*lK) / (rnorm*rnorm*cs2norm)
    f = f1+f2
    
    Pgas = beta*rho*cs2
    
    etai = Ti/(Ti+Te)
    etae = 1.0-etai
    
    Factor1 = r / (Pgas*v)
    Qie = qie2(r,TiR,TeR,v) * Factor1
    Qmin = qminus2(r,TiR,TeR,v) * Factor1
    
    Factor2 = 2.0*alpha*(j/(vnorm*rnorm))
    aaux = a_aux(1.0/te) + te*aux_der(te)
    
    B = etae/2.0
    C = Qvcs
    D = f
    F = etae*(beta*0.5*etai-invM2i)
    G = beta*etai+invM2i
    H = Factor2*(1.0-delta)-invM2i-beta*(etai*f1+Qie)
    J = etae*(beta*(aaux+0.5*etae)-invM2e)
    K = beta*etae+invM2e
    L = Factor2*delta-invM2e - beta*(etae*f1-Qie+Qmin)
    
    return B*G*L - B*H*K + F*D*K - F*C*L + J*C*H - J*D*G

def rhs_r(logr,y):
    
    logTi,logTe,logv,j = y
    
    Ti = np.exp(logTi)
    TiR = Ti*iMMW
    ti = boltzmann*TiR/(m_p*cLight2)
    Te = np.exp(logTe)
    TeR = Te*eMMW
    te = boltzmann*TeR/(m_e*cLight2)
    v  = -np.exp(logv)
    rnorm = np.exp(logr)
    r = schwRadius*rnorm
    
    rho = massDensity(r,TiR,TeR,v)
    cs2 = sqrdSoundVel(TiR,TeR)
    
    vnorm = v/cLight
    cs2norm = cs2/cLight2
    
    l = -alpha * cs2norm/vnorm * rnorm + j
    lK = np.power(rnorm,1.5)/(rnorm-1.0)/np.sqrt(2.0)
    
    machNumber2 = vnorm*vnorm / cs2norm
    invM2 = alpha*alpha/machNumber2
    invM2e = delta*invM2
    invM2i = (1.0-delta)*invM2
    Qvcs = machNumber2-1.0
    f1 = 1.5-s + 1.0/(1-1.0/rnorm)
    f2 = (l*l-lK*lK) / (rnorm*rnorm*cs2norm)
    f = f1+f2
    
    Pgas = beta*rho*cs2
    
    etai = Ti/(Ti+Te)
    etae = 1.0-etai
    
    Factor1 = r / (Pgas*v)
    Qie = qie2(r,TiR,TeR,v) * Factor1
    Qmin = qminus2(r,TiR,TeR,v) * Factor1
    
    Factor2 = 2.0*alpha*(j/(vnorm*rnorm))
    #ae = a_aux(1.0/te) + te*aux_der(te)
    ae = 3.0-6.0/(4.0+5.0*te) + te*30.0/np.square(4.0+5.0*te)
    ai = 3.0-6.0/(4.0+5.0*ti) + ti * 30.0/np.square(4.0+5.0*ti)
    #ai = 1.5
    
    A = etai/2.0
    B = etae/2.0
    C = Qvcs
    D = f
    E = etai*(beta*(ai+0.5*etai)-invM2i)
    F = etae*(beta*0.5*etai-invM2i)
    G = beta*etai+invM2i
    H = Factor2*(1.0-delta)-invM2i-beta*(etai*f1+Qie)
    I = etai*(beta*0.5*etae-invM2e)
    J = etae*(beta*(ae+0.5*etae)-invM2e)
    K = beta*etae+invM2e
    L = Factor2*delta-invM2e - beta*(etae*f1-Qie+Qmin)
    
    DEN = A*F*K - A*G*J + C*E*J - C*F*I + B*G*I - B*E*K
    N1 = B*G*L - B*H*K + F*D*K - F*C*L + J*C*H - J*D*G
    N2 = D*G*I - D*E*K + C*E*L - C*H*I + A*H*K - A*G*L
    N3 = D*E*J - D*F*I + B*H*I - B*E*L + A*F*L - A*H*J
    
    rhs_1 = N1/DEN
    rhs_2 = N2/DEN
    rhs_3 = N3/DEN
    rhs_4 = 0.0
    
    #print(np.log10(r/schwRadius),N1,DEN)
    return np.vstack((rhs_1,rhs_2,rhs_3,rhs_4))

def rhs_w_print(logr,y):
    
    logTi,logTe,logv,j = y
    
    Ti = np.exp(logTi)
    TiR = Ti*iMMW
    ti = boltzmann*TiR/(m_p*cLight2)
    Te = np.exp(logTe)
    TeR = Te*eMMW
    te = boltzmann*TeR/(m_e*cLight2)
    v  = -np.exp(logv)
    rnorm = np.exp(logr)
    r = schwRadius*rnorm
    
    rho = massDensity(r,TiR,TeR,v)
    cs2 = sqrdSoundVel(TiR,TeR)
    
    vnorm = v/cLight
    cs2norm = cs2/cLight2
    
    l = -alpha * cs2norm/vnorm * rnorm + j
    lK = np.power(rnorm,1.5)/(rnorm-1.0)/np.sqrt(2.0)
    
    machNumber2 = vnorm*vnorm / cs2norm
    invM2 = alpha*alpha/machNumber2
    invM2e = delta*invM2
    invM2i = (1.0-delta)*invM2
    Qvcs = machNumber2-1.0
    f1 = 1.5-s + 1.0/(1-1.0/rnorm)
    f2 = (l*l-lK*lK) / (rnorm*rnorm*cs2norm)
    f = f1+f2
    
    Pgas = beta*rho*cs2
    
    etai = Ti/(Ti+Te)
    etae = 1.0-etai
    
    Factor1 = r / (Pgas*v)
    Qie = qie2(r,TiR,TeR,v) * Factor1
    Qmin = qminus2(r,TiR,TeR,v) * Factor1
    
    Factor2 = 2.0*alpha*(j/(vnorm*rnorm))
    #ae = a_aux(1.0/te) + te*aux_der(te)
    ae = 3.0-6.0/(4.0+5.0*te) + te*30.0/np.square(4.0+5.0*te)
    ai = 3.0-6.0/(4.0+5.0*ti) + ti * 30.0/np.square(4.0+5.0*ti)
    #ai = 1.5
    
    A = etai/2.0
    B = etae/2.0
    C = Qvcs
    D = f
    E = etai*(beta*(ai+0.5*etai)-invM2i)
    F = etae*(beta*0.5*etai-invM2i)
    G = beta*etai+invM2i
    H = Factor2*(1.0-delta)-invM2i-beta*(etai*f1+Qie)
    I = etai*(beta*0.5*etae-invM2e)
    J = etae*(beta*(ae+0.5*etae)-invM2e)
    K = beta*etae+invM2e
    L = Factor2*delta-invM2e - beta*(etae*f1-Qie+Qmin)
    
    DEN = A*F*K - A*G*J + C*E*J - C*F*I + B*G*I - B*E*K
    N1 = B*G*L - B*H*K + F*D*K - F*C*L + J*C*H - J*D*G
    N2 = D*G*I - D*E*K + C*E*L - C*H*I + A*H*K - A*G*L
    N3 = D*E*J - D*F*I + B*H*I - B*E*L + A*F*L - A*H*J
    
    rhs_1 = N1/DEN
    rhs_2 = N2/DEN
    rhs_3 = N3/DEN
    rhs_4 = 0.0
    
    print(np.log10(r/schwRadius),N1,DEN)
    return np.vstack((rhs_1,rhs_2,rhs_3,rhs_4))


def rhs_r_new(logr,y):
    
    logTi,logTe,logv,l_in = y
    
    Ti = np.exp(logTi)
    TiR = Ti*iMMW
    ti = boltzmann*TiR/(m_p*cLight2)
    Te = np.exp(logTe)
    TeR = Te*eMMW
    te = boltzmann*TeR/(m_e*cLight2)
    v  = -np.exp(logv)
    r_norm = np.exp(logr)
    r = schwRadius*r_norm
    
    rho = massDensity(r,TiR,TeR,v)
    cs2 = sqrdSoundVel(TiR,TeR)
    
    v_norm = v/cLight
    cs2_norm = cs2/cLight2
    
    lamda_r = r_norm/(r_norm-1.0)
    lamda_r_plus = lamda_r + 0.5
    Lamda = (lamda_r*(lamda_r-1.0)-1.0)/(lamda_r_plus)
    
    l_norm = l_in + lamda_r_plus*alpha*cs2_norm*r_norm/(-v_norm)
    l_norm = l_in + alpha*cs2_norm*r_norm/(-v_norm)
    lK_norm = np.power(r_norm,1.5)/(np.sqrt(2.0)*(r_norm-1.0))
    
    M2 = v_norm*v_norm / cs2_norm
    M2_a = M2/(alpha*alpha*lamda_r_plus*lamda_r_plus)
    #M2_a = M2/alpha/alpha
    
    p = rho*cs2
    
    etai = Ti/(Ti+Te)
    etae = 1.0-etai
    
    Factor = M2_a * r / (p*v)
    Qie = qie2(r,TiR,TeR,v) * Factor
    Qmin = qminus2(r,TiR,TeR,v) * Factor
    
    #ae = a_aux(1.0/te) + te*aux_der(te)
    ae = 3.0-6.0/(4.0+5.0*te) + te*30.0/np.square(4.0+5.0*te)
    ai = 3.0-6.0/(4.0+5.0*ti) + ti * 30.0/np.square(4.0+5.0*ti)
    
    f1 = 2.0*l_in/(l_norm-l_in)
    f2 = beta*M2_a*(lamda_r+1.5-s)
    
    A = etai/2.0
    B = etae/2.0
    C = M2-1.0
    
    D = (l_norm*l_norm-lK_norm*lK_norm)/(r_norm*r_norm*cs2_norm) + 1.5 - s + lamda_r
    E = etai*(M2_a*beta*(ai+0.5*etai)-(1.0-delta))
    F = etae*(beta*M2_a*0.5*etai-(1.0-delta))
    G = beta*M2_a*etai+(1.0-delta)
    
    #Lamda = -1.0
    H = (1.0-delta)*(Lamda-f1) - f2*etai - Qie
    
    I = etai*(beta*0.5*etae*M2_a-delta)
    J = etae*(beta*M2_a*(ae+0.5*etae)-delta)
    K = beta*etae*M2_a+delta
    
    L = delta*(Lamda-f1) - f2*etae + Qie - Qmin
    
    DEN = A*F*K - A*G*J + C*E*J - C*F*I + B*G*I - B*E*K
    N1 = B*G*L - B*H*K + F*D*K - F*C*L + J*C*H - J*D*G
    N2 = D*G*I - D*E*K + C*E*L - C*H*I + A*H*K - A*G*L
    N3 = D*E*J - D*F*I + B*H*I - B*E*L + A*F*L - A*H*J
    
    rhs_1 = N1/DEN
    rhs_2 = N2/DEN
    rhs_3 = N3/DEN
    rhs_4 = 0.0
    
    print(np.log10(r/schwRadius),N1,DEN,l_norm,l_in)
    #print(A,B,C,D,E,F,G,H,I,J,K,L)
    
    return np.vstack((rhs_1,rhs_2,rhs_3,rhs_4))

def Den_r_new(logr,logTi,logTe,logv,l_in):
    
    Ti = np.exp(logTi)
    TiR = Ti*iMMW
    ti = boltzmann*TiR/(m_p*cLight2)
    Te = np.exp(logTe)
    TeR = Te*eMMW
    te = boltzmann*TeR/(m_e*cLight2)
    v  = -np.exp(logv)
    r_norm = np.exp(logr)
    
    cs2 = sqrdSoundVel(TiR,TeR)
    
    v_norm = v/cLight
    cs2_norm = cs2/cLight2
    
    lamda_r = r_norm/(r_norm-1.0)
    lamda_r_plus = lamda_r + 0.5
    
    M2 = v_norm*v_norm / cs2_norm
    M2_a = M2/(alpha*alpha*lamda_r_plus*lamda_r_plus)
    
    etai = Ti/(Ti+Te)
    etae = 1.0-etai
    
    ae = a_aux(1.0/te) + te*aux_der(te)
    ae = 3.0-6.0/(4.0+5.0*te) + te*30.0/np.square(4.0+5.0*te)
    ai = 3.0-6.0/(4.0+5.0*ti) + ti * 30.0/np.square(4.0+5.0*ti)
    
    A = etai/2.0
    B = etae/2.0
    C = M2-1.0
    
    E = etai*(M2_a*beta*(ai+0.5*etai)-(1.0-delta))
    F = etae*(beta*M2_a*0.5*etai-(1.0-delta))
    G = beta*M2_a*etai+(1.0-delta)
    
    I = etai*(beta*0.5*etae*M2_a-delta)
    J = etae*(beta*M2_a*(ae+0.5*etae)-delta)
    K = beta*etae*M2_a+delta
    
    return A*F*K - A*G*J + C*E*J - C*F*I + B*G*I - B*E*K

def Num1_r_new(logr,logTi,logTe,logv,l_in):
    
    Ti = np.exp(logTi)
    TiR = Ti*iMMW
    ti = boltzmann*TiR/(m_p*cLight2)
    Te = np.exp(logTe)
    TeR = Te*eMMW
    te = boltzmann*TeR/(m_e*cLight2)
    v  = -np.exp(logv)
    r_norm = np.exp(logr)
    r = schwRadius*r_norm
    
    rho = massDensity(r,TiR,TeR,v)
    cs2 = sqrdSoundVel(TiR,TeR)
    
    v_norm = v/cLight
    cs2_norm = cs2/cLight2
    
    lamda_r = r_norm/(r_norm-1.0)
    lamda_r_plus = lamda_r + 0.5
    Lamda = (lamda_r*(lamda_r-1.0)-1.0)/(lamda_r_plus)
    
    l_norm = l_in + (0.5+lamda_r)*alpha*cs2_norm*r_norm/(-v_norm)
    lK_norm = np.power(r_norm,1.5)/(np.sqrt(2.0)*(r_norm-1.0))
    
    M2 = v_norm*v_norm / cs2_norm
    M2_a = M2/(alpha*alpha*lamda_r_plus*lamda_r_plus)

    p = rho*cs2
    
    etai = Ti/(Ti+Te)
    etae = 1.0-etai
    
    Factor = M2_a * r / (p*v)
    Qie = qie2(r,TiR,TeR,v) * Factor
    Qmin = qminus2(r,TiR,TeR,v) * Factor
    
    ae = a_aux(1.0/te) + te*aux_der(te)
    ae = 3.0-6.0/(4.0+5.0*te) + te*30.0/np.square(4.0+5.0*te)
    ai = 3.0-6.0/(4.0+5.0*ti) + ti * 30.0/np.square(4.0+5.0*ti)
    
    f1 = 2.0*l_in/(l_norm-l_in)
    f2 = beta*M2_a*(lamda_r+1.5-s)
    
    B = etae/2.0
    C = M2-1.0  
    D = (l_norm*l_norm-lK_norm*lK_norm)/(r_norm*r_norm*cs2_norm) + 1.5 - s + lamda_r    
    F = etae*(beta*M2_a*0.5*etai-(1.0-delta))
    G = beta*M2_a*etai+(1.0-delta)   
    H = (1.0-delta)*(Lamda-f1) - f2*etai - Qie
    J = etae*(beta*M2_a*(ae+0.5*etae)-delta)
    K = beta*etae*M2_a+delta  
    L = delta*(Lamda-f1) - f2*etae + Qie - Qmin
    
    return B*G*L - B*H*K + F*D*K - F*C*L + J*C*H - J*D*G

def event(logr,y):
    logTi, logTe, logv, j = y
    return Den_r(logr,logTi,logTe,logv,j)
event.terminal = True

def bounds(log10j,logTiOut,logTeOut,logvOut):
    
    l_in = np.power(10.0,log10j)
    print("l_in = ",l_in)
    
    y0 = np.array([logTiOut,logTeOut,logvOut,l_in])
    solution = solve_ivp(rhs_r,(np.log(rOut),0.5),y0,method='LSODA',vectorized=False,events=event)
    
    print(solution.message)
    result = Num1_r(solution.t[-1],solution.y[0][-1],solution.y[1][-1],solution.y[2][-1],l_in)
    print("RESULT = ",result) 
    print("rSonic = ",np.exp(solution.t[-1]))
    print()
    
    return result

def bounds_new(log10j,logTiOut,logTeOut,logvOut):
    
    l_in = np.power(10.0,log10j)
    print("j = ",l_in)
    
    y0 = np.array([logTiOut,logTeOut,logvOut,l_in])
    solution = solve_ivp(rhs_r_new,(np.log(rOut),0.5),y0,vectorized=True,events=event)
    
    print(solution.message)
    result = Num1_r_new(solution.t[-1],solution.y[0][-1],solution.y[1][-1],solution.y[2][-1],l_in)
    print("RESULT = ",result) 
    return result
