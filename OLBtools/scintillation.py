from . import *

import numpy as np

import scipy.special as scsp
import scipy.optimize as scop
import scipy.integrate as scin

#----------------------------------------------------------
# Cn2 Models

def Cn2_SLC(h):
    # Submarine Laser Comunication (SLC) model, day and night, for h > 1500m
    # For some reason, the SLC model is for the atmosphere, and is based on the median values for the AIR Force Maui Optical Station 
    # Laser Beam Propagation through random Media, 2nd edition  Larry C. Andrews and Ronald L. Phillips page 481-482
    return (1500<=h)*(h<7200)*8.87e-7/h**3 + (7200<=h)*(h<20000)*2e-16/h**(1/2)

def Cn2_HV_ACTLIMB(h, H0):
    # ACTLIMB (Hufnagel-Valley 5/7 with Gurvich
    HV = 1.7e-14*np.exp(-h/100) + 2.7e-16*np.exp(-h/1500) + 3.6e-3*(1e-5*h)**10*np.exp(-h/1000)
    Nh2 = 1e-7*np.exp(-2*h/H0)
    Ck = 1e-10*10**(h/25e3)
    return HV + Nh2*Ck

def Cn2_HV_57(h):
    # Hufnagel-Valley 5/7
    HV = 1.7e-14*np.exp(-h/100) + 2.7e-16*np.exp(-h/1500) + 3.6e-3*(1e-5*h)**10*np.exp(-h/1000)
    return HV

def Cn2_HV_ACTLIMB_best(h):
    # ACTLIMB best (x0.1)
    return 0.1*Cn2_HV_ACTLIMB(h)
    
def Cn2_HV_ACTLIMB_worst(h):
    # ACTLIMB worst (x10)
    return 10*Cn2_HV_ACTLIMB(h)

def Cn2_HV_best(h):
    # HV57 best (x0.1)
    return 0.1*Cn2_HV_57(h)
    
def Cn2_HV_worst(h):
    # HV57 worst (x10)
    return 10*Cn2_HV_57(h)

def Cn2_ITU_R_P_1621_2(h, v_g=2.3, C0=1.7e-14):
    '''ITU-R P.1621-2 suggested model under 5.1.1
    h: altitude in metters
    v_g: ground wind speed in m.s-1, assume 2.3 m.s-1 by default
    C_0: Cn2 on the ground, in m-2/3, typically 1.7e-14 m-2/3'''
    v_rms = np.sqrt(v_g**2 + 3.11*v_g + 360.61)
    Cn2 = 8.148e-56 * v_rms**2 * h**10 * np.exp(-h/1000) + 2.7e-16 * np.exp(-h/1500) + C0*np.exp(-h/100)
    return Cn2
    
#----------------------------------------------------------

def _integration_range(h_0, integration_mode, n_int):
    '''h_0: ground sation altitude
    H: target altitude
    integration_mode: 'linear', 'exponential', or 'ITU'. ITU is per ITU-R P1.1621-2 in 5.1.1.
    n_int: number of integration point'''
    if integration_mode == 'linear':      h = np.linspace(h_0,np.minimum(H,20e3),n_int,axis=-1)
    if integration_mode == 'exponential': raise NotImplementedError
    if integration_mode == 'ITU':
        i = np.linspace(h/h,139,139,axis=-1)
        h_i = np.exp(i - 1 / 20e3)
        h = h_0 + np.cumsum(h_i,axis=-1)

    np.logspace(start=h_0, stop=np.log(np.minimum(H,20e3)), num=n_int, base=np.e ,axis=-1)
    
def Rytov_var(Cn2_h,k,L):
    # Laser Beam Propagation through random Media, 2nd edition  Larry C. Andrews and Ronald L. Phillips page 140
    return 1.23*Cn2_h*k**(5/7)*L**(11/6)
    
def Fried_param(zenith,k,Cn2,h_0,H,integration_mode='linear',n_int=1000):
    '''Fried's pararamter for a ground station.
    Andrews Ch12, p492, Eq23
    zenith:  zenith angle in radians
    k: angular wave number
    Cn2: function of h
    h_0: ground sation altitude
    H: target altitude
    integration_mode: 'linear', 'exponential', or 'ITU'. ITU is per ITU-R P1.1621-2 in 5.1.1.
    n_int: number of integration point'''
    
    # Integration range
    h = np.linspace(h_0,np.minimum(H,20e3),n_int,axis=-1)
    
    # Integral term
    integ = np.trapz(Cn2(h),h,axis=-1)
    
    return (0.42/np.cos(zenith)*k**2*integ)**(-3/5)

def normalized_distance_uplink(h,h_0,H):
    '''Normalized diatance, uplink case, for use in Andrews ch12
    Andrews Ch12, p490, Eq14'''
    return 1-(h-h_0)/(H-h_0)
    
def gaussian_beam_parameters_at_waist(L,k,W_0):
    G_Lambda_0 =  (2*L)/(k*W_0**2)
    G_Theta_0 = 1
    return G_Lambda_0,G_Theta_0
    
def gaussian_beam_parameters_colimated(L,k,W_0):
    G_Lambda_0,G_Theta_0 = gaussian_beam_parameters_at_waist(L,k,W_0)
    divisor = G_Lambda_0**2 + G_Theta_0**2
    G_Lambda = G_Lambda_0/divisor
    G_Theta = G_Theta_0/divisor
    return G_Lambda,G_Theta
    
def mu3u_par(G_Lambda,G_Theta,Cn2,h_0,H,n_int=1000):
    G_Lambda,G_Theta,h_0_i,H_i = np.broadcast_arrays(G_Lambda,G_Theta,h_0,H)
    
    # Integration range
    #h = np.linspace(h_0_i,H_i,n_int,axis=-1)
    h = np.linspace(h_0,np.minimum(H,20e3),n_int,axis=-1)
    
    G_Lambda = G_Lambda[..., np.newaxis]
    G_Theta = G_Theta[..., np.newaxis]
    h_0_i = h_0_i[..., np.newaxis]
    H_i = H_i[..., np.newaxis]
    
    # Integration variable
    hnu = normalized_distance_uplink(h,h_0_i,H_i)
    
    #Eq 55
    mu3u_to_integ = Cn2(h)*(
        (hnu*(G_Lambda*hnu + 1j*(1-(1-G_Theta)*hnu)))**(5/6)
        - G_Lambda**(5/6)*hnu**(5/3) )
    mu3u = np.real(np.trapz(mu3u_to_integ,h,axis=-1))
    
    return mu3u
    
def mu2d_par(Cn2,h_0,H,n_int=1000):
    h_0_i,H_i = np.broadcast_arrays(h_0,H)
    
    # Integration range
    #h = np.linspace(h_0_i,H_i,n_int,axis=-1)
    h = np.linspace(h_0,np.minimum(H,20e3),n_int,axis=-1)
    
    h_0_i = h_0_i[..., np.newaxis]
    H_i = H_i[..., np.newaxis]
    
    # Integration variable
    hnu = 1-normalized_distance_uplink(h,h_0_i,H_i)
    
    #Eq 55
    mu2d_to_integ = Cn2(h)*hnu**(5/3)
    mu2d = np.real(np.trapz(mu2d_to_integ,h,axis=-1))
    
    return mu2d

def scintillation_weak_uplink_tracked(G_Lambda,G_Theta,Cn2,h_0,H,zenith,k,n_int=1000):
    '''Sicntillation under weak fluctuation theory for a tracked uplink
    Andreaws ch12, p503, eq55, and p504, eq58.
    G_Lambda and G_Theta: Gausian beam parrameters, per Andrews Ch12 p489 eq9
    Cn2: function of h
    h_0: ground sation altitude
    H: target altitude
    return normalized scintillation index squared'''
    
    mu3u = mu3u_par(G_Lambda,G_Theta,Cn2,h_0,H,n_int=1000)
    
    #Eq 58
    sigBu2 = 8.70*mu3u*k**(7/6)*(H-h_0)**(5/6)/np.cos(zenith)**(11/6)
    
    return sigBu2
    
def scintillation_weak_uplink_tracked_alt(W_0,Cn2,h_0,H,zenith,k,n_int=1000):
    '''Sicntillation under weak fluctuation theory for a tracked uplink
    Andreaws ch12, p503, eq55, and p504, eq58.
    G_Lambda and G_Theta: Gausian beam parrameters, per Andrews Ch12 p489 eq9
    Cn2: function of h
    h_0: ground sation altitude
    H: target altitude
    return normalized scintillation index squared'''
    
    W_0,k_i,h_0_i,H_i = np.broadcast_arrays(W_0,k,h_0,H)
    
    # Integration range
    #h = np.linspace(h_0_i,H_i,n_int,axis=-1)
    h = np.linspace(h_0,np.minimum(H,20e3),n_int,axis=-1)
    
    W_0 = W_0[..., np.newaxis]
    k_i = k_i[..., np.newaxis]
    h_0_i = h_0_i[..., np.newaxis]
    H_i = H_i[..., np.newaxis]
    
    # Integration variable
    hnu = normalized_distance_uplink(h,h_0_i,H_i)
    
    G_Lambda,G_Theta = gaussian_beam_parameters_colimated(h,k_i,W_0)
    
    #Eq 55
    mu3u_to_integ = Cn2(h)*(
        (hnu*(G_Lambda*hnu + 1j*(1-(1-G_Theta)*hnu)))**(5/6)
        - G_Lambda**(5/6)*hnu**(5/3) )
    mu3u = np.real(np.trapz(mu3u_to_integ,h,axis=-1))
    
    #Eq 58
    sigBu2 = 8.70*mu3u*k**(7/6)*(H-h_0)**(5/6)
    
    return sigBu2
    
def scintillation_downlink_alt(W_0,Cn2,h_0,H,zenith,k,n_int=1000):
    '''Sicntillation under weak fluctuation theory for a tracked uplink
    Andreaws ch12, p503, eq55, and p504, eq58.
    G_Lambda and G_Theta: Gausian beam parrameters, per Andrews Ch12 p489 eq9
    Cn2: function of h
    h_0: ground sation altitude
    H: target altitude
    return normalized scintillation index squared'''
    
    W_0,k_i,h_0_i,H_i = np.broadcast_arrays(W_0,k,h_0,H)
    
    # Integration range
    #h = np.linspace(h_0_i,H_i,n_int,axis=-1)
    h = np.linspace(h_0,np.minimum(H,20e3),n_int,axis=-1)
    
    h_0_i = h_0_i[..., np.newaxis]
    
    sigR2int = np.trapz(Cn2(h)*(h-h_0_i)**(5/6),h,axis=-1)
    
    #Eq 38
    sigR2 = 2.25*k**(7/6)*sigR2int/np.cos(zenith)**(11/6)
    
    return sigR2
    
def scintillation_uplink_tracked(G_Theta,sigBu2):
    '''Sicntillation without weak theory restrictions for a tracked uplink
    Andreaws ch12, p506, eq60.
    G_Theta: Gausian beam parrameter, per Andrews Ch12 p489 eq9
    sigBu2: normalized scintillation index squared, logitudinal axis, under weak fluctuation theory 
    return normalized scintillation index squared'''
    
    sigIltracked2 = np.exp(
        0.49*sigBu2/(1+(1+G_Theta)*0.56*sigBu2**(6/5))**(7/6)
        + 0.51*sigBu2/(1+0.69*sigBu2**(6/5))**(5/6)
        ) - 1
    
    return sigIltracked2
    
def scintillation_uplink_tracked_xy(G_Theta,sigBu2):
    '''Sicntillation without weak theory restrictions for a tracked uplink
    Andreaws ch12, p506, eq60.
    G_Theta: Gausian beam parrameter, per Andrews Ch12 p489 eq9
    sigBu2: normalized scintillation index squared, logitudinal axis, under weak fluctuation theory 
    return normalized scintillation index, small and large scale, squared'''
        
    sigIltracked2_x = np.exp(0.49*sigBu2/(1+(1+G_Theta)*0.56*sigBu2**(6/5))**(7/6)) - 1
    sigIltracked2_y = np.exp(0.51*sigBu2/(1+0.69*sigBu2**(6/5))**(5/6)) - 1
    
    return sigIltracked2_x,sigIltracked2_y
    
def scintillation_downlink_xy(sigR2):
    '''Sicntillation for a downlink
    Andreaws ch12, p511, eq68.
    sigR2: scintillation index for square plane, squared, per Andrews Ch12 p495 eq38
    return normalized scintillation index, small and large scale, squared'''
    
    sigIltracked2_x = np.exp(0.49*sigR2/(1+1.11*sigR2**(6/5))**(7/6)) - 1
    sigIltracked2_y = np.exp(0.51*sigR2/(1+0.69*sigR2**(6/5))**(5/6)) - 1
    
    return sigIltracked2_x,sigIltracked2_y
    
def pointing_error_variance(h_0,H,zenith,W_0,k,r_0,Cr=2*np.pi):
    '''Pointing error variance under weak fluctuation theory for uplink
    Andreaws ch12, p503, eq53, second approximation.
    h_0: ground sation altitude, m
    H: target altitude, m
    zenith:  zenith angle in radians
    W_0: 1/e2 beam radius at transmit, m
    k: angular wave number
    r_0: Fired's parrameter, m
    '''
    
    a = Cr**2*W_0**2/r_0**2
    sigpe2 = 0.54*(H-h_0)**2/np.cos(zenith)**2*(np.pi/(W_0*k))**2*(2*W_0/r_0)**(5/3) \
        *(1-(a/(1+a))**(1/6))

    return sigpe2
    
def pointing_error_variance_alt(h_0,H,zenith,W_0,k,r_0,Cr=2*np.pi,n_int=1000):
    '''Pointing error variance under weak fluctuation theory for uplink
    Andreaws ch12, p503, eq53, second approximation.
    h_0: ground sation altitude, m
    H: target altitude, m
    zenith:  zenith angle in radians
    W_0: 1/e2 beam radius at transmit, m
    k: angular wave number
    r_0: Fired's parrameter, m
    '''
    
    h_0_i,H_i = np.broadcast_arrays(h_0,H)
    
    a = Cr**2*W_0**2/r_0**2
    
    h = np.linspace(h_0_i,np.minimum(H_i,20e3),n_int,axis=-1)
    
    h_0_i = h_0_i[..., np.newaxis]
    H_i = H_i[..., np.newaxis]
    
    # Integration variable
    hnu = normalized_distance_uplink(h,h_0_i,H_i)
    
    #Eq 55
    sigpe2_integ = np.trapz(Cn2(h)*hnu**2,h,axis=-1)
    
    sigpe2 = 0.725*(H-h_0)**2/np.cos(zenith)**3*W_0**(-1/3) \
        *(1-(a/(1+a))**(1/6))*sigpe2_integ

    return sigpe2
    
def scintillation_uplink_untracked(sigpe2,h_0,H,zenith,W_0,r_0,L,r,sigIltracked2,W):

    sigpe = np.sqrt(sigpe2)

    sigIluntracked2 = 5.95*(H-h_0)**2/np.cos(zenith)**2*(2*W_0/r_0)**(5/3) \
        *( ((r-sigpe)/(L*W))**2*((r-sigpe) > 0) + (sigpe/(L*W))**2 ) \
        + sigIltracked2
    
    return sigIluntracked2
    
def get_scintillation_uplink_untracked(h_0,H,zenith,k,W_0,Cn2,r,Cr=2*np.pi):

    L = slant_range(h_0,H,zenith,Re)

    G_Lambda,G_Theta = gaussian_beam_parameters_colimated(L,k,W_0)
    
    W = beam_radius(W_0,G_Theta,G_Lambda)
    
    sigBu2 = scintillation_weak_uplink_tracked(G_Lambda,G_Theta,Cn2,h_0,H,zenith,k)
    
    sigIltracked2 = scintillation_uplink_tracked(G_Theta,sigBu2)
    
    r_0 = Fried_param(zenith,k,Cn2,h_0,H)
    
    sigpe2 = pointing_error_variance(h_0,H,zenith,W_0,k,r_0,Cr)
    
    sigIluntracked2 = scintillation_uplink_untracked(sigpe2,h_0,H,zenith,W_0,r_0,L,r,sigIltracked2,W)
    
    return sigIluntracked2
    
def get_scintillation_uplink_untracked_xy(h_0,H,zenith,k,W_0,Cn2,r,Cr=2*np.pi):
    
    L = slant_range(h_0,H,zenith,Re)

    G_Lambda,G_Theta = gaussian_beam_parameters_colimated(L,k,W_0)
    
    W = beam_radius(W_0,G_Theta,G_Lambda)
    
    sigBu2 = scintillation_weak_uplink_tracked(G_Lambda,G_Theta,Cn2,h_0,H,zenith,k)
    
    sigIltracked2_x,sigIltracked2_y = scintillation_uplink_tracked_xy(G_Theta,sigBu2)
    
    r_0 = Fried_param(zenith,k,Cn2,h_0,H)
    
    sigpe2 = pointing_error_variance(h_0,H,zenith,W_0,k,r_0,Cr)
    
    sig2_x = scintillation_uplink_untracked(sigpe2,h_0,H,zenith,W_0,r_0,L,r,sigIltracked2_x,W)
    
    sig2_y = sigIltracked2_y
    
    sig2_x, sig2_y = np.broadcast_arrays(sig2_x, sig2_y)
    
    return apply_min(sig2_x), apply_min(sig2_y)
    
def get_scintillation_uplink_tracked_xy(h_0,H,zenith,k,W_0,Cn2,Cr=2*np.pi):
    
    L = slant_range(h_0,H,zenith,Re)

    G_Lambda,G_Theta = gaussian_beam_parameters_colimated(L,k,W_0)
    
    W = beam_radius(W_0,G_Theta,G_Lambda)
    
    sigBu2 = scintillation_weak_uplink_tracked(G_Lambda,G_Theta,Cn2,h_0,H,zenith,k)
    
    sigIltracked2_x,sigIltracked2_y = scintillation_uplink_tracked_xy(G_Theta,sigBu2)
    
    sig2_x = sigIltracked2_x
    
    sig2_y = sigIltracked2_y
    
    return apply_min(sigIltracked2_x), apply_min(sigIltracked2_y)
    
def get_scintillation_downlink_xy(h_0,H,zenith,k,W_0,Cn2,Cr=2*np.pi):
    
    sigR2 = scintillation_downlink_alt(W_0,Cn2,h_0,H,zenith,k)
    
    sigIltracked2_x,sigIltracked2_y = scintillation_downlink_xy(sigR2)
    
    sig2_x = sigIltracked2_x
    
    sig2_y = sigIltracked2_y
    
    return apply_min(sigIltracked2_x), apply_min(sigIltracked2_y)
    
def gamma_gamma_distrib_pdf(sig2_x,sig2_y,Ie,I):

    alpha = 1/mpin(sig2_x)
    beta  = 1/mpin(sig2_y)
    Ie = mpin(Ie)
    I = mpin(I)
    
    avg = (alpha+beta)/2
    
    P_I = 2*(alpha*beta)**avg/(gamma(alpha)*gamma(beta)*I)*(I/Ie)**avg*besselk(alpha-beta,2*np.sqrt(alpha*beta*I/Ie))
    
    return mpout(P_I)
    
def gamma_gamma_distrib_cdf_direct(sig2_x,sig2_y,Ie,It):
    alpha = 1/mpin(sig2_x)
    beta  = 1/mpin(sig2_y)
    It = mpin(It)/mpin(Ie)
    
    P_I = np.pi/(mpsin(np.pi*(alpha-beta))*gamma(alpha)*gamma(beta)) \
        *(  (alpha*beta*It)**beta /( beta*gamma(beta-alpha+1))*hyp1F2( beta, beta+1,beta-alpha+1,alpha*beta*It) \
           -(alpha*beta*It)**alpha/(alpha*gamma(alpha-beta+1))*hyp1F2(alpha,alpha+1,alpha-beta+1,alpha*beta*It) )
        
    return mpout(P_I)
    
def gamma_gamma_distrib_cdf_hypercomb(sig2_x,sig2_y,Ie,It):

    pdf = gamma_gamma_distrib_pdf(sig2_x,sig2_y,Ie,I)
    
    mphypercomb = np.frompyfunc(lambda i,o: mp.hypercomb(i,o,zeroprec=5),2,1) #,verbose=True
    array_of_list = np.frompyfunc(lambda u:[u],1,1)
    
    alpha = 1/mpin(sig2_x)
    beta  = 1/mpin(sig2_y)
    
    It = mpin(It)/mpin(Ie)
    
    def comb_param_function(a,b,I):
        tpl1 = ([a*b*I,  1/b], [b, 1], [], [a, b, b-a+1], [b], [b+1, b-a+1], a*b*I)
        tpl2 = ([a*b*I, -1/a], [a, 1], [], [a, b, a-b+1], [a], [a+1, a-b+1], a*b*I)
        return [tpl1,tpl2]
    
    hyp_inputs = array_of_list(alpha)+array_of_list(beta)+array_of_list(It)
 
    P_I = np.pi/(mpsin(np.pi*(alpha-beta))) * mphypercomb(comb_param_function, hyp_inputs)
    return mpout(P_I)    
      
    
def gamma_gamma_to_alpha_mu(sig2_x,sig2_y,orders=[2,3],add=False,I0=1):
    lgamma = scsp.gammaln
    def select_lsq(fun,x0,jac,bounds): return scop.least_squares(fun,x0,jac,bounds,'trf',ftol=1e-15,xtol=1e-15,gtol=1e-15,max_nfev=1000)
    array_lsq = np.frompyfunc(select_lsq,4,1)
    
    bdshape = list(np.broadcast(sig2_x,sig2_y).shape)
    
    def EXn(n,sigx,sigy,I0):
        a = 1/sigx
        b = 1/sigy
        return scsp.gammaln(a+n) +  scsp.gammaln(b+n) - scsp.gammaln(a) - scsp.gammaln(b) - np.log(a*b/I0)*n
        
    ERgg = [EXn(nx,sig2_x,sig2_y,I0) for nx in orders]
    
    def ERalphamu(x,ERn,orders):
        alpha,mu = x
        ERau = [(nx-1)*lgamma(mu)+lgamma(mu+nx/alpha)-nx*lgamma(mu+1/alpha) for nx in orders]
        return [ERau[0]-ERn[0],ERau[1]-ERn[1]]
        
    def Jacobian(x,orders):
        alpha,mu = x
        digmu = scsp.digamma(mu)
        dign  = [scsp.digamma(mu+nx/alpha) for nx in orders]
        dig1  = scsp.digamma(mu+1/alpha)
        return np.array([[ nx/alpha**2*(dig1-dignx), (nx-1)*digmu+dignx-nx*dig1] for nx, dignx in zip(orders,dign)])
        
    if add:
        func  = np.frompyfunc(lambda ER0,ER1: lambda x:ERalphamu(x,[ER0,ER1],orders),2,1)(ERgg[0],ERgg[1])
        initc = np.frompyfunc(lambda alpha0,mu0:[alpha0,mu0],2,1)(0.5,1)
        bnds = np.frompyfunc(lambda a0,m0,a1,m1:([a0,m0],[a1,m1]),4,1)(min_log,min_log,np.inf,np.inf)
    else:
        func  = np.frompyfunc(lambda ER0,ER1: lambda x:ERalphamu(x,[ER0,ER1],orders),2,1)(ERgg[0],ERgg[1])
        initc = np.frompyfunc(lambda alpha0,mu0:[alpha0,mu0],2,1)(0.5*np.ones(bdshape),np.sqrt(1/sig2_x/sig2_y))
        bnds = np.frompyfunc(lambda a0,m0,a1,m1:([a0,m0],[a1,m1]),4,1)(min_log*np.ones(bdshape),min_log*np.ones(bdshape),np.inf,np.inf)
    jac = lambda x:Jacobian(x,orders)
    
    res = array_lsq(func,initc,jac,bnds)
    resa = np.frompyfunc(lambda obj:tuple(obj.x),1,2)(res)
    resst = np.frompyfunc(lambda obj:obj.status,1,1)(res)
    #assert np.all(resst != 0)
    alpha, mu = resa
    mu = mu.astype(np.float64)
    alpha = alpha.astype(np.float64)
    r = np.exp( np.log(mu)/alpha + lgamma(mu) - lgamma(mu+1/alpha) )
    
    return (alpha,mu,r)
    
def alpha_mu_cdf(alpha,mu,r,I0,It):
    gvar = mu*(It/I0/r)**alpha
    gvar,mu = np.broadcast_arrays(gvar,mu)
    cdf = 1 - scsp.gammaincc(mu,gvar)
    return cdf

def alpha_mu_inv_cdf(alpha,mu,r,I0,P):
    gvar = scsp.gammainccinv(mu,P)
    It = I0*r*(gvar/mu)**(1/alpha)
    return It
    
def alpha_mu_pdf(alpha,mu,r,I0,It):
    lgmu = np.log(mu)
    lgal = np.log(alpha)
    lgP = np.log(It/I0/r)
    return 1/I0*np.exp( lgal + mu*lgmu + (alpha*mu)*lgP - scsp.gammaln(mu) -mu*(It/I0/r)**alpha )
    
def alpha_mu_cdf_sum(alpha,mu,r,Pe,internal_scale,output_scale=None,cumulative=False):

    internal_scale_ax = internal_scale[np.newaxis,:]
    
    if output_scale is None: output_scale = internal_scale

    cdfs = alpha_mu_cdf(alpha[0],mu[0],r[0],Pe[0],internal_scale_ax)
    cdfst = cdfs.transpose().flatten()
    covoltmp = np.concatenate([[cdfst[0]],cdfst[1:]-cdfst[:-1]],0)

    if cumulative:
        covlog = np.interp(output_scale,internal_scale,np.cumsum(covoltmp))
        covolres = np.zeros((len(alpha),len(output_scale)))
        covolres[0,:] = covlog

    for i in range(1,len(alpha)):
        cdfs = alpha_mu_cdf(alpha[i],mu[i],r[i],Pe[i],internal_scale_ax)
        cdfst = cdfs.transpose().flatten()
        cdfst = np.concatenate([[cdfst[0]],cdfst[1:]-cdfst[:-1]],0)
        
        covolval = np.convolve(covoltmp,cdfst,mode='full')[0:(0+len(cdfst))]
        covoltmp = covolval
        
        if cumulative:
            covlog = np.interp(output_scale,internal_scale,np.cumsum(covolval))
            covolres[i,:] = covlog
        
    if cumulative: return covolres
    else: return np.interp(output_scale,internal_scale,np.cumsum(covolval))
    
#def SNR0_shot(i_sig):pass
    
#def SNR0_APD(i_sig,sig_n):

def SNR0_NEP(p_sig,NEP,BW):
    return p_sig / (NEP*np.sqrt(BW))

def SNR0_sig(i_sig,sig_n): return i_sig/sig_n
    
def gamma_gamma_BER_NEP_single(sig2_x,sig2_y,Ie,NEP,BW):
    def integrand(u):
        return gamma_gamma_distrib_pdf(sig2_x,sig2_y,Ie,u)*scsp.erfc(SNR0_NEP(u,NEP,BW)*u/(2*np.sqrt(2)))
    try:
        return 0.5*scin.quad(integrand,0,np.inf)[0]
    except ZeroDivisionError:
        return 1   
gamma_gamma_BER_NEP = np.frompyfunc(gamma_gamma_BER_NEP_single,5,1)        
    

def alpha_mu_BER_NEP_single(alpha,mu,r,Ie,NEP,BW):
    def integrand(u):
        return alpha_mu_pdf(alpha,mu,r,Ie,u)*scsp.erfc(SNR0_NEP(u,NEP,BW)*u/(2*np.sqrt(2)))
    try:
        #return 0.5*scin.quad(integrand,0,1e10,points=[Ie])[0]
        return 0.5*scin.quad(integrand,0,np.inf)[0]
        #return 0.5*np.float(scin.romberg(integrand,0,100))
    except ZeroDivisionError:
        return 1
alpha_mu_BER_NEP = np.frompyfunc(alpha_mu_BER_NEP_single,6,1)
    
def alpha_mu_BER_NEP_fixed(alpha,mu,r,Ie,NEP,BW):
    u = np.linspace(0,100,10000)
    integrand = alpha_mu_pdf(alpha,mu,r,Ie,u)*scsp.erfc(SNR0_NEP(u,NEP,BW)*u/(2*np.sqrt(2)))
    return 0.5*scin.trapz(integrand,u)
alpha_mu_BER_NEP_f = np.frompyfunc(alpha_mu_BER_NEP_fixed,6,1)
    
'''def gamma_gamma_distrib_cdf_alt7(sig2_x,sig2_y,Ie,It,orders=[2,3]):
    lgamma = scsp.gammaln
    array_fsolve = np.frompyfunc(scop.fsolve,2,3)
    def select_lsq(fun,x0,jac,bounds): return scop.least_squares(fun,x0,jac,bounds,'trf',ftol=1e-15,xtol=1e-15,gtol=1e-15,max_nfev=1000)
    array_lsq    = np.frompyfunc(select_lsq,4,1)
    
    a  = 1/sig2_x
    b  = 1/sig2_y
    It = It/Ie
    
    bdshape = list(np.broadcast(a,b).shape)
    
    abdiv = lgamma(a+1)+lgamma(b+1)
    abmul = lgamma(a)+lgamma(b)
    ERgg = [lgamma(a+nx)+lgamma(b+nx)+(nx-1)*abmul-abdiv for nx in orders]
    
    def ERalphamu(x,ERn,orders):
        alpha,mu = x
        
        ERau = [(nx-1)*lgamma(mu)+lgamma(mu+nx/alpha)-nx*lgamma(mu+1/alpha) for nx in orders]
        
        return [ERau[0]-ERn[0],ERau[1]-ERn[1]]
        
    def Jacobian(x,orders):
        alpha,mu = x
        digmu = scsp.digamma(mu)
        dign  = [scsp.digamma(mu+nx/alpha) for nx in orders]
        return np.array([[ nx*(nx-1)/alpha**2*dignx, (nx-1)*(digmu-dignx)] for nx, dignx in zip(orders,dign)])
        
    func  = np.frompyfunc(lambda ER0,ER1: lambda x:ERalphamu(x,[ER0,ER1],orders),2,1)(ERgg[0],ERgg[1])
    initc = np.frompyfunc(lambda alpha0,mu0:[alpha0,mu0],2,1)(0.5*np.ones(bdshape),np.sqrt(a*b))
    min_log = 1e-30*np.ones(bdshape)
    bnds = np.frompyfunc(lambda a0,m0,a1,m1:([a0,m0],[a1,m1]),4,1)(min_log,min_log,np.inf,np.inf)
    jac = lambda x:Jacobian(x,orders)
    
    print('solving...')
    res = array_lsq(func,initc,jac,bnds)
    resa = np.frompyfunc(lambda obj:tuple(obj.x),1,2)(res)
    resst = np.frompyfunc(lambda obj:obj.status,1,1)(res)
    assert np.all(resst != 0)
    alpha, mu = resa
    mu = mu.astype(np.float64)
    alpha = alpha.astype(np.float64)
    r = np.exp( np.log(mu)/alpha + lgamma(mu) - lgamma(mu+1/alpha) )
    print('done!') 
    
    gvar = mu*(It/r)**alpha
    gvar,mu = np.broadcast_arrays(gvar,mu)

    cdf = 1 - scsp.gammaincc(mu,gvar)
    
    return (cdf,alpha,mu,r)'''