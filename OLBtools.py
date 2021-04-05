'''
Copyright © 2020 Paul Serra, Peter Grenfell, Ondrej cierny

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
The Software is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the Software.
'''

#----------------------------------------------------------
# Imports
import inspect
from functools import wraps

import numpy as np
import mpmath as mp
import scipy.special as scsp
import scipy.optimize as scop
import scipy.integrate as scin
import scipy.interpolate as scit

mp.mp.dps = 300

#----------------------------------------------------------
# Constants
qe = 1.60217662e-19 # charge of an electron, C (A.s)
kb = 1.38064852e-23 # Boltzmann, J/K
hp = 6.62607004e-34 # Planck, J.s
c  = 299.792458e6   # Speed of light, m/s
T  = 298.15         # Temperature of electronics, K
Re = 6378e3         # Earth radius, m
mu = 3.986004418e14 # Standard gravitational parameter, m3/s2
we = 7.292115e-5    # Earth rotation, radians/seconds

earth_rotation = 7.292115e-5 # Earth angular speed, radians by seconds

min_log = 1e-30

#----------------------------------------------------------
# Helpers
def degrees(a): return a*180/np.pi
def radians(a): return a*np.pi/180

def angular_wave_number(vlambda):return 2*np.pi/vlambda

def extend_integ_axis(arrays,integ_var):
    for ar in arrays:
        ar = ar[..., np.newaxis]
    arl = np.broadcast_arrays(integ_var,h)
    return arl[1:]
    
mpin    = np.frompyfunc(mp.mpf,1,1)
gamma   = np.frompyfunc(mp.gamma,1,1)
besselk = np.frompyfunc(mp.besselk,2,1)
hyp1F2  = np.frompyfunc(mp.hyp1f2,4,1)
mpexp   = np.frompyfunc(mp.exp,1,1)
mpsin   = np.frompyfunc(mp.sin,1,1)
mpout   = np.frompyfunc(float,1,1)

def beam_radius(W_0,G_Theta,G_Lambda):
    return W_0/np.sqrt(G_Theta**2 + G_Lambda**2)
    
def apply_min(x):
    return x #(x>min_log)*x + (x<=min_log)*min_log
    
def initializer(func):
    """
    Automatically assigns the parameters.
    Nadia Alramli, from stackoverflow
    https://stackoverflow.com/questions/1389180/automatically-initialize-instance-variables

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper

#----------------------------------------------------------
# Link Geometry

def slant_range(h_0,H,zenith,Re):
    '''Slant range for a spacecraft, per appendix A
    h_0: Station altitude above atmosphere 0 ref, in m
    H: spacecraft altitude above atmosphere 0 ref, in m
    Re: distance from geotcenter to atmosphere 0 ref, in m
    zenith: ground station zenith angle in radians'''
    h_0 = h_0 + Re
    H = H + Re
    return np.sqrt( (h_0*np.cos(zenith))**2 + H**2 - h_0**2 ) - h_0*np.cos(zenith)
    
def earth_centered_intertial_to_latitude_longitude(latitude,longitude,thetaE,x,y,z):
    '''Peter Grenfell'''

    ctE = np.cos(thetaE);    stE = np.sin(thetaE)
    cph = np.cos(latitude);  sph = np.sin(latitude)
    cla = np.cos(longitude); sla = np.sin(longitude)
    
    tx =  cla*cph*x + sla*cph*y + sph*z
    ty =     -sla*x +     cla*y
    tz = -cla*sph*x - sla*sph*y + cph*z
    
    x =  ctE*tx + stE*ty
    y = -stE*tx + ctE*ty
    z = tz
    
    return x,y,z
    

def pass_azimuth_elevation_and_time(latitude,longitude,altitude,inclination,min_zenith,max_zenith,npts,mu,Re,we):
    '''Peter Grenfell
    latitude,longitude
    orbit_altitude
    min_zenith
    Re: distance from geotcenter to atmosphere 0 ref, in m'''
    
    #Slant range at min zenith (max elevation)
    sr_min = slant_range(0,altitude,min_zenith,Re)
    
    #Slant range at max zenith (min elevation)
    sr_max = slant_range(0,altitude,max_zenith,Re)
    
    def law_of_cosines(a,b,c):
        #law of cosines, return angle, c is the far side
        return np.arccos( (a**2+b**2-c**2)/(2*a*b) )
    
    #Top of pass in earth centered referential, at OGS lat/long
    x = altitude*np.sin(min_zenith)*np.cos(inclination)
    y = altitude*np.sin(min_zenith)*np.sin(inclination)
    h = Re + altitude*np.cos(min_zenith)
    
    # Angle of OGS from orbit plan at geocenter.
    orbit_plan_angle_geocenter = law_of_cosines(Re,Re+altitude,sr_min)
    
    # Angle of the horizon at geocenter from vertical (zentith = 0), in corbit plane.
    horizon_angle_in_plane_geocenter = law_of_cosines(Re,Re+altitude,sr_max)
    
    # Angle of horizon our of orbit plane, using spherical pythagorean theorem. 
    horizon_angle_geocenter = np.arccos( np.cos(horizon_angle_in_plane_geocenter) / np.cos(orbit_plan_angle_geocenter) )
    
    # Orbit periode
    T_orbit = 2*np.pi*np.sqrt( (altitude+Re)**3/mu )
    
    # time to the horizon from vertical (zentith = 0)
    t_horizon = horizon_angle_geocenter*T_orbit/(2*np.pi)
    
    # Timescale
    t = np.linspace(-t_horizon,t_horizon,npts)
    
    # Orbit phase
    sat_phase = 2*np.pi*t/T_orbit
    
    #Top of pass in earth centered inertial
    #x,y,z = earth_centered_intertial_to_latitude_longitude(-latitude,-longitude,0,x=h,y=x,z=y)
    
    # Circular equatorial orbit
    x_orbit = (Re + altitude)*np.cos(sat_phase)
    y_orbit = (Re + altitude)*np.sin(sat_phase)
    z_orbit = 0
    
    # offset from OGS, rotation arround y axis
    coff = np.cos(orbit_plan_angle_geocenter)
    soff = np.sin(orbit_plan_angle_geocenter)
    x_off =  coff*x_orbit
    y_off =  y_orbit
    z_off =  soff*x_orbit
    
    # Adding inclnation, rotation arround x axis
    cinc = np.cos(inclination)
    sinc = np.sin(inclination)
    x_inc = x_off
    y_inc =  cinc*y_off + sinc*z_off
    z_inc = -sinc*y_off + cinc*z_off
   
    # Putting it on top off OGS at t0
    x, y, z  = earth_centered_intertial_to_latitude_longitude(latitude,longitude,0,x_inc,y_inc,z_inc)
    
    # OGS coordinates
    x0,y0,z0 = earth_centered_intertial_to_latitude_longitude(latitude,longitude,we*t,Re,0,0)
    x0,y0,z0 = np.broadcast_arrays(x0,y0,z0)
    ogs = np.array([x0,y0,z0]).transpose()
    
    if 0:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x,y,z)
        ax.scatter(x0,y0,z0)
        ax.scatter(0,0,0)
    
    # Sat vector from OGS 
    xs = x-x0
    ys = y-y0
    zs = z-z0
    ns = np.sqrt(xs**2 + ys**2 + zs**2)
    
    # satelite unit vector
    sat = np.array([xs/ns,ys/ns,zs/ns]).transpose()
    
    # Zentih angle at OGS
    zenith = np.arccos((x0*xs + y0*ys + z0*zs)/ns/Re)
    
    # Sat unit vector in OGS horizontal plane
    sat_horizontal = np.cross(ogs,sat)/Re #90 deg away
    
    # East vector at OGS
    east = np.cross([0,0,1],ogs)/Re
    
    # North vector at OGS
    north = np.cross(ogs,east)/Re
    
    def dot_2D(a,b):
        return a[:,0]*b[:,0] + a[:,1]*b[:,1] + a[:,2]*b[:,2]
    
    cos_east  = dot_2D(sat_horizontal,east)
    cos_north = dot_2D(sat_horizontal,north)
    
    # Azimuth angle at OGS
    azimuth = np.arctan2(cos_north,-cos_east)
    
    #on top of lat/long = 0
    #+z
    #^
    #|
    #O->+y
    #+x
    
    
    return t,zenith,azimuth
    

#----------------------------------------------------------
# Turbulence and scintillation
#----------------------------------------------------------

#----------------------------------------------------------
# Cn2 Models

def Cn2_SLC(h):
    # Submarine Laser Comunication (SLC) model, day and night, for h > 1500m
    # For some reason, the SLC model is for the atmosphere, and is based on the median values for the AIR Force Maui Optical Station 
    # Laser Beam Propagation through random Media, 2nd edition  Larry C. Andrews and Ronald L. Phillips page 481-482
    return (1500<=h)*(h<7200)*8.87e-7/h**3 + (7200<=h)*(h<20000)*2e-16/h**(1/2)

def Cn2_HV_ACTLIMB(h):
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
    
#----------------------------------------------------------
    
def Rytov_var(Cn2_h,k,L):
    # Laser Beam Propagation through random Media, 2nd edition  Larry C. Andrews and Ronald L. Phillips page 140
    return 1.23*Cn2_h*k**(5/7)*L**(11/6)
    
def Fried_param(zenith,k,Cn2,h_0,H,n_int=1000):
    '''Fried's pararamter for a ground station.
    Andrews Ch12, p492, Eq23
    zenith:  zenith angle in radians
    k: angular wave number
    Cn2: function of h
    h_0: ground sation altitude
    H: target altitude'''
    
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
    
#----------------------------------------------------------
# Link budget functions
#----------------------------------------------------------

def path_loss_gaussian(beam_radius, wavelength, distance, rx_diameter, pointing_error=0):
    # From Matlab linkbudget, author O cierny / P serra
    '''Calculates path loss given a Gaussian beam, and an Rx aperture at a
    specific distance. Assumes normalized input power, returns dB loss.
    All units in meters / radians.
    beam_radius: 1/e^2 radius of the beam at Tx [m]
    pointing_error: misalignment between Tx and Rx [rad] (optional)'''

    # Calculate beam waist at the given distance due to diffraction [m]
    beam_radius_rx = beam_radius * np.sqrt(1 + ((wavelength*distance)/(np.pi*beam_radius**2))**2);

    # Calculate distance from center of Gaussian due to pointing error [m]
    radial_distance = np.tan(pointing_error) * distance;

    # Calculate irradiance using Gaussian formula [W/m^2]
    irradiance = (2/(np.pi*beam_radius_rx**2))*np.exp((-2*radial_distance**2)/beam_radius_rx**2);

    # Calculate power at Rx aperture [W]
    rx_area = np.pi*(rx_diameter/2)**2 # [m^2]
    rx_power = irradiance * rx_area # [W]

    # Calculate dB loss
    path_loss_dB = 10*np.log10(rx_power) # [dB]
    return path_loss_dB
    
def fwhm_to_radius(fwhm, wavelength):
    # From Matlab linkbudget, author O cierny
    '''Calculates the 1/e^2 Gaussian beam radius given the full width half
    maximum angle.
    fwhm: full width half maximum angle [rad]
    beam_radius: 1/e^2 beam radius at origin [m]'''
    beam_radius = (wavelength * np.sqrt(2*np.log(2))) / (np.pi * fwhm);
    return beam_radius
    
#----------------------------------------------------------
# LOWTRAN functions
#----------------------------------------------------------

def LOWTRAN_transmittance(zenith,lambda_min,lambda_max,step=1e7/20,model=5,H=0,haze=0):
    '''Based on TransmittanceGround2Space.py example
    model: 0-6, see Card1 "model" reference in LOWTRAN user manual. 5=subarctic winter
    H: altitude of observer [m]
    zenith: observer zenith angle [rad]
    lambda_min: shortest wavelength [nm] ???
    lambda_max: longest wavelength cm^-1 ???
    step: wavelength step size
    '''
    
    import lowtran

    c1 = {'model': model,
          'ihaze': haze,
          'h1': H*1e3,
          'angle': degrees(zenith),
          'wlshort': lambda_min*1e9,
          'wllong': lambda_max*1e9,
          'wlstep': 1e7/step,
          }

    TR = lowtran.transmittance(c1)
    
    return TR
    
def transmittance(zenith,wavelength,model,H):
    
    wavelength = np.asarray(wavelength)
    
    wl_step_large=1e7/20
    #wl_step_min=1e7/5

    lambda_low  = 1/(1/wavelength.min() + wl_step_large*1e-2)
    lambda_high = 1/(1/wavelength.max() - wl_step_large*1e-2)
    
    #degrees(zenith)
    T = LOWTRAN_transmittance(zenith,lambda_low,lambda_high,wl_step_large,model,H)
    
    x = T.wavelength_nm[::-1]*1e-9
    y = T['transmission'][0,::-1,:]
    w = wavelength
    
    if w.shape: ws = w.shape[0]
    else: ws = 1
    
    r = np.zeros((ws,y.shape[1]))
    for n in range(y.shape[1]):
       r[:,n] = np.interp(w, x, y[:,n])
    
    return r
    
#def MODTRAN_transmittance(zenith

class Quadcell:
    @initializer #automaticaly add agurment to each instances
    def __init__(self, gap=0,responsivity=0.5,transimpedance=1e6,amplifier_noise=1e-5,bandwidth=1e3):
        ''' Object for a 4-quadrant pin detector, provides methods for postion, noise and noise-equivalent angle
        gap: size of the gap between quadrants, in m or PSF size units
        responsivity: photodiode repsonsivity in A/W
        transimpedance: amplifier gain in V/A or ohm
        amplifier_noise: output refered noise of the amplifier, in v/rtHz
        bandwidth: bandwidth for the detection, in Hz
        '''
        
        #Quadrants defined as A,B,C,D, in trigonometric order, in quadcell front view.
        # A: +x,+y, B:-x,+y, C:-x,-y, D:+x,-y
        
        #Cumulative sum of the PSF over each quadrant, as a scipy interpolation function
        self.PSF_sum_2D_A = None
        self.PSF_sum_2D_B = None
        self.PSF_sum_2D_C = None
        self.PSF_sum_2D_D = None
        
    def cumulated_gaussian_PSF_on_square_mask(W,x1,x2,y1,y2):
        #W: spot size for the PSF, such as PSF(r) = 2/(pi*W**2)*exp(-2*r**2/w**2)
        coef = np.sqrt(2)/W
        return 0.25*(scsp.erf(coef*x2)-scsp.erf(coef*x1))*(scsp.erf(coef*y2)-scsp.erf(coef*y1))
        
    def cumulated_gaussian_PSF_on_quadrant_mask(W,x1,y1):
        #W: spot size for the PSF, such as PSF(r) = 2/(pi*W**2)*exp(-2*r**2/w**2)
        coef = np.sqrt(2)/W
        return 0.25*(1-scsp.erf(coef*x1))*(1-scsp.erf(coef*y1))
        
    def set_PSF_gaussian(self,W):
        #W: spot size for the PSF, such as PSF(r) = 2/(pi*W**2)*exp(-2*r**2/w**2)
        
        def culative_sum_function(x,y):            
            x_shifted = self.gap/2-x
            y_shifted = self.gap/2-y
            return Quadcell.cumulated_gaussian_PSF_on_quadrant_mask(W,x_shifted,y_shifted)
            
        self.set_all_quadrant_sum(culative_sum_function)
        
    def set_PSF_1D(self,r_samp,v_samp,n_int2d=1000,deg=3):
        ''' Compute and set the cumulative sum of the PSF for each quadrant, using radial PSF samples
            The PSF values are linearly interpolated
            The PSF is automaticaly normalised
        r_samp: sample postion, m or interpolation function input unit
        v_samp: sample relative intesisty
        n_int2d resolution of the cumulative sum on the quadrant
        def: interpolation degree for the cumulative sum on the quadrant
        '''
        r1 = r_samp[:-1]
        r2 = r_samp[1:]
        v1 = v_samp[:-1]
        v2 = v_samp[1:]
        normalize = np.pi*np.sum(r2**2*v2 + (r1**2+r1*r2+r2**2)*(v1-v2)/3 - r1**2*v1)
        v_samp = v_samp/normalize
        
        rf = r_samp[-1]
        x = np.linspace(-rf,rf,n_int2d)
        y = np.linspace(-rf,rf,n_int2d)
        xx, yy = np.meshgrid(x,y)
        r = np.sqrt(xx**2+yy**2)
        v = (r<=rf)*np.interp(r,r_samp,v_samp)
        
        v_integ = np.cumsum(np.cumsum(v,axis=0),axis=1)*(2*rf/n_int2d)**2

        v_integ_lookup = scit.RectBivariateSpline(x-self.gap/2,y-self.gap/2,v_integ,kx=deg,ky=deg)

        self.set_all_quadrant_sum(v_integ_lookup)
        
    def set_all_quadrant_sum(self,sum_function):
        # Set all quadrant if the PSF is axi-sytmetrical.
        self.PSF_sum_2D_A = lambda x,y,dx=0,dy=0:sum_function( x, y,dx,dy,grid=False)
        self.PSF_sum_2D_B = lambda x,y,dx=0,dy=0:sum_function(-x, y,dx,dy,grid=False)
        self.PSF_sum_2D_C = lambda x,y,dx=0,dy=0:sum_function(-x,-y,dx,dy,grid=False)
        self.PSF_sum_2D_D = lambda x,y,dx=0,dy=0:sum_function( x,-y,dx,dy,grid=False)
    
    def set_PSF_2D(self,x_samp,y_samp,v_samp,n_int2d=1000,deg=3):
        ''' Compute and set the cumulative sum of the PSF for each quadrant, 2D grid PSF samples
            The PSF values are linearly interpolated
            The PSF is automaticaly normalised
        x_samp: sample postion, m or interpolation function input unit
        y_samp: sample postion, m or interpolation function input unit
        v_samp: sample relative intesisty
        n_int2d resolution of the cumulative sum on the quadrant
        deg: interpolation degree for the cumulative sum on the quadrant
        '''
        
        normalize = np.sum(v_samp)*(x_samp[1]-x_samp[0])*(y_samp[1]-y_samp[0])
        v_samp = v_samp/normalize
        
        # PSF linear interpolation
        x = np.linspace(x_samp[0],x_samp[-1],n_int2d)#[:, np.newaxis]
        y = np.linspace(y_samp[0],y_samp[-1],n_int2d)#[np.newaxis, :]
        xx, yy = np.meshgrid(x,y)
        v_samp_lookup = scit.RectBivariateSpline(x_samp,y_samp,v_samp,kx=1,ky=1)
        
        dx_dy_integ_coef = (1/n_int2d)**2
        dx_dy_integ_coef *= (x_samp[-1]-x_samp[0])*(y_samp[-1]-y_samp[0])
        
        #integration over each quadrant, and sum interpolation function, shifted by gap value
        v = v_samp_lookup( xx, yy, grid=False)
        v_integ = np.cumsum(np.cumsum(v,axis=0),axis=1)*dx_dy_integ_coef
        self.PSF_sum_2D_A = lambda vx,vy,dx=0,dy=0: (scit.RectBivariateSpline(x+self.gap/2,y+self.gap/2,v_integ,kx=deg,ky=deg))( vx, vy,dx,dy,grid=False)
        
        v = v_samp_lookup(-xx, yy, grid=False)
        v_integ = np.cumsum(np.cumsum(v,axis=0),axis=1)*dx_dy_integ_coef
        self.PSF_sum_2D_B = lambda vx,vy,dx=0,dy=0: (scit.RectBivariateSpline(x+self.gap/2,y+self.gap/2,v_integ,kx=deg,ky=deg))(-vx, vy,dx,dy,grid=False)
        
        v = v_samp_lookup(-xx,-yy, grid=False)
        v_integ = np.cumsum(np.cumsum(v,axis=0),axis=1)*dx_dy_integ_coef
        self.PSF_sum_2D_C = lambda vx,vy,dx=0,dy=0: (scit.RectBivariateSpline(x+self.gap/2,y+self.gap/2,v_integ,kx=deg,ky=deg))(-vx,-vy,dx,dy,grid=False)
        
        v = v_samp_lookup( xx,-yy, grid=False)
        v_integ = np.cumsum(np.cumsum(v,axis=0),axis=1)*dx_dy_integ_coef
        self.PSF_sum_2D_D = lambda vx,vy,dx=0,dy=0: (scit.RectBivariateSpline(x+self.gap/2,y+self.gap/2,v_integ,kx=deg,ky=deg))( vx,-vy,dx,dy,grid=False)
        
    def eval_quadrants(self,x_spot,y_spot,dx=0,dy=0):
        #Normalized quadrant amplitude for a give spot postition
        A = self.PSF_sum_2D_A(x_spot,y_spot,dx,dy)
        B = self.PSF_sum_2D_B(x_spot,y_spot,dx,dy)
        C = self.PSF_sum_2D_C(x_spot,y_spot,dx,dy)
        D = self.PSF_sum_2D_D(x_spot,y_spot,dx,dy)
        return A,B,C,D
        
    def response_from_quadrants(self,quadrants):
        #give the slope of the quadcell response for given quadrant values
        A,B,C,D = quadrants
        quad_sum = A+B+C+D
        x_resp = ((A+D)-(B+C))/quad_sum
        y_resp = ((A+B)-(C+D))/quad_sum
        return x_resp, y_resp
    
    def response(self,x_spot,y_spot):
        #give the quadcell response for a give spot position
        return self.response_from_quadrants(self.eval_quadrants(x_spot,y_spot))
        
    def slope_from_quadrant(self,quadrants,quadrants_da):
        #give the slope of the quadcell response for given quadrant values and derivative
        A,B,C,D = quadrants
        A_da,B_da,C_da,D_da = quadrants_da
        quad_sum = A+B+C+D
        quad_sum_da = A_da+B_da+C_da+D_da
        
        x_resp_num = (A+D)-(B+C)
        x_resp_num_da = (A_da+D_da)-(B_da+C_da)
        y_resp_num = (A+B)-(C+D)
        y_resp_num_da = (A_da+B_da)-(C_da+D_da)
        
        x_resp_da = (quad_sum*x_resp_num_da - x_resp_num*quad_sum_da)/quad_sum**2
        y_resp_da = (quad_sum*y_resp_num_da - y_resp_num*quad_sum_da)/quad_sum**2
        
        return x_resp_da,y_resp_da
        
    def slope(self,x_spot,y_spot):
        #give the slope of the quadcell response for a given spot position
        quad    = self.eval_quadrants(x_spot,y_spot,dx=0,dy=0)
        quad_dx = self.eval_quadrants(x_spot,y_spot,dx=1,dy=0)
        quad_dy = self.eval_quadrants(x_spot,y_spot,dx=0,dy=1)
        
        x_resp_dx,y_resp_dx = self.slope_from_quadrant(quad,quad_dx)
        x_resp_dy,y_resp_dy = self.slope_from_quadrant(quad,quad_dy)
        
        return x_resp_dx,x_resp_dy,y_resp_dx,y_resp_dy
        
    def angular_slope(self,x_spot,y_spot,focal_lenght,magnification=1):
        x_resp_dx,x_resp_dy,y_resp_dx,y_resp_dy = self.slope(x_spot,y_spot)
        
        x_angle = magnification*np.arctan2(x_spot,focal_lenght)
        y_angle = magnification*np.arctan2(x_spot,focal_lenght)
        
        x_resp_dx = x_resp_dx/x_spot*x_angle
        x_resp_dy = x_resp_dy/x_spot*x_angle
        y_resp_dx = y_resp_dx/y_spot*y_angle
        y_resp_dy = y_resp_dy/y_spot*y_angle
        
        return x_resp_dx,x_resp_dy,y_resp_dx,y_resp_dy
        
    def SNR(self,x_spot,y_spot,optical_power):
        ''' Return the quacel signal SNR
        x_spot, y_spot: postion of the spot on the quadcell in m or PSF postition units
        optical_power: total spot power, W
        
        '''
        
        A,B,C,D = self.eval_quadrants(x_spot,y_spot,dx=0,dy=0)
        all_quadrants = np.stack((A,B,C,D))
        
        all_power = all_quadrants*optical_power
        print('power',all_power[0][750][750])
        all_current = all_power*self.responsivity
        #print('Rqc',self.responsivity)
        #print('current',all_current[0][750][750])
        all_voltage = all_current*self.transimpedance
        #print('volts',all_voltage[0][750][750])
        all_shot_noise = self.transimpedance*np.sqrt(2*qe*all_current*self.bandwidth)
        all_amp_noise = self.amplifier_noise*np.sqrt(self.bandwidth)
        all_noise = np.sqrt(all_shot_noise**2+all_amp_noise**2)
        
        signal = sum([all_voltage[i,:,:] for i in range(4)])
        noise = np.sqrt(sum([all_noise[i,:,:]**2 for i in range(4)]))
        SNR = signal / noise
        
        return SNR
        
    def NEA(self,x_spot,y_spot,optical_power,focal_lenght,magnification=1):
        slopes = self.angular_slope(x_spot,y_spot,focal_lenght,magnification)
        slope = np.sqrt(sum([s**2 for s in slopes]))
        SNR = self.SNR(x_spot,y_spot,optical_power)
        
        NEA = 1/SNR/slope
        return NEA

class Photodiode:
    @initializer #automaticaly add agurment to each instances
    def __init__(self,gain=1,responsivity=0.5,bandwidth=1e6,excess_noise_factor=1,dark_current=0):
        pass
    
    def signal(self,optical_power):
        return self.gain*self.responsivity*optical_power
    
    def noise(self,optical_power):
        return np.sqrt(2*qe*self.excess_noise_factor*(self.signal(optical_power)+self.dark_current)*self.bandwidth)
        
    def SNR(self,optical_power):
        #Defined as (I/In)**2
        return (self.signal(optical_power)/self.noise(optical_power))**2
        
def BER_OOK(SNR):
    return 0.5*scsp.erfc(np.sqrt(SNR)/(2*np.sqrt(2)))
    
def BER_OOK_integrated(SNR,PDF):
    BER = scsp.erfc(np.sqrt(SNR)/(2*np.sqrt(2)))
    return 0.5*np.sum(PDF*BER, axis = 0)

# =====================================================================================================
# deprecated
# =====================================================================================================

def gaussian_PSF_on_square_mask(W,x1,x2,y1,y2):
    #W: spot size for the PSF, such as PSF(r) = 2/(pi*W**2)*exp(-2*r**2/w**2)
    coef = np.sqrt(2)/W
    return 0.25*(scsp.erf(coef*x2)-scsp.erf(coef*x1))*(scsp.erf(coef*y2)-scsp.erf(coef*y1))
    
def sampled_linear_interpolation_PSF_corner(r_samp,v_samp,n_int2d=1000,deg=3):
    r1 = r_samp[:-1]
    r2 = r_samp[1:]
    v1 = v_samp[:-1]
    v2 = v_samp[1:]
    normalize = np.pi*np.sum(r2**2*v2 + (r1**2+r1*r2+r2**2)*(v1-v2)/3 - r1**2*v1)
    v_samp = v_samp/normalize
    
    rf = r_samp[-1]
    x = np.linspace(-rf,rf,n_int2d)
    y = np.linspace(-rf,rf,n_int2d)
    xx, yy = np.meshgrid(x,y)
    r = np.sqrt(xx**2+yy**2)
    v = (r<=rf)*np.interp(r,r_samp,v_samp)
    
    v_integ = np.cumsum(np.cumsum(v,axis=0),axis=1)*(2*rf/n_int2d)**2

    v_integ_lookup = scit.RectBivariateSpline(x,y,v_integ,kx=deg,ky=deg)

    return v_integ_lookup
    
def sampled_2D_interpolation_PSF_corner(x_samp,y_samp,v_samp,n_int2d=1000,deg=3):
    normalize = np.sum(v_samp)*(x_samp[1]-x_samp[0])*(y_samp[1]-y_samp[0])
    v_samp = v_samp/normalize
    center_x = np.sum(v_samp*x_samp)/np.sum(v_samp)
    center_y = np.sum(v_samp*y_samp)/np.sum(v_samp)
    #print(v_samp.shape)
    
    x = np.linspace(x_samp[0],x_samp[-1],n_int2d)
    y = np.linspace(y_samp[0],y_samp[-1],n_int2d)
    xx, yy = np.meshgrid(x,y)
    v_samp_lookup = scit.RectBivariateSpline(x_samp,y_samp,v_samp,kx=1,ky=1)
    v = v_samp_lookup(x,y)
    
    v_integ = np.cumsum(np.cumsum(v,axis=0),axis=1)*((x_samp[-1]-x_samp[0])/n_int2d)*((y_samp[-1]-y_samp[0])/n_int2d)

    v_integ_lookup = scit.RectBivariateSpline(x,y,v_integ,kx=deg,ky=deg)

    return v_integ_lookup