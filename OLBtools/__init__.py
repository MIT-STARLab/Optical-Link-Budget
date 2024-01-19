'''
Copyright © 2020 Paul Serra, Peter Grenfell, Ondrej cierny

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
The Software is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the Software.
'''

#----------------------------------------------------------
# Imports
from dataclasses import dataclass

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

def fwhw_to_radius1e2(fwhm):
    return fwhm/np.sqrt(2*np.log(2))
def diam1e2_to_fwhw(diam1e2):
    return np.sqrt(2*np.log(2))*diam1e2

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

def filter_maximum(wi, wc, n):
    # Smooth maximum based on the transfer function of order n butterworth filter 
    w = wi/wc
    ratio = 1 / np.sqrt(1 + w**(2*n))
    if n == 1: return wi*ratio
    else: return wi*ratio + wc*(1-ratio)
    
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
# Link budget functions
#----------------------------------------------------------

def path_loss_gaussian(beam_radius, wavelength, distance, rx_diameter, pointing_error=0, M2=1.0):
    # From Matlab linkbudget, author O cierny / P serra
    '''Calculates path loss given a Gaussian beam, and an Rx aperture at a
    specific distance. Assumes normalized input power, returns dB loss.
    All units in meters / radians.
    beam_radius: 1/e^2 radius of the beam at Tx [m]
    pointing_error: misalignment between Tx and Rx [rad] (optional)
    M2: M square beam quality factor, unitless (optional)'''

    # Calculate beam waist at the given distance due to diffraction [m]
    beam_radius_rx = beam_radius * np.sqrt(1 + ((M2*wavelength*distance)/(np.pi*beam_radius**2))**2)

    # Calculate distance from center of Gaussian due to pointing error [m]
    radial_distance = np.tan(pointing_error) * distance

    # Calculate irradiance using Gaussian formula [W/m^2]
    irradiance = (2/(np.pi*beam_radius_rx**2))*np.exp((-2*radial_distance**2)/beam_radius_rx**2)

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
    beam_radius = (wavelength * np.sqrt(2*np.log(2))) / (np.pi * fwhm)
    return beam_radius
    
def divergence_to_radius(divergence_1e2:np.ndarray, wavelength) -> np.ndarray:
    '''Calculates the 1/e^2 Gaussian beam radius given the 1e2 half-angle'''
    beam_radius = wavelength  / (np.pi * divergence_1e2)
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

@dataclass
class Quadcell:
    '''Class for a 4-quadrant pin detector, provides methods for postion, noise and noise-equivalent angle
    gap: size of the gap between quadrants, in m or PSF size units
    responsivity: photodiode repsonsivity in A/W
    transimpedance: amplifier gain in V/A or ohm
    amplifier_noise: output refered noise of the amplifier, in v/rtHz
    bandwidth: bandwidth for the detection, in Hz
        '''

    gap:float=0.0
    responsivity:float=0.9
    transimpedance:float=1e6
    amplifier_noise:float=1e-5

    def __post_init__(self):
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
    
@dataclass
class Photodiode:
    '''Class for an APD detector
    Uses dark current and excess noise factor to derive noise, and add NEP inquadrature if specified
    '''
   
    gain:float=1.0
    responsivity:float=0.9
    bandwidth:float=1e6
    excess_noise_factor:float=1.0
    dark_current:float=0
    amp_noise_density:float=0

    def estimatedNEP(self):
        """Find estimated NEP based on shot noise and dark current alone"""
        # We want to solve S (signal) for SNR = 1, noise = signal
        # S**2 = 2*qe*ENF*(S+dark)*BW + i_amp**2*BW
        # S**2 - K*S - K*dark - i_amp**2*BW = 0 with K = 2*qe*ENF*BW
        # S = (K + sqrt(K**2 + 4*(K*dark+i_amp**2*BW))) / 2
        K = 2*qe*self.excess_noise_factor*self.bandwidth
        signal = (K + np.sqrt(K**2 + 4*(K*self.dark_current+self.amp_noise_density**2*self.bandwidth))) / 2
        NEP = signal/(self.gain*self.responsivity) / np.sqrt(self.bandwidth)
        return NEP
    
    def signal(self,optical_power):
        return self.gain*self.responsivity*optical_power
    
    def noise(self,optical_power):
        APD_shot_noise_squared = 2*qe*self.excess_noise_factor*(self.signal(optical_power)+self.dark_current)*self.bandwidth
        amplifier_noise_squared = self.amp_noise_density**2*self.bandwidth
        return np.sqrt(APD_shot_noise_squared + amplifier_noise_squared)

        
    def SNR(self,optical_power):
        #Defined as I/In, In is the noise variance
        return self.signal(optical_power)/self.noise(optical_power)
    
    def supportedBandwidth(self,optical_power,required_SNR):
        # Noise RMS is Signal / SNR
        supported_noise = self.signal(optical_power)/required_SNR
        # Re-arange Photodiode.noise
        APD_shot_noise_density_squared = 2*qe*self.excess_noise_factor*(self.signal(optical_power)+self.dark_current)
        amplifier_noise_density_squared = self.amp_noise_density**2
        supported_bandwidth = supported_noise**2 / (APD_shot_noise_density_squared + amplifier_noise_density_squared)
        return supported_bandwidth
        
def BER_OOK(SNR):
    # With SNR defined as I/In
    # signal total amplituide is I, and variance of both 0 and 1 is asumed to In
    # Optimal thresholding, at 0 with both normal distribution at +/- I/2
    # 0 and 1 CDF(x): 0.5 * [1+ERF( (x +/- I/2)/(sqrt(2)*In) )]
    # BER = P(1)*P(0|1) + P(0)*P(1|0) = P(0|1) = 2*CDF(0)
    # BER = 0.5*[1+ERF( (-I/2)/(sqrt(2)*In) )]
    # BER = 0.5*[1-ERF( I/(In*2*sqrt(2)) )]
    # BER = 0.5*[ERFC( (I/In)/(2*sqrt(2)) )]
    return 0.5*scsp.erfc(SNR/(2*np.sqrt(2)))

def SNR_from_BER_OOK(BER):
    # With SNR defined as I/In
    # Inverse of BER_OOK
    # 2*BER = ERFC( (I/In)/(2*sqrt(2)) )
    # ERFCinv(2*BER) = (I/In)/(2*sqrt(2))
    # I/In = ERFCinv(2*BER)*2*sqrt(2)
    return scsp.erfcinv(2*BER)*2*np.sqrt(2)
    
def BER_OOK_integrated(SNR,PDF):
    BER = BER_OOK(SNR)
    BER_out_of_pdf = 0.5*( 1- np.sum(PDF, axis = 0) )
    return np.sum(PDF*BER, axis = 0) + BER_out_of_pdf

def suported_bandwidth_OOK(pd:Photodiode,optical_power,BER):

    #Inverse of BER_OOK:
    required_SNR = SNR_from_BER_OOK(BER)

    supported_bandwidth = pd.supportedBandwidth(optical_power,required_SNR)

    return supported_bandwidth


# =====================================================================================================
# Check against other link budget
# =====================================================================================================



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