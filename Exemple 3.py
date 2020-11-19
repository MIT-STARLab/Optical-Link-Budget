import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpc

import OLBtools as olb

elevation = olb.radians(20)  #20 degrees

# Orbits
altitude    = 500e3 # spacecraft altitude

# Transmit
P_avg =  2.50         # Transmit power laser, W
lambda_gl = 978e-9     # Laser 1 wavelength, m
beam_width_min = 1e-6  # beam width, FWMH radian
beam_width_max = 10e-3

# Receive
aperture = 95e-3    # aperture diameter, m

# Losses
pointing_error    = 5e-6 # radian
tx_system_loss = 3.00   # dB (10Log)
rx_system_loss = 3.00   # dB (10Log)

# Atmosphere
Cn2 = olb.Cn2_HV_57      #Hufnagel-valley 5/7 model

#----------------------------------------------------------
# LINK
#----------------------------------------------------------

zenith = np.pi/2-elevation
 
H = altitude
h_0 = 0

link_range = olb.slant_range(h_0,H,zenith,olb.Re)
r = np.tan(pointing_error)*link_range
beam_width = np.logspace(np.log10(beam_width_min),np.log10(beam_width_max),200)
W_0 = olb.fwhm_to_radius(beam_width,lambda_gl)
k = olb.angular_wave_number(lambda_gl)

range_loss = olb.path_loss_gaussian(W_0, lambda_gl, link_range, aperture, pointing_error)
all_losses = range_loss-tx_system_loss-rx_system_loss
Pe = P_avg*10**(all_losses/10)

Ps = np.logspace(-12,-1,200)
Psn = Ps[np.newaxis,:]
Pe = Pe[:,np.newaxis]
     


sig2_x, sig2_y = olb.get_scintillation_uplink_untracked_xy(h_0,H,zenith,k,W_0,Cn2,r)
sig2_x = sig2_x[:,np.newaxis]
sig2_y = sig2_y[:,np.newaxis]
alpha,mu,r =  olb.gamma_gamma_to_alpha_mu(sig2_x,sig2_y,orders=[2,3])

def plt_1_distrib(sx,sy,psv,pev,pnv):
    alpha,mu,r =  olb.gamma_gamma_to_alpha_mu(sx,sy,orders=[2,3])
    cdfs = olb.alpha_mu_cdf(alpha,mu,r,pev,pnv)
    cdfst = cdfs.transpose()
    
    p50 = olb.alpha_mu_inv_cdf(alpha,mu,r,pev,0.50)
    p90 = olb.alpha_mu_inv_cdf(alpha,mu,r,pev,0.90)
    p99 = olb.alpha_mu_inv_cdf(alpha,mu,r,pev,0.99)
    
    if 1:
        plt.figure()
        plt.title('$P(I>I_{th})$, with $C_n^2 = 10 \\times HV_{5/7}$')
        plt.yscale('log')
        plt.xscale('log')
        plt.pcolormesh(beam_width*1e3,psv, 1-cdfst, cmap = 'RdYlBu')
        plt.xlabel('Divergence, mrad')
        plt.ylabel('Received power $I_{th}$, W')
        plt.plot(beam_width*1e3,pev,color='black',label='Without scintillation')
        plt.plot(beam_width*1e3,p50,color='black',linestyle=':', label='50% confidence')
        plt.plot(beam_width*1e3,p90,color='black',linestyle='--',label='90% confidence')
        plt.plot(beam_width*1e3,p99,color='black',linestyle='-.',label='99% confidence')
        plt.ylim([psv[0],psv[-1]])
        plt.legend()
        plt.colorbar()

if 1:
    plt_1_distrib(sig2_x,sig2_y,Ps,Pe,Psn)
    plt.title('$P(I>I_{th})$, uplink, 20 deg elevation')
    plt.legend()

plt.show()