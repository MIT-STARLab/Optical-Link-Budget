import numpy as np
import scipy.special as scsp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpc

import OLBtools as olb

# Objectives
elevation_min = olb.radians(00)  #20 degrees
elevation_max = olb.radians(90)  #20 degrees

# Orbits
altitude    = 500e3 # spacecraft altitude

# Transmit
P_avg =  0.050         # Transmit power laser, W
lambda_gl = 915e-9     # Laser 1 wavelength, m
beam_width = 200e-6    # beam width, FWMH radian

# Receive
aperture = 600e-3       # Aperture diameter, m
aperture_scaling = 0.60 # Fraction of clear aperture

# Detector
Responsivity = 50        # A.W-1
Fn_apd = 3.2             # Excess noise factor @M=100
i_dark_apd = 1.5e-9

# Losses
pointing_error = 50e-6   # radian
tx_system_loss = 3.00    # dB (10Log)
rx_system_loss = 3.00    # dB (10Log)

# Atmosphere
Cn2 = olb.Cn2_HV_57      # Hufnagel-valley 5/7 model

#----------------------------------------------------------

#500 poitnts, from specified minimum elevation to top.
elevation = np.linspace(elevation_min,elevation_max,500)
zenith = np.pi/2-elevation
 
H = altitude
h_0 = 0

# Link range for elevation range
link_range = olb.slant_range(h_0,H,zenith,olb.Re)

# Misspointing as distance at receiver
r_s = np.tan(pointing_error)*link_range

# 1/e2 beam radius
W_0 = olb.fwhm_to_radius(beam_width,lambda_gl)

# Wavenumber
k = olb.angular_wave_number(lambda_gl)

range_loss = olb.path_loss_gaussian(W_0, lambda_gl, link_range, aperture, pointing_error)

all_losses = range_loss-tx_system_loss-rx_system_loss

# Expected value of the power at he receiver
Pe = aperture_scaling*P_avg*10**(all_losses/10)
Pe = Pe[:,np.newaxis] #on second dim

# Scintillation
sig2_x, sig2_y = olb.get_scintillation_downlink_xy(h_0,H,zenith,k,W_0,Cn2)
sig2_x = sig2_x[:,np.newaxis]
sig2_y = sig2_y[:,np.newaxis]

# Power distribution coeficients
alpha,mu,r =  olb.gamma_gamma_to_alpha_mu(sig2_x,sig2_y,orders=[2,3])

# Power log scale, must include 1% to 99% cumulated power
Hv = np.ceil(np.log10(olb.alpha_mu_inv_cdf(alpha,mu,r,Pe,0.01).max()))
Hl = np.floor(np.log10(olb.alpha_mu_inv_cdf(alpha,mu,r,Pe,0.99).min()))
Ps = np.logspace(Hl,Hv,500)
Psn = Ps[np.newaxis,:]#on second dim
# Sacle interval centers
pdfPs = (Ps[1:]+Ps[:-1])/2
dPs = Ps[1:]-Ps[:-1]
dPs = dPs[:,np.newaxis]

# Cumulative distribution function
cdfs = olb.alpha_mu_cdf(alpha,mu,r,Pe,Psn)
cdfst = cdfs.transpose()

# Prpablility densisty function, over log scale intervals
pdfst = cdfst[1:,:]-cdfst[:-1,:]


pdfPs = pdfPs[:,np.newaxis]

pd = olb.Photodiode(
    gain=1,
    responsivity=Responsivity,
    bandwidth=10e6,
    excess_noise_factor=Fn_apd,
    dark_current=i_dark_apd)
SNR_10 = pd.SNR(pdfPs)
BER_10 = olb.BER_OOK_integrated(SNR_10,pdfst)

pd = olb.Photodiode(
    gain=1,
    responsivity=Responsivity,
    bandwidth=100e6,
    excess_noise_factor=Fn_apd,
    dark_current=i_dark_apd)
SNR_100 = pd.SNR(pdfPs)
BER_100 = olb.BER_OOK_integrated(SNR_100,pdfst)

pd = olb.Photodiode(
    gain=1,
    responsivity=Responsivity,
    bandwidth=1000e6,
    excess_noise_factor=Fn_apd,
    dark_current=i_dark_apd)
SNR_1000 = pd.SNR(pdfPs)
BER_1000 = olb.BER_OOK_integrated(SNR_1000,pdfst)

if 1:
    plt.figure()
    plt.yscale('log')
    plt.axhline(1e-5,color='k',linestyle='--',label='$BER = 10^{-5}$')
    plt.plot(olb.degrees(elevation),BER_10,label='10 MHz detector')
    plt.plot(olb.degrees(elevation),BER_100,label='100 MHz detector')
    plt.plot(olb.degrees(elevation),BER_1000,label='1 GHz detector')
    plt.xlabel('Elevation, degrees')
    plt.axvline(30,color='red')
    plt.ylabel('BER')
    plt.xlim(0,90)
    plt.ylim(1e-15,1)
    plt.legend()
    plt.title('BER, 3dB')

def plt_1_distrib(sx,sy,psv,pev,pnv):
    alpha,mu,r =  olb.gamma_gamma_to_alpha_mu(sx,sy,orders=[2,3])
    cdfs = olb.alpha_mu_cdf(alpha,mu,r,pev,pnv)
    cdfst = cdfs.transpose()
    
    p01 = olb.alpha_mu_inv_cdf(alpha,mu,r,pev,0.01)
    p10 = olb.alpha_mu_inv_cdf(alpha,mu,r,pev,0.10)
    p50 = olb.alpha_mu_inv_cdf(alpha,mu,r,pev,0.50)
    p90 = olb.alpha_mu_inv_cdf(alpha,mu,r,pev,0.90)
    p99 = olb.alpha_mu_inv_cdf(alpha,mu,r,pev,0.99)
    
    plt.figure()
    plt.yscale('log')
    plt.pcolormesh(olb.degrees(elevation),psv, 1-cdfst, cmap = 'RdYlBu')
    plt.xlabel('Elevation, degrees')
    plt.ylabel('Received power $I_{th}$, W')
    #plt.plot(olb.degrees(elevation),pev,color='black',label='Without scintillation')
    plt.plot(olb.degrees(elevation),p01,color='black',linestyle='-.',label='1% confidence')
    plt.plot(olb.degrees(elevation),p10,color='black',linestyle='--',label='10% confidence')
    plt.plot(olb.degrees(elevation),p50,color='black',linestyle=':', label='50% confidence')
    plt.plot(olb.degrees(elevation),p90,color='black',linestyle='--',label='90% confidence')
    plt.plot(olb.degrees(elevation),p99,color='black',linestyle='-.',label='99% confidence')
    plt.ylim([psv[0],psv[-1]])
    plt.clim([0,1])
    plt.legend(loc=4)
    plt.colorbar()

if 1:
    plt_1_distrib(sig2_x,sig2_y,Ps,Pe,Psn)
    plt.title('$P(I>I_{th})$, 3dB')
    plt.ylim([Ps[0],Ps[-1]])
    
plt.show()