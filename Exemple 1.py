import numpy as np
import OLBtools as olb

# Transmit
P_tx = 200e-3           # Transmit power laser, W
lambda_gl = 1550e-9     # Laser 1 wavelenght, m
beam_width = 15e-6      # beam width, FWMH radian
pointing_error = 5e-6   # radian
tx_system_loss = 3      # dB

# Receive
apperture = 95e-3       # Apperture diameter, m
rx_system_loss = 3      # dB

link_range = 1000e3     # Link distance, m

# position error at receiver
r = np.tan(pointing_error)*link_range

# beam waist
W_0 = olb.fwhm_to_radius(beam_width,lambda_gl)

# Angular wave number, = 2*pi/lambda
k = olb.angular_wave_number(lambda_gl)

range_loss = olb.path_loss_gaussian(W_0, lambda_gl, link_range, apperture, pointing_error)

all_losses = range_loss-tx_system_loss-rx_system_loss

P_rx = P_tx*10**(all_losses/10)

print('Received power: %.3f uW' % (P_rx*1e6))