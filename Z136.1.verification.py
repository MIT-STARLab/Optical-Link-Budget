import numpy as np

import matplotlib.pyplot as plt

import OLBtools.Z136 as z136

def C_A_verification():
    plt.figure()
    wavelength=np.linspace(400e-9,1400e-9,1000)
    plt.semilogy(wavelength, z136.C_A(wavelength))
    plt.xlim(400e-9,1400e-9)
    plt.ylim(0.8,12)

def C_B_verification():
    plt.figure()
    wavelength=np.linspace(400e-9,600e-9,1000)
    plt.semilogy(wavelength, z136.C_B(wavelength))
    plt.xlim(400e-9,600e-9)
    plt.ylim(0.8,1.2e3)

def C_C_verification():
    plt.figure()
    wavelength=np.linspace(1050e-9,1400e-9,1000)
    plt.semilogy(wavelength, z136.C_C(wavelength))
    plt.xlim(1050e-9,1400e-9)
    plt.ylim(0.8,1.2e6)

def MPE_verification():
    plt.figure()
    exposure_time = np.logspace(-13,1,1000)
    _, retina_mpe_850, _ = z136.ocular_MPE_point_source_NIR(850e-9, exposure_time)
    mpe_650 = z136.ocular_MPE_point_source_VIS(650e-9, exposure_time)
    plt.loglog(exposure_time,retina_mpe_850*exposure_time*1e-4)
    plt.loglog(exposure_time,mpe_650*exposure_time*1e-4)
    plt.xlim(1e-13,1e1)
    plt.ylim(1e-8,1e-1)


C_A_verification()
C_B_verification()
C_C_verification()
MPE_verification()
plt.show()