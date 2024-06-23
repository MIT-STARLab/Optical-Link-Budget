from . import *
import numpy as np

import warnings
warnings.warn("This library does not come with any garanties of accuracy, please refer to Z136.1 directly for any safety-related or important determinations")


def C_A(wavelength:np.ndarray):
    C_a = 10**(0.002*(wavelength*1e9-700))
    return np.maximum(np.minimum(C_a,5.0),1.0)

def C_B(wavelength:np.ndarray):
    C_b = 10**(0.02*(wavelength*1e9-450))
    return np.maximum(C_b,1.0)

def C_C(wavelength:np.ndarray):
    C_c_1 =   10**(0.018*(wavelength*1e9-1150))
    C_c_2 = 8+10**(0.040*(wavelength*1e9-1250))
    mask = (wavelength < 1250e-9)
    C_c = C_c_2*(1-mask) + np.minimum(C_c_1,C_c_2)*mask
    return np.maximum(C_c,1.0)

def T_1(wavelength:np.ndarray):
    T_1 = 10*10**(0.02*(wavelength*1e9-450))
    return T_1

def K_lambda(wavelength:np.ndarray):
    K_l = 10**(0.01*(1400-wavelength*1e9))
    return K_l



def ocular_MPE_point_source_NIR(wavelength:np.ndarray=1064e-9,exposure_time:np.ndarray=30000):
    '''Based on Z136.1 table 5c, return MPE in W.m-2
    Valid between 700 and 1400 nm'''
    retina_MPE = 0
    cornea_MPE = 0

    C_a = C_A(wavelength)
    C_c = C_C(wavelength)
    K_l = K_lambda(wavelength)

    mask_0700_1050 = ( 700e-9 <= wavelength)*(wavelength < 1050e-9)
    mask_1050_1200 = (1050e-9 <= wavelength)*(wavelength < 1200e-9)
    mask_1200_1400 = (1200e-9 <= wavelength)*(wavelength < 1400e-9)
    mask_1050_1400 = mask_1050_1200 + mask_1200_1400

    mask_100fs_10ps = ( 1e-13 <= exposure_time)*(exposure_time <  1e-11)
    mask_10ps_5us   = ( 1e-11 <= exposure_time)*(exposure_time <  5e-06)
    mask_5us_10s    = ( 5e-06 <= exposure_time)*(exposure_time <     10)
    mask_10ps_13us  = ( 1e-11 <= exposure_time)*(exposure_time < 13e-06)
    mask_13us_10s   = (13e-06 <= exposure_time)*(exposure_time <     10)
    mask_10ps_1ms   = ( 1e-11 <= exposure_time)*(exposure_time <  1e-03)
    mask_1ms_4s     = ( 1e-03 <= exposure_time)*(exposure_time <      4)
    mask_4s_10s     = (     4 <= exposure_time)*(exposure_time <     10)
    mask_above_10s  = (exposure_time >= 10)
    
    retina_MPE += mask_0700_1050 * mask_100fs_10ps * 1e-7
    retina_MPE += mask_0700_1050 * mask_10ps_5us   * 2e-7*C_a
    retina_MPE += mask_0700_1050 * mask_5us_10s    * 1.8e-3*C_a*exposure_time**(0.75)

    retina_MPE += mask_1050_1400 * mask_100fs_10ps * 1e-7*C_c
    retina_MPE += mask_1050_1400 * mask_10ps_13us  * 2e-6*C_c
    retina_MPE += mask_1050_1400 * mask_13us_10s   * 9e-3*C_c*exposure_time**(0.75)

    retina_MPE /= exposure_time

    retina_MPE += mask_0700_1050 * mask_above_10s  * 1e-3*C_a
    retina_MPE += mask_1050_1400 * mask_above_10s  * 5e-3*C_c

    cornea_MPE += mask_1200_1400 * mask_10ps_1ms   * 0.3*K_l
    cornea_MPE += mask_1200_1400 * mask_1ms_4s     * 0.3*K_l + 0.56*exposure_time**(0.25) - 0.1
    cornea_MPE += mask_1200_1400 * mask_4s_10s     * 0.3*K_l + 0.7

    cornea_MPE /= exposure_time # convert to W

    cornea_MPE += mask_1200_1400 * mask_above_10s  * 0.03*K_l + 0.07

    cornea_MPE *= 1e4 # cm2 to m2
    retina_MPE *= 1e4

    MPE = np.minimum( (retina_MPE == 0)*1e20 + retina_MPE, (cornea_MPE == 0)*1e20 + cornea_MPE)
    MPE = (MPE < 1e19)*MPE
   
    return MPE, retina_MPE, cornea_MPE

def ocular_MPE_point_source_VIS(wavelength:np.ndarray=1064e-9,exposure_time:np.ndarray=30000):
    '''Based on Z136.1 table 5b, return MPE in W.m-2
    Valid between 400 and 700 nm'''
    all_MPE = 0

    C_b = C_B(wavelength)
    t_1 = T_1(wavelength)

    mask_400_450 = (400e-9 <= wavelength)*(wavelength < 450e-9)
    mask_450_500 = (450e-9 <= wavelength)*(wavelength < 500e-9)
    mask_500_700 = (500e-9 <= wavelength)*(wavelength < 700e-9)
    mask_400_500 = mask_400_450 + mask_450_500
    mask_400_700 = mask_400_500 + mask_500_700

    mask_100fs_10ps = ( 1e-13 <= exposure_time)*(exposure_time <  1e-11)
    mask_10ps_5us   = ( 1e-11 <= exposure_time)*(exposure_time <  5e-06)
    mask_5us_10s    = ( 5e-06 <= exposure_time)*(exposure_time <     10)
    mask_10s_100s   = (    10 <= exposure_time)*(exposure_time <    100)
    mask_10s_T1     = (    10 <= exposure_time)*(exposure_time <    t_1)
    mask_T1_100s    = (   t_1 <= exposure_time)*(exposure_time <    100)
    mask_above_100s = (exposure_time >= 100)

    all_MPE += mask_400_700 * mask_100fs_10ps * 1e-7
    all_MPE += mask_400_700 * mask_10ps_5us   * 2e-7
    all_MPE += mask_400_700 * mask_5us_10s    * 1.8e-3*exposure_time**(0.75)

    all_MPE += mask_400_450 * mask_10s_100s   * 1e-2
    all_MPE += mask_450_500 * mask_T1_100s    * 1e-2*C_b

    all_MPE /= exposure_time  # convert to W

    all_MPE += mask_400_500 * mask_above_100s * 1e-4*C_b
    all_MPE += mask_450_500 * mask_10s_T1     * 1e-3
    all_MPE += mask_500_700 * (mask_10s_100s+mask_above_100s) * 1e-3 

    all_MPE *= 1e4 # cm2 to m2

    return all_MPE


def limiting_aperture_VIS_NIR(wavelength:np.ndarray=1064e-9,exposure_time:np.ndarray=30000):
    mask_0400_1200 = ( 400e-9 <= wavelength)*(wavelength <  1200e-9)
    mask_1200_1400 = (1200e-9 <= wavelength)*(wavelength <= 1400e-9)
    mask_0400_1400 = mask_0400_1200 + mask_1200_1400

    mask_100fs_300ms = ( 1e-13 <= exposure_time)*(exposure_time < 0.3)
    mask_300ms_10s   = ( 0.3 <= exposure_time)*(exposure_time < 10)
    mask_above_10s   = (exposure_time >= 10)

    retina = mask_0400_1400 * 7.0

    cornea  = mask_1200_1400 * mask_100fs_300ms * 1.0
    cornea += mask_1200_1400 * mask_300ms_10s   * 1.5*exposure_time**0.375  # "Under normal conditions these exposure durations would not be used for hazard evaluation or classification." so for what?
    cornea += mask_1200_1400 * mask_above_10s   * 3.5

    skin   = mask_0400_1400 * 3.5

    return retina*1e-3, cornea*1e-3, skin*1e-3