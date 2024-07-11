import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpt

#----------------------------------------------------------
# Beam convertion functions

def half1e2_to_FWHM(half1e2:np.ndarray) -> np.ndarray:
    '''Convert a half angle or radius to a FWHM matching quantity'''
    import warnings
    warnings.warn("Was incorrect", UserWarning)
    FWHM = half1e2*np.sqrt(2*np.log(2))
    return FWHM

def FWHM_to_half1e2(FWHM:np.ndarray) -> np.ndarray:
    '''Convert a FWHM quantity to a matching half angle or radius'''
    import warnings
    warnings.warn("Was incorrect", UserWarning)
    half1e2 = FWHM/np.sqrt(2*np.log(2))
    return half1e2

def W0_to_radius1e2(W0:np.ndarray) -> np.ndarray:
    '''In this module, we assume W0 is the 1/e2 Gaussian beam radius'''
    radius1e2 = W0
    return radius1e2

def radius1e2_to_W0(radius1e2:np.ndarray) -> np.ndarray:
    '''In this module, we assume W0 is the 1/e2 Gaussian beam radius'''
    W0 = radius1e2
    return W0

def W0_to_diam1e2(W0:np.ndarray) -> np.ndarray:
    '''Convert W0 to the 1/e2 Gaussian beam diameter'''
    diam1e2 = 2*W0
    return diam1e2

def diam1e2_to_W0(diam1e2:np.ndarray) -> np.ndarray:
    '''Convert the 1/e2 Gaussian beam diameter to W0'''
    W0 = 0.5*diam1e2
    return W0

def W0_to_FWHMdiam(W0:np.ndarray) -> np.ndarray:
    '''Convert W0 to the Full Width Half Max Gaussian beam diameter'''
    FWHM_diam = half1e2_to_FWHM(W0)
    return FWHM_diam

def FWHMdiam_to_W0(FWHM_diam:np.ndarray) -> np.ndarray:
    '''Convert the Full Width Half Max Gaussian beam diameter to W0'''
    W0 = FWHM_to_half1e2(FWHM_diam)
    return W0

def W0_to_halfangle1e2(W0:np.ndarray, wavelength) -> np.ndarray:
    '''Convert W0 to the 1/e2 Max Gaussian beam half-angle, angles in radians and distances in meters.'''
    half_angle_1e2 = wavelength  / (np.pi * W0)
    return half_angle_1e2

def halfangle1e2_to_W0(half_angle_1e2:np.ndarray, wavelength) -> np.ndarray:
    '''Convert the 1/e2 Max Gaussian beam half-angle to W0, angles in radians and distances in meters.'''
    W0 = wavelength  / (np.pi * half_angle_1e2)
    return W0

def W0_to_fullangle1e2(W0:np.ndarray, wavelength) -> np.ndarray:
    '''Convert W0 to the 1/e2 Max Gaussian beam full angle, angles in radians and distances in meters.'''
    full_angle_1e2 = 2*W0_to_halfangle1e2(W0, wavelength)
    return full_angle_1e2

def fullangle1e2_to_W0(full_angle_1e2:np.ndarray, wavelength) -> np.ndarray:
    '''Convert the 1/e2 Max Gaussian beam full angle to W0, angles in radians and distances in meters.'''
    W0 = halfangle1e2_to_W0(full_angle_1e2/2, wavelength)
    return W0

def W0_to_FWHMangle(W0:np.ndarray, wavelength) -> np.ndarray:
    '''Convert W0 to the Full Width Half Max Gaussian beam angle, angles in radians and distances in meters.'''
    half_angle_1e2 = W0_to_halfangle1e2(W0, wavelength)
    FWHM_angle =  half1e2_to_FWHM(half_angle_1e2)
    return FWHM_angle

def FWHMangle_to_W0(FWHM_angle:np.ndarray, wavelength) -> np.ndarray:
    '''Convert the Full Width Half Max Gaussian beam angle to W0, angles in radians and distances in meters.'''
    half_angle_1e2 = FWHM_to_half1e2(FWHM_angle)
    W0 = halfangle1e2_to_W0(half_angle_1e2, wavelength)
    return W0

#----------------------------------------------------------
# Gaussian beam fits and centroids 

def ravel_I_x_y(I:np.ndarray, x:np.ndarray, y:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''Check if the I/x/y are already 1d, if not generate a mesh for x/y and flattens it'''

    # Check if the data is already 1D
    input_shape = I.shape
    is_1d = True
    if len(input_shape) > 1:
        for size in input_shape[1:]:
            if size > 1:
                is_1d = False
                break

    # if not 1d check dimmention, generate x,y arrays, ravel
    if not is_1d:
        assert I.shape[0] == x.shape[0]
        assert I.shape[1] == y.shape[0]
        if (I.shape == x.shape) and (I.shape == x.shape):
            xx = x
            yy = y
        else:
            xx,yy = np.meshgrid(y,x)
        I = I.flatten()
        x = xx.flatten()
        y = yy.flatten()

    return I,x,y

def centroid(I:np.ndarray, x:np.ndarray, y:np.ndarray, threshold:float=None) -> tuple[float,float]:
    '''Find the centroid of the I array
    I: N x M array
    x: Nx1 array, cooridnates
    y: Mx1 array, cooridnates
    threshold: centroid will ignore value below this. Default is 0.3 of I amplitude
    return centroid as tuple, for x and y axis, in x and y units'''

    I,x,y = ravel_I_x_y(I,x,y)

    # If no threhold is available 
    if threshold is None:
        min = np.min(I)
        max = np.max(I)
        threshold = min + 0.3*(max - min)

    mask = I > threshold
    weight = mask*I
    total = np.sum(weight)

    x_pondarated = weight * x
    x_tot = np.sum(x_pondarated)
    x_beam_center = x_tot/total

    y_pondarated = weight * y
    y_tot = np.sum(y_pondarated)
    y_beam_center = y_tot/total

    return x_beam_center,y_beam_center

def fit_2d_gaussian(I:np.ndarray, x:np.ndarray, y:np.ndarray, threshold:float=None):
    '''Fit a 2d Gaussian distribution to I.
    I: N x M array, data to fit
    x: Nx1 array, cooridnates
    y: Mx1 array, cooridnates
    Alternatively, shapes can be 1d N for I,x,y
    threshold: centroid will ignore value below this. Default is 0.2 of I amplitude
    return:
    amplitude in I unit, x0, y0 spot position, sigx, sigy, spot size in x/y units
    spot principal axis angle in radian'''

    I,x,y = ravel_I_x_y(I,x,y)

    x0_est, y0_est = centroid(I,x,y, threshold=threshold)

    x -= x0_est
    y -= y0_est

    # The model can only handle positive values
    min = np.min(I)
    assert min >= 0

    # If no threhold is available, set to 20%
    max = np.max(I)
    if threshold is None:
        threshold =  0.4*max

    # Remove based on threshold
    mask = I > threshold
    I = I[mask]
    x = x[mask]
    y = y[mask]

    if sum(mask) < 10: print(sum(mask))

    # Normalize
    I /= max
    angular_scale = np.maximum(np.max(x),np.max(y))
    x /= angular_scale
    y /= angular_scale

    # Log domain
    logI = -np.log(I)

    # varriance
    var = I # Flat noise

    # Find needed power of x,y in a very overcomplicated manner
    # Positon of power of x, y
    xmask = np.array([2, 0, 1, 1, 0, 0])
    ymask = np.array([0, 2, 1, 0, 1, 0])
    xmask = xmask[np.newaxis,:] + xmask[:,np.newaxis]
    ymask = ymask[np.newaxis,:] + ymask[:,np.newaxis]
    xx, yy = np.meshgrid(range(6),range(6))
    xx = xx.ravel()
    yy = yy.ravel()
    

    # Recoord coordinated for each needed power
    coord_dict = {}
    for xidx,yidx,xpow,ypow in zip(xx.ravel(), yy.ravel(), xmask.ravel(), ymask.ravel()):
        power = (xpow,ypow)
        coord = (xidx,yidx)
        if power in coord_dict: coord_dict[power].append(coord)
        else: coord_dict[power] = [coord]

    # Compute the sum of powers
    power_dict = {}
    for power in coord_dict.keys():
        power_dict[power] = np.sum((x**power[0])*(y**power[1])*var)

    # Place back the power in the LSQ matrix
    HtWH = np.zeros((6,6))
    for power, coord_list in coord_dict.items():
        value = power_dict[power]
        for coord in coord_list:
            HtWH[coord] = value
    HtWH[5,5] = np.sum(var)

    HtWY = np.array([
        np.sum(x**2*var*logI),
        np.sum(y**2*var*logI),
        np.sum(x*y*var*logI),
        np.sum(x*var*logI),
        np.sum(y*var*logI),
        np.sum(1*var*logI),
    ])[:,np.newaxis]

    # Solve the matrix inversion
    res = np.linalg.solve(HtWH, HtWY)

    px2, py2, pxy, px, py, p1 = res[:,0]

    # Main axis angle
    angle = 0.5*np.arctan2(pxy, py2-px2)

    # Spot size
    sig_p = px2 + py2
    sig_m = np.sqrt( (py2 - px2)**2 + pxy**2 )
    sigy2 = 2 / (sig_p + sig_m)
    sigx2 = 2 / (sig_p - sig_m)

    # Sport center
    div0 = 1/ (pxy**2 - 4*px2*py2)
    y0 = (2*py2*px - pxy*py) * div0
    x0 = (2*px2*py - pxy*px) * div0

    # Amplitude
    k = np.exp(- ( py2*px**2 - pxy*px*py + px2*py**2 )/( pxy**2 - 2*px2*py2 ) - p1)

    x0 = x0*angular_scale + x0_est
    y0 = y0*angular_scale + y0_est
    sigx2 *= angular_scale**2
    sigy2 *= angular_scale**2

    return k*max, x0, y0, sigx2, sigy2, angle

def plot_fit_2d_gaussian(ax:plt.Axes, I,x,y, k, x0, y0, sigx2, sigy2, angle, threshold=None):

    # Adapt fit param
    sigx = np.sqrt(sigx2)
    sigy = np.sqrt(sigy2)

    if threshold: I = I*(I>threshold)

    # Draw the raw data
    extent = [x[0],x[-1],y[-1],y[0]]
    ax.imshow(I,extent=extent, interpolation='quadric')

    # Cross lines
    x_vert = sigx*np.cos(angle)
    y_vert = sigx*np.sin(angle)
    x_line_vert = [x0 - x_vert, x0 + x_vert]
    y_line_vert = [y0 - y_vert, y0 + y_vert]
    ax.plot(x_line_vert, y_line_vert, color='r')
    x_hori = sigy*np.sin(angle)
    y_hori = -sigy*np.cos(angle)
    x_line_hori = [x0 - x_hori, x0 + x_hori]
    y_line_hori = [y0 - y_hori, y0 + y_hori]
    ax.plot(x_line_hori, y_line_hori, color='r')

    # 1/e2 Ellipse
    ellipse = mpt.Ellipse(xy=(x0,y0),width=2*sigx,height=2*sigy,angle=angle*180/np.pi,edgecolor='r', fc='None', lw=1)
    ax.add_patch(ellipse)