#/usr/bin/env python
"""Commonly used utilities

    Function    
    ---------------
    obv_examiner(obv_df):
        Examine the input observational data
    
    throw_error(source, msg):
        Throw error with call source and error message

"""
import numpy as np
import logging

DEG2RAD=np.pi/180.0

def throw_error(source, msg):
    '''
    throw error and exit
    '''
    logging.error(source+msg)
    exit()

def write_log(msg, lvl=20):
    '''
    write logging log to log file
    level code:
        CRITICAL    50
        ERROR   40
        WARNING 30
        INFO    20
        DEBUG   10
        NOTSET  0
    '''

    logging.log(lvl, msg)

def wswd2uv(ws, wd):
    """ convert wind component to UV """
    WD_DIC={'N':  0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
            'E': 90.0, 'ESE':112.5, 'SE':135.0, 'SSE':157.5,
            'S':180.0, 'SSW':202.5, 'SW':225.0, 'WSW':247.5,
            'W':270.0, 'WNW':292.5, 'NW':315.0, 'NNW':337.5}
    
    # below test valid wind direction input
    wd_error=False
    try: 
        wd_int=int(wd)
        if wd_int>=0.0 and wd_int<=360.0:
            wd_rad=wd_int*DEG2RAD
        else:
            wd_error=True
    except ValueError:
        try:
            wd_rad=WD_DIC[wd]*DEG2RAD
        except KeyError:
            wd_error=True
    
    
    if wd_error:
        throw_error('utils.wswd2uv>>','Invalid wind direction input! '
                +'with record: wind_dir='+wd)

    u=-np.sin(wd_rad)*ws
    v=-np.cos(wd_rad)*ws
    
    return (u,v)

def wind_speed(u, v):
    """ calculate wind speed according to U and V """
    return np.sqrt(u*u+v*v)

def temp_prof(temp0, h0, p0, tgt_h, l):
    """ 
    calculate temperature at tgt_h according to
    temp0 at h0 and lapse rate l

    result converted to WRF perturbated potential T
    """
    TZERO=273.15
    BASE_T=300.0
    REF_P=100000.0 # 1000 hPa
    R=287.43
    CP=1005.0

    
    # change according lapse rate in Kelvin
    k=TZERO + temp0-l*(tgt_h-h0)/100.0
    
    # transfer to potential temp
    theta=k*np.power((REF_P/p0),R/CP)
    
    return theta - BASE_T

def wind_prof(ws0, h0, tgt_h, p):
    """ 
    calculate wind speed at tgt_h according to
    ws0 at h0 and exponent value p
    """
    return ws0*pow((tgt_h/h0), p)

def wind_prof_2d(ws0_2d, h0, tgt_h, p):
    """ 
    calculate wind speed at tgt_h according to
    ws0 at h0 and exponent value p, all data in 2D
    """
    return ws0_2d*(np.power((tgt_h/h0), p)).values


def get_closest_idx(a1d, val):
    """
        Find the nearest idx in 1-D array (a1d) according to a given val
    """
    
    dis=abs(val-a1d)
    return np.argwhere(dis==dis.min())[0][0]

def get_closest_idxy(lat2d, lon2d, lat0, lon0):
    """
        Find the nearest idx, idy in lat2d and lon2d for lat0 and lon0
    """
    dis_lat2d=lat2d-lat0
    dis_lon2d=lon2d-lon0
    dis=abs(dis_lat2d)+abs(dis_lon2d)
    idx=np.argwhere(dis==dis.min())[0].tolist() # x, y position
    return idx[0], idx[1]

def great_cir_dis_2d(lat0, lon0, lat2d, lon2d):
    """ Haversine formula to calculate great circle distance"""  
    R_EARTH=6371
    
    lat0_rad, lon0_rad = lat0*DEG2RAD, lon0*DEG2RAD
    lat2d_rad, lon2d_rad=lat2d*DEG2RAD, lon2d*DEG2RAD
    
    A=np.power(np.sin((lat2d_rad-lat0_rad)/2),2)
    B=np.cos(lat0_rad)*np.cos(lat2d_rad)*np.power(np.sin((lon2d_rad-lon0_rad)/2),2)

    return 2*R_EARTH*np.arcsin(np.sqrt(A+B))

def div_2d(uwnd, vwnd, dx, dy):
    """ 
        Calculate divergence on the rightmost 2 dims of uwnd and vwnd
        given dx and dy (in SI), in staggered mesh
    """
    
    (nz,ny,nx)=uwnd.shape
    nx=nx-1

    div=np.zeros((nz,ny,nx))
    div=(uwnd[:,:,1:nx+1]-uwnd[:,:,0:nx])/dx+(vwnd[:,1:ny+1,:]-vwnd[:,0:ny,:])/dy
    return div

def pad_var2d(var_org, direction, dim):
    (n_sn, n_we)=var_org.shape
    if dim==1:
        var=np.zeros((n_sn, n_we+1))
        if direction=='tail':
            var[:,0:n_we]=var_org
            var[:,n_we]=var_org[:,n_we-1]
        elif direction=='head':
            var[:,1:]=var_org
            var[:,0]=var_org[:,0]
    else:
        var=np.zeros((n_sn+1, n_we))
        if direction=='tail':
            var[0:n_sn,:]=var_org
            var[n_sn,:]=var_org[n_sn-1,:]
        elif direction=='head':
            var[1:,:]=var_org
            var[0,:]=var_org[0,:]

    return var

def pad_var3d(var_org, direction, dim):
    (n_z, n_sn, n_we)=var_org.shape
    if dim==2:
        var=np.zeros((n_z, n_sn, n_we+1))
        if direction=='tail':
            var[:,:,0:n_we]=var_org
            var[:,:,n_we]=var_org[:,:,n_we-1]
        elif direction=='head':
            var[:,:,1:]=var_org
            var[:,:,0]=var_org[:,:,0]
    else:
        var=np.zeros((n_z, n_sn+1, n_we))
        if direction=='tail':
            var[:,0:n_sn,:]=var_org
            var[:,n_sn,:]=var_org[:,n_sn-1,:]
        elif direction=='head':
            var[:,1:,:]=var_org
            var[:,0,:]=var_org[:,0,:]

    return var



