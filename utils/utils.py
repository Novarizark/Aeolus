#/usr/bin/env python
"""Commonly used utilities"""

import numpy as np

DEG2RAD=np.pi/180.0

def wswd2uv(ws, wd):
    """ convert wind component to UV """
    WD_DIC={'N':  0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
            'E': 90.0, 'ESE':112.5, 'SE':135.0, 'SSE':157.5,
            'S':180.0, 'SSW':202.5, 'SW':225.0, 'WSW':247.5,
            'W':270.0, 'WNW':292.5, 'NW':315.0, 'NNW':337.5}
    try: 
        wd=int(wd)
        if wd>=0.0 and wd<=360.0:
            wd_rad=wd*DEG2RAD
        else:
            print('2Error in wind direction input!!!'+' wd:'+wd)
            exit()
    except ValueError:
        wd_rad=WD_DIC[wd]*DEG2RAD
    except:
        print('1Error in wind direction input!!!'+ ' wd:'+wd)
        exit()

    u=-np.sin(wd_rad)*ws
    v=-np.cos(wd_rad)*ws
    return (u,v)

def wind_speed(u, v):
    """ calculate wind speed according to U and V """
    return np.sqrt(u*u+v*v)

def wind_prof(ws0, h0, tgt_h, p):
    """ 
    calculate wind speed at tgt_h according to
    ws0 at h0 and exponent value p
    """
    return ws0*pow((tgt_h/h0), p)

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
    
