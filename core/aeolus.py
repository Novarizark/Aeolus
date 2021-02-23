#/usr/bin/env python
"""
Core Component: Aeolus Interpolator
    Classes: 
    -----------
        aeolus: core class, domain interpolator

    Functions:
    -----------
        diag_vert_vel(aeolus, fields_hdl): Diagnose vertical velocity according to divergence in eta coordinate
        inv_dis_wgt_2d(ws_lst, dis_mtx): Inverse Distance Weighting (IDW) Interpolation
        output_fields(cfg, aeolus): Output diagnostic fields into WRF template file
"""

import xarray as xr
import numpy as np

import sys, os
sys.path.append('../')
from utils import utils

print_prefix='core.aeolus>>'

class aeolus:

    '''
    Aeolus interpolator, interpolate in-situ obvs onto wrf mesh 
    
    Attributes
    -----------
    dis_mtx_u(n_sn, n_we_stag, n_obv), float, distance between obv and grid point, matrix on staggered u grid
    dis_mtx_v(n_sn_stag, n_we, n_obv), float, distance between obv and grid point, matrix on staggered v grid
    U(bottom_top,n_sn, n_we_stag), float, interpolated U
    V(bottom_top,n_sn_stag, n_we), float, interpolated V
    W(bottom_top,n_sn_stag, n_we), float, diagnostic W

    Methods
    -----------
    cast(), cast interpolation on WRF mesh
    adjust_mass(), adjust result according to continuity equation (mass conservation)
    '''
    
    def __init__(self, obv_lst, fields_hdl):
        """ construct aeolus interpolator """
        
        n_obv=len(obv_lst)
         
        # construct distance matrix on staggered u grid
        self.dis_mtx_u=np.zeros((fields_hdl.n_sn, fields_hdl.n_we+1, n_obv))
        
        # construct distance matrix on staggered v grid
        self.dis_mtx_v=np.zeros((fields_hdl.n_sn+1, fields_hdl.n_we, n_obv))
        
        for idx, obv in enumerate(obv_lst):
            self.dis_mtx_u[:,:,idx]=utils.great_cir_dis_2d(obv.lat, obv.lon, fields_hdl.XLAT_U, fields_hdl.XLONG_U)
            self.dis_mtx_v[:,:,idx]=utils.great_cir_dis_2d(obv.lat, obv.lon, fields_hdl.XLAT_V, fields_hdl.XLONG_V)
        self.U=fields_hdl.U
        self.V=fields_hdl.V
        self.W=fields_hdl.W
   
    def cast(self, obv_lst, fields_hdl):
        """ cast interpolation on WRF mesh """
        print(print_prefix+'Interpolate UV...')
        u_profs=np.array([obv.u_prof for obv in obv_lst])
        v_profs=np.array([obv.v_prof for obv in obv_lst])
        for idz in range(0, u_profs.shape[1]):
            self.U.values[idz,:,:]=inv_dis_wgt_2d(u_profs[:,idz], self.dis_mtx_u)
        
        for idz in range(0, v_profs.shape[1]):
            self.V.values[idz,:,:]=inv_dis_wgt_2d(v_profs[:,idz], self.dis_mtx_v)

        print(print_prefix+'First-guess W...')
        self.W.values=diag_vert_vel(self, fields_hdl)
        
        print(print_prefix+"Adjust results by mass conservation...")
        self.adjust_mass(fields_hdl)

    def adjust_mass(self, fields_hdl):
        """ adjust result according to continuity equation (mass conservation) """
        pass

def diag_vert_vel(aeolus, fields_hdl):
    """ Diagnose vertical velocity according to divergence in eta coordinate"""
    dx, dnw, ter, ztop=fields_hdl.dx, fields_hdl.dnw, fields_hdl.ter, fields_hdl.ztop
    n_sn, n_we=fields_hdl.n_sn, fields_hdl.n_we
    n_z=dnw.shape[0]
    U=aeolus.U.values
    V=aeolus.V.values
    W=aeolus.W.values
    # bottom boundary condition for vertical velocity
    W[0,:,:]=0.0

    ter_xstag=np.zeros(U.shape[1:])
    ter_xstag[:,0:n_we]=ter
    ter_xstag[:,n_we]=ter[:,n_we-1]
    ter_ystag=np.zeros(V.shape[1:])
    ter_ystag[0:n_sn,:]=ter
    ter_ystag[n_sn,:]=ter[n_sn-1,:]
    
    div=utils.div_2d(U, V, dx, dx)

    # eta coordinate adjustment terms, see (4.9') in Magnusson.pdf
    for ii in range(0,n_z):
        adj_term1=(1.0/(ztop-ter))*U[ii,:,0:n_we]*((ter_xstag[:,1:]-ter_xstag[:,0:n_we])/dx)
        adj_term2=(1.0/(ztop-ter))*V[ii,0:n_sn,:]*((ter_ystag[1:,:]-ter_ystag[0:n_sn,:])/dx)
        W[ii+1,:,:]=(adj_term1+adj_term2-div[ii,:,:])*dnw[ii]+W[ii,:,:]

    return W

def inv_dis_wgt_2d(ws_lst, dis_mtx):
    """ Inverse Distance Weighting (IDW) Interpolation """
    
    (n_sn,n_we)=dis_mtx[:,:,0].shape
    ws_2d=np.zeros((n_sn,n_we))
    
    wgt_mtx=np.power(dis_mtx, -2)
    ws_2d=np.dot(wgt_mtx, ws_lst)/np.sum(wgt_mtx, axis=2)
    
    return ws_2d



def output_fields(cfg, aeolus):
    """ Output diagnostic fields into WRF template file """
    template_fn=cfg['INPUT']['input_root']+cfg['INPUT']['input_wrf']
    out_fn=cfg['OUTPUT']['output_root']+cfg['OUTPUT']['output_prefix']+'.'+cfg['INPUT']['input_wrf']+'.'+cfg['OUTPUT']['output_fmt']

    os.system('cp '+template_fn+' '+out_fn)
    
    ds=xr.open_dataset(out_fn) 
    ds['U'].values[0,:,:,:]=aeolus.U.values
    ds['V'].values[0,:,:,:]=aeolus.V.values
    ds['W'].values[0,:,:,:]=aeolus.W.values
    ds.to_netcdf(out_fn, mode='a')

if __name__ == "__main__":
    pass
