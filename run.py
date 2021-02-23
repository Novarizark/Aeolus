#/usr/bin/env python3
'''
Date: Feb 04, 2021
Aeolus is a wind diagnostic model to interpolate in-situ wind observations
onto WRF output template file.

This is the main script to drive the model

Revision:
Feb 19, 2021 --- MVP v0.01 completed
Feb 24, 2021 --- MVP v0.02 Unit test finished

Zhenning LI
'''

import numpy as np
import pandas as pd
import os

import lib 
import core

def main_run():
    
    print('*************************AEOLUS START*************************')
   
    # clock ticks
    time_mgr=lib.time_manager.time_manager()

    print('Read Config...')
    cfg_hdl=lib.cfgparser.read_cfg('./conf/config.ini')
    
    print('Read Input Observations...')
    obv_df=pd.read_csv(cfg_hdl['INPUT']['input_root']+cfg_hdl['INPUT']['input_obv'])
    
    print('Read Wind Profile Exponents...')
    wind_prof_df=pd.read_csv('./db/power_coef_wind.csv')
    
    time_mgr.toc('INPUT MODULE')

    print('Construct WRFOUT Handler...')
    fields_hdl=lib.preprocess_wrfinp.wrf_mesh(cfg_hdl)
    
    print('Construct Observation Satation Objs...')
    obv_lst=[] # all observations packed into a list
    for row in obv_df.itertuples():
        obv_lst.append(lib.obv_constructor.obv(row, wind_prof_df, cfg_hdl, fields_hdl))
    
    print('Construct Interpolating Estimator...')
    estimator=core.aeolus.aeolus(obv_lst,fields_hdl)
    
    time_mgr.toc('CONSTRUCTION MODULE')

    print('Aeolus Interpolating Estimator Casting...')
    estimator.cast(obv_lst,fields_hdl)
   
    time_mgr.toc('EXECUTION MODULE')

    print('Output Diagnostic UVW Fields...')
    core.aeolus.output_fields(cfg_hdl, estimator)
    
    time_mgr.toc('OUTPUT MODULE')
    
    time_mgr.dump()

    print('*********************AEOLUS ACCOMPLISHED*********************')


if __name__=='__main__':
    main_run()
