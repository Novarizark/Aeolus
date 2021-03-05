#/usr/bin/env python3
'''
Date: Feb 04, 2021
Aeolus is a wind diagnostic model to interpolate in-situ wind observations
onto WRF output template file.

This is the main script to drive the model

Revision:
Feb 19, 2021 --- MVP v0.01 completed
Feb 24, 2021 --- MVP v0.02 Unit test finished
Mar 10, 2021 --- MVP v0.10 Unit test finished
    --- Adevection Distance Adjustement
    --- Wind 16 directions and degree directions
    --- Interpolation of Nearest 3 stations.

Zhenning LI
'''

import numpy as np
import pandas as pd
import os

import lib 
import core

def main_run():
    
    print('*************************AEOLUS START*************************')
   
    # wall-clock ticks
    time_mgr=lib.time_manager.time_manager()

    print('Read Config...')
    cfg_hdl=lib.cfgparser.read_cfg('./conf/config.ini')
    
    print('Read Input Observations...')
    obv_df=pd.read_csv(cfg_hdl['INPUT']['input_root']+cfg_hdl['INPUT']['input_obv'])
    # make sure the list is sorted by datetime
    obv_df=obv_df.sort_values(by='yyyymmddhhMM') 
    
    print('Read Wind Profile Exponents...')
    wind_prof_df=pd.read_csv('./db/power_coef_wind.csv')
    
    time_mgr.toc('INPUT MODULE')

    print('Construct WRFOUT Handler...')
    fields_hdl=lib.preprocess_wrfinp.wrf_mesh(cfg_hdl)
    
    print('Construct Observation Satation Objs...')
   
    print('Construct Model Clock...')
    clock=lib.model_clock.model_clock(cfg_hdl)

    print('Construct Interpolating Estimator...')
    estimator=core.aeolus.aeolus(fields_hdl)

    time_mgr.toc('CONSTRUCTION MODULE')

    print('Aeolus Interpolating Estimator Casting...')
    while not(clock.done):
        # Memory Refresh BUG: Required to reconstruct obv_lst in each clock period
        obv_lst=[] 
        for row in obv_df.itertuples():
            obv_lst.append(lib.obv_constructor.obv(row, wind_prof_df, cfg_hdl, fields_hdl))
    
        estimator.cast(obv_lst,fields_hdl, clock)

        print('Output Diagnostic UVW Fields...')
        core.aeolus.output_fields(cfg_hdl, estimator, clock)
        
        clock.advance()

    time_mgr.toc('EXECUTION MODULE')
    
    time_mgr.dump()

    print('*********************AEOLUS ACCOMPLISHED*********************')


if __name__=='__main__':
    main_run()
