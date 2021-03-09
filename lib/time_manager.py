#/usr/bin/env python
"""Time manager obj to record execution time."""

import time

class time_manager:
    '''
    Time manager object to record execution time
    
    Attributes
    -----------
    tic0, float, absolute start time of the program
    tic, float, absolute start time before each module in runtime
    record, list[i]=(evt_str, dt), runtime duration of each individual event 

    Methods
    -----------
    toc(evt_str), press toc after each event (evt_str)
    dump(), dump time manager object in output stream

    '''

    def __init__(self):
        """construct time manager object"""
        self.tic0=time.time()
        self.tic=self.tic0
        self.record=[]
        self.loop_times=0

    def toc(self, evt_str, loop_flag=False):
        """press toc after each event (evt_str)"""
        if loop_flag:
            self.loop_times=self.loop_times+1
            self.record.append((evt_str,self.loop_times,time.time()-self.tic))
            self.tic=time.time()
        else:
            self.record.append((evt_str,1,time.time()-self.tic))
            self.tic=time.time()


    def dump(self):
        """Dump time manager object in output stream"""
        fmt='%20s:%10.4fs%6dx%6.1f%%'
        print('\n----------------TIME MANAGER PROFILE----------------\n\n')
        total_t=time.time()-self.tic0
        for rec in self.record:
            print(fmt % (rec[0],rec[2],rec[1],100.0*rec[2]/total_t))
        print(fmt % ('TOTAL ELAPSED TIME', total_t, 1, 100.0))
        print('\n----------------TIME MANAGER PROFILE----------------\n\n')
