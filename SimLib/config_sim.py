import json
import os
import numpy as np
import sys
import pandas as pd




class SIM_DATA(object):

    # Only filenames are read. The rest is taken from json file
    def __init__(self,filename="sim_config.json",read=True):
        self.filename = filename
        self.data=[]

        if (read==True):
            self.config_read()
        else:
            # These are default values.
            # L1 output data frame = QDC[10] + TDC[10] + SiPM[20] = 40 bits
            self.data= {'ENVIRONMENT'  :{'ch_rate'     :1500E3,
                                        'temperature' :300,
                                        'path_to_files': "/home/viherbos/DAQ_DATA/NEUTRINOS/LESS_4mm/",
                                        'file_name': "p_FR_infinity_4mm_",
                                        'out_file_name':"daq_output",
                                        'n_files' : 7,
                                        'n_events': 1000},

                        'SIPM'        :{'size'        :[1,3,3]},

                        'TOPOLOGY'    :{'radius_int'   :994,
                                        'radius_ext'   :1294,
                                        'sipm_int_row':157,
                                        'sipm_ext_row':204,
                                        'n_rows'      :16},

                        'TOFPET'      :{'n_channels'  :64,
                                        'outlink_rate': (2.6E9/80)/2.0,
                                        # 80 bits per TOFPET output frame
                                        'IN_FIFO_depth':4,
                                        'OUT_FIFO_depth':64*4,
                                        'MAX_WILKINSON_LATENCY':5120,
                                        'TE':2,
                                        'TGAIN':1},

                        'L1'          :{'L1_outrate'    :650E6,
                                        'frame_process' :10E6,
                                        'FIFO_L1a_depth':1024,
                                        'FIFO_L1b_depth':256,
                                        'buffer_size'   :800,
                                        'n_asics'       :16,
                                        'TE'            :5 }
                       }

    def config_write(self):
        writeName = self.filename
        try:
            with open(writeName,'w') as outfile:
                json.dump(self.data, outfile, indent=4, sort_keys=False)
                print self.data
        except IOError as e:
            print(e)

    def config_read(self):
        try:
            with open(self.filename,'r') as infile:
                self.data = json.load(infile)
                print self.data
        except IOError as e:
            print(e)



if __name__ == '__main__':

    filename = "temp" #"infinity_4mm_16_2_5_buf800"
    SIM=SIM_DATA(filename = "/home/viherbos/DAQ_DATA/NEUTRINOS/LESS_4mm/"+filename+".json",
                 read = False)
    SIM.config_write()
