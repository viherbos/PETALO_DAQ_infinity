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
            self.data= {'ENVIRONMENT'  :{'ch_rate'     :10E6,
                                        'temperature' :300,
                                        'path_to_files': "/home/viherbos/DAQ_DATA/NEUTRINOS/LESS_4mm/",
                                        'file_name': "p_FR_infinity_4mm_",
                                        'MC_file_name':"full_ring_depth3cm_pitch4mm",
                                        'out_file_name':"daq_output",
                                        'MC_out_file_name':"daq_output_IDEAL",
                                        'time_bin': 5,
                                        'n_files' : 5,
                                        'n_events': 30000},

                        'SIPM'        :{'size'        :[1,3,3]},

                        'TOPOLOGY'    :{'radius_int'   :994,
                                        'radius_ext'   :1294,
                                        'sipm_int_row':157,
                                        'sipm_ext_row':204,
                                        'n_rows'      :16},

                        'TOFPET'      :{'n_channels'  :64,
                                        'outlink_rate': (2.6E9/80)/1.0,
                                        # 80 bits per TOFPET output frame
                                        'IN_FIFO_depth':4,
                                        'OUT_FIFO_depth':64*4,
                                        'MAX_WILKINSON_LATENCY':5120,
                                        'TE':3,
                                        'TGAIN':1},

                        'L1'          :{'L1_outrate'    :550E6,
                                        'frame_process' :3E6,
                                        'FIFO_L1a_depth':4096,
                                        'FIFO_L1b_depth':128,
                                        'buffer_size'   :512,
                                        'n_asics'       :16,
                                        'TE'            :9,
                                        'map_style'     :'striped_2',
                                        'L1_mapping_I'  :[8,8,8,8,8],
                                        'L1_mapping_O'  :[7,7,8,7,7,7,8]}
                       }

# 'L1_mapping_I'  :[10,10,10,10],
# 'L1_mapping_O'  :[10,10,11,10,10]}
# 'L1_mapping_I'  :[5,5,5,5,5,5,5,5],
# 'L1_mapping_O'  :[6,7,6,7,6,6,7,6]}

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

    filename = "extreme_10M_1"
    SIM=SIM_DATA(filename = "/home/viherbos/DAQ_DATA/NEUTRINOS/LESS_4mm/"+filename+".json",
                 read = False)
    SIM.config_write()
