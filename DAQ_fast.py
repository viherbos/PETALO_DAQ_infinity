import os
import pandas as pd
import tables as tb
import numpy as np
import multiprocessing as mp
from functools import partial
from SimLib import config_sim as CFG
#from SimLib import DAQ_infinity as DAQ
from SimLib import sipm_mapping as DAQ

""" HIGH LEVEL MODEL OF DAQ
    Output file is a HF5 based pandas structure with the following fields:
    MC           : Table with QDC > TE2 > TE1 (photons) event -> row // sipm -> column
    MC_tof       : Table with TDC (QDC > TE2 > TE1)     event -> row // sipm -> column
    iter1        : First iteration point for both gamma. Last 2 columns show closest SiPM
    subth_QDC_L1 : Accumulation of TE1 < QDC < TE2 per L1
    subth_TDC_L1 : Lowest TDC for channel with TE1 < QDC < TE2 per L1

    Configuration Parameters (JSON file):
    'MC_file_name':"full_ring_depth3cm_pitch4mm"
    'MC_out_file_name':"daq_output_IDEAL"
    'time_bin': 5
    'n_files' : 5
"""


class DAQ_MODEL(object):
    def __init__(self,path,jsonfilename,file_number):
        self.SIM_CONT = CFG.SIM_DATA(filename=path+jsonfilename+".json",read=True)
        self.path     = self.SIM_CONT.data['ENVIRONMENT']['path_to_files']
        self.in_file  = self.SIM_CONT.data['ENVIRONMENT']['MC_file_name']+"."+\
                        str(file_number)+".pet.h5"
        self.out_file = self.SIM_CONT.data['ENVIRONMENT']['MC_out_file_name']+"."+\
                        str(file_number)+\
                        ".h5"
        self.TE1      = self.SIM_CONT.data['TOFPET']['TE']
        self.TE2      = self.SIM_CONT.data['L1']['TE']
        self.time_bin = self.SIM_CONT.data['ENVIRONMENT']['time_bin']

        # TOPOLOGY
        self.L1, SiPM_I, SiPM_O, topology = DAQ.SiPM_Mapping(self.SIM_CONT.data,
                                                        self.SIM_CONT.data['L1']['map_style'])

        self.sipmtoL1 = np.zeros(topology['n_sipms'],dtype='int32')
        # Vector with SiPM assignment into L1
        L1_count = 0
        for i in self.L1:
            for j in i:
                for l in j:
                    self.sipmtoL1[l] = L1_count
            L1_count += 1

        self.waves    = np.array([])
        self.tof      = np.array([])
        self.extents  = np.array([])
        self.sensors  = np.array([])
        self.n_events = 0
        self.out_table = np.array([])
        self.out_table_tof = np.array([])
        self.sensors_t = np.array([])
        self.gamma1_i1 = np.array([])
        self.gamma2_i1 = np.array([])
        self.table = None
        self.h5file = None


    def read(self):
        os.chdir(self.path)

        self.waves   = np.array( pd.read_hdf(self.in_file,key='MC/waveforms'),
                            dtype = 'int32')
        self.tof     = np.array( pd.read_hdf(self.in_file,key='MC/tof_waveforms'),
                            dtype = 'int32')
        self.extents = np.array( pd.read_hdf(self.in_file,key='MC/extents'),
                            dtype = 'int32')
        self.n_events = self.extents.shape[0]

        self.sensors_t = np.array( pd.read_hdf(self.in_file,key='MC/sensor_positions'),
                            dtype = 'int32')

        self.sensors = self.sensors_t[:,0]
        self.sensors_order = np.argsort(self.sensors)
        self.sensors = self.sensors[self.sensors_order]

        self.h5file = tb.open_file(self.in_file, mode="r")
        self.table = self.h5file.root.MC.particles


    def write(self,iter=False):
        with pd.HDFStore(self.out_file,
                            complevel=9, complib='zlib') as store:
            panel_array = pd.DataFrame( data=self.out_table,
                                             columns=self.sensors)
            tof_array = pd.DataFrame( data=self.out_table_tof,
                                             columns=self.sensors)
            subth_QDC_L1 = pd.DataFrame( data=self.subth_QDC_L1)
            subth_TDC_L1 = pd.DataFrame( data=self.subth_TDC_L1)
            sipm2L1 = pd.DataFrame( data=self.sipmtoL1)

            sensors_xyz = np.array( pd.read_hdf(self.in_file,
                                        key='MC/sensor_positions'),
                                        dtype = 'float32')
            sensors_order = np.argsort(sensors_xyz[:,0])
            sensors_array = pd.DataFrame( data=sensors_xyz[sensors_order,:],
                                                columns=['sensor','x','y','z'])

            iter1_array = pd.DataFrame( data=np.concatenate((self.gamma1_i1,
                                                             self.gamma2_i1,
                                                             self.sipm_iter1A,
                                                             self.sipm_iter1B),
                                                             axis=1),
                                        columns=['x1','y1','z1','x2','y2','z2','SiPMA','SiPMB'])
            store.put('iter1',iter1_array)
            store.put('MC',panel_array)
            store.put('MC_tof',tof_array)
            store.put('sensors',sensors_array)
            store.put('sipm2L1',sipm2L1)
            store.put('subth_QDC_L1',subth_QDC_L1)
            store.put('subth_TDC_L1',subth_TDC_L1)
            store.close()



    def process(self):
        n_sensors = self.sensors.shape[0]
        self.out_table      = np.zeros((self.n_events,n_sensors),dtype='int32')
        self.out_table_tof  = np.zeros((self.n_events,n_sensors),dtype='int32')
        self.subth_QDC_L1   = np.zeros((self.n_events,len(self.L1)),dtype='int32')
        self.subth_TDC_L1   = np.zeros((self.n_events,len(self.L1)),dtype='int32')
        low_limit = 0
        low_limit_tof = 0
        count = 0
        count_a = 0

        for i in range(0,self.n_events):
            high_limit      = self.extents[i,1]
            high_limit_tof  = self.extents[i,2]
            event_wave = self.waves[low_limit:high_limit+1,:]
            event_tof  = self.tof[low_limit_tof:high_limit_tof+1,:]

            for j in self.sensors:
                condition   = (event_wave[:,0] == j)
                condition_tof = (event_tof[:,0] == -j)
                L1_index = self.sipmtoL1[j-1000]
                # Paola's style

                sensor_data = np.sum(event_wave[condition,2])

                if np.sum(condition_tof)==False:
                    sensor_data_tof = 0
                else:
                    sensor_data_tof = np.amin(event_tof[condition_tof,1])*self.time_bin

                if (sensor_data > self.TE2):
                    self.out_table[i,count_a]     = sensor_data
                    self.out_table_tof[i,count_a] = sensor_data_tof

                if ((sensor_data > self.TE1) and (sensor_data <= self.TE2)):
                    self.subth_QDC_L1[i,L1_index] = self.subth_QDC_L1[i,L1_index] + sensor_data
                    if self.subth_TDC_L1[i,L1_index]==0:
                        self.subth_TDC_L1[i,L1_index] = sensor_data_tof
                    else:
                        self.subth_TDC_L1[i,L1_index] = np.amin(np.array([sensor_data_tof,
                                                                    self.subth_TDC_L1[i,L1_index]]))

                count_a += 1

            low_limit = high_limit+1
            low_limit_tof = high_limit_tof+1
            count_a = 0


    def process_table(self):
        self.gamma1_i1 = np.zeros((self.n_events,3),dtype='float32')
        self.gamma2_i1 = np.zeros((self.n_events,3),dtype='float32')
        self.sipm_iter1A = np.zeros((self.n_events,1),dtype='int32')
        self.sipm_iter1B = np.zeros((self.n_events,1),dtype='int32')
        low_limit = 0
        count = 0
        count_a = 0

        for i in range(0,self.n_events):
            high_limit = self.extents[i,4]
            event_particles = self.table[low_limit:high_limit+1]

            cond1 = np.array(event_particles[:]['particle_name']=="e-")
            cond2 = np.array(event_particles[:]['initial_volume']=="ACTIVE")
            cond3 = np.array(event_particles[:]['final_volume']=="ACTIVE")
            cond4 = np.array(event_particles[:]['mother_indx']==1)
            cond5 = np.array(event_particles[:]['mother_indx']==2)


            A1 = event_particles[cond1 * cond2 * cond3 * cond4]
            A2 = event_particles[cond1 * cond2 * cond3 * cond5]

            if len(A1)==0:
                self.gamma1_i1[i,:] = np.zeros((1,3))
            else:
                A1_index = A1[:]['initial_vertex'][:,3].argmin()
                self.gamma1_i1[i,:] = A1[A1_index]['initial_vertex'][0:3]

            if len(A2)==0:
                self.gamma2_i1[i,:] = np.zeros((1,3))
            else:
                A2_index = A2[:]['initial_vertex'][:,3].argmin()
                self.gamma2_i1[i,:] = A2[A2_index]['initial_vertex'][0:3]

            low_limit = high_limit+1

            # Sensors close to first interaction
            if (self.gamma1_i1[i,:].all() == 0):
                self.sipm_iter1A[i,0] = 0
            else:
                self.sipm_iter1A[i,0] = np.argmin(np.sqrt(np.sum(np.square(self.sensors_t[:,1:]-self.gamma1_i1[i,:]),axis=1))) + 1000

            if (self.gamma2_i1[i,:].all() == 0):
                self.sipm_iter1B[i,0] = 0
            else:
                self.sipm_iter1B[i,0] = np.argmin(np.sqrt(np.sum(np.square(self.sensors_t[:,1:]-self.gamma2_i1[i,:]),axis=1))) + 1000



def DAQ_out(file_number,path,jsonfilename):

    TEST_c = DAQ_MODEL(path,jsonfilename,file_number)
    TEST_c.read()
    TEST_c.process()
    TEST_c.process_table()
    TEST_c.h5file.close()
    TEST_c.write(iter=True)



if __name__ == "__main__":

    kargs = {'path'         :"/home/viherbos/DAQ_DATA/NEUTRINOS/LESS_4mm/",
             'jsonfilename' :"OF_4mm_min"}
    SIM_JSON = CFG.SIM_DATA(filename=kargs['path']+kargs['jsonfilename']+".json",read=True)

    TRANS_map = partial(DAQ_out, **kargs)
    # Multiprocess Work
    # pool_size = mp.cpu_count() // 2
    # pool = mp.Pool(processes=pool_size)

    # Range of Files to Translate
    # pool.map(TRANS_map, [i for i in range(0,SIM_JSON.data['ENVIRONMENT']['n_files'])])


    # pool.close()
    # pool.join()

    DAQ_out(0,**kargs)
