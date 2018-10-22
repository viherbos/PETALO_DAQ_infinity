import os
import time
import pandas as pd
import tables as tb
import numpy as np
import multiprocessing as mp
from functools import partial
from SimLib import config_sim as CFG
#from SimLib import DAQ_infinity as DAQ
from SimLib import sipm_mapping as DAQ
from SimLib import Encoder_tools as ENC

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
    'AUTOENCODER_file_name': Weigths and Biasing matrix
    'time_bin': 5
    'n_files' : 5
"""


class DAQ_MODEL(object):
    def __init__(self,path,jsonfilename,file_number,encoder_data):
        self.SIM_CONT = CFG.SIM_DATA(filename=path+jsonfilename+".json",read=True)
        self.path     = self.SIM_CONT.data['ENVIRONMENT']['path_to_files']
        self.in_file  = self.SIM_CONT.data['ENVIRONMENT']['MC_file_name']+"."+\
                        str(file_number).zfill(3)+".pet.h5"
        self.out_file = self.SIM_CONT.data['ENVIRONMENT']['MC_out_file_name']+"."+\
                        str(file_number).zfill(3)+\
                        ".h5"
        self.TE1      = self.SIM_CONT.data['TOFPET']['TE']
        self.TE2      = self.SIM_CONT.data['L1']['TE']
        self.time_bin = self.SIM_CONT.data['ENVIRONMENT']['time_bin']
        self.n_rows   = self.SIM_CONT.data['TOPOLOGY']['n_rows']
        self.n_L1     = len(self.SIM_CONT.data['L1']['L1_mapping_I'])+\
                        len(self.SIM_CONT.data['L1']['L1_mapping_O'])
        # TOPOLOGY
        self.L1, SiPM_I, SiPM_O, topology = DAQ.SiPM_Mapping(self.SIM_CONT.data,
                                                        self.SIM_CONT.data['L1']['map_style'])

        self.sipmtoL1 = np.zeros(topology['n_sipms'],dtype='int32')

        # Vector with SiPM assignment into L1
        # Number of SiPMs per L1
        L1_count = 0
        for i in self.L1:
            for j in i:
                for l in j:
                    self.sipmtoL1[l] = L1_count
            L1_count += 1


        self.COMP = encoder_data

        self.waves    = np.array([])
        self.tof      = np.array([])
        self.extents  = np.array([])
        self.sensors  = np.array([])
        self.n_events = 0
        self.out_table = np.array([])
        self.out_table_tof = np.array([])

        self.data_enc = np.array([])
        self.data_recons = np.array([])

        self.sensors_t = np.array([])
        self.gamma1_i1 = np.array([])
        self.gamma2_i1 = np.array([])
        self.table = None
        self.h5file = None



    def read_files(self):

        os.chdir(self.path)
        self.waves   = np.array( pd.read_hdf(self.in_file,key='MC/waveforms'),
                            dtype = 'int32')
        self.tof     = np.array( pd.read_hdf(self.in_file,key='MC/tof_waveforms'),
                            dtype = 'int32')
        self.extents = np.array( pd.read_hdf(self.in_file,key='MC/extents'),
                            dtype = 'int32')
        self.sensors_t = np.array( pd.read_hdf(self.in_file,key='MC/sensor_positions'),
                            dtype = 'int32')

        self.sensors = self.sensors_t[:,0]
        self.sensors_order = np.argsort(self.sensors)
        self.sensors = self.sensors[self.sensors_order]
        self.h5file = tb.open_file(self.in_file, mode="r")
        self.table = self.h5file.root.MC.particles

        self.events_infile  = self.extents.shape[0]
        self.n_sensors      = self.sensors.shape[0]

        # Empty out matrices
        self.out_table_B     = np.array([]).reshape(0,self.n_sensors)
        self.out_table_tof_B = np.array([]).reshape(0,self.n_sensors)
        self.data_recons_B   = np.array([]).reshape(0,self.n_sensors)
        self.data_enc_B   = np.array([]).reshape(0,self.SIM_CONT.data['L1']['enc_out_len'])
        self.subth_QDC_L1_B  = np.array([]).reshape(0,len(self.L1))
        self.subth_TDC_L1_B  = np.array([]).reshape(0,len(self.L1))


    def read_data(self,event_array):

        self.n_events = len(event_array) #self.extents.shape[0]
        # Create empty matrixs for current file prrocessing

        self.out_table      = np.zeros((self.n_events,self.n_sensors),dtype='float')
        self.out_table_tof  = np.zeros((self.n_events,self.n_sensors),dtype='int32')
        self.subth_QDC_L1   = np.zeros((self.n_events,len(self.L1)),dtype='int32')
        self.subth_TDC_L1   = np.zeros((self.n_events,len(self.L1)),dtype='int32')
        self.data_recons    = np.zeros((self.n_events,self.n_sensors),dtype='float')
        self.data_enc       = np.array([]).reshape(0,self.SIM_CONT.data['L1']['enc_out_len'])


    def write(self,iter=False):
        with pd.HDFStore(self.out_file,
                            complevel=9, complib='zlib') as store:
            panel_array = pd.DataFrame( data=self.out_table_B,
                                             columns=self.sensors)
            tof_array = pd.DataFrame( data=self.out_table_tof_B,
                                             columns=self.sensors)
            enc_array    = pd.DataFrame( data=self.data_enc_B)
            recons_array = pd.DataFrame( data=self.data_recons_B,
                                             columns=self.sensors)

            subth_QDC_L1 = pd.DataFrame( data=self.subth_QDC_L1_B)
            subth_TDC_L1 = pd.DataFrame( data=self.subth_TDC_L1_B)
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
            store.put('MC_TE',panel_array)
            store.put('MC_tof',tof_array)
            store.put('sensors',sensors_array)
            store.put('sipm2L1',sipm2L1)
            store.put('subth_QDC_L1',subth_QDC_L1)
            store.put('subth_TDC_L1',subth_TDC_L1)
            store.put('MC_encoded',enc_array)
            store.put('MC_recons',recons_array)
            store.close()


    # def encoder(self,event,first,diff_threshold):
    #
    #     def sigmoid(x, derivative=False):
    #         return x*(1-x) if derivative else 1/(1+np.exp(-x))
    #
    #     i = event
    #     data_enc_event = np.array([[]])
    #
    #     for L1 in self.L1:
    #         # L1 NUMBER
    #         L1_SiPM = np.array([],dtype='int').reshape(self.n_rows,0)
    #         for asic in L1:
    #             L1_SiPM = np.hstack((L1_SiPM,np.array(asic).reshape((self.n_rows,-1),
    #                                 order='F')))
    #         # Data PROCESSING
    #         data = self.out_table[i-first,L1_SiPM.T]
    #         data = data.reshape(1,-1)[0]
    #
    #
    #         if (data.shape[0]==self.COMP['ENC_weights_A'].shape[0]):
    #             data = (data-self.COMP['minA'].transpose())/ \
    #                    (self.COMP['maxA'].transpose()-self.COMP['minA'].transpose())
    #             data_enc_aux = sigmoid(np.dot(data,self.COMP['ENC_weights_A']) + self.COMP['ENC_bias_A'].T)
    #         else:
    #             data = (data-self.COMP['minB'].transpose())/ \
    #                    (self.COMP['maxB'].transpose()-self.COMP['minB'].transpose())
    #             data_enc_aux = sigmoid(np.dot(data,self.COMP['ENC_weights_B']) + self.COMP['ENC_bias_B'].T)
    #
    #         cond_TENC = (data_enc_aux > diff_threshold)
    #         data_enc_aux = data_enc_aux*cond_TENC
    #
    #         data_enc_event = np.hstack((data_enc_event,data_enc_aux))
    #
    #     # Store compressed information for every event
    #     self.data_enc = np.vstack((self.data_enc,data_enc_event))
    #
    #     ####################################################################
    #     ###            Event reconstruction after encoding               ###
    #     ####################################################################
    #
    #     index_2 = 0
    #
    #     for L1 in self.L1:
    #         index_1 = index_2
    #
    #         # Build L1_SiPM matrix
    #         L1_SiPM = np.array([],dtype='int').reshape(self.n_rows,0)
    #         for asic in L1:
    #             L1_SiPM = np.hstack((L1_SiPM,np.array(asic).reshape((self.n_rows,-1),
    #                                 order='F')))
    #
    #         if (L1_SiPM.shape[1]==(self.COMP['DEC_weights_A'].shape[1]//self.n_rows)):
    #             L1_size_compressed = self.COMP['DEC_weights_A'].shape[0]
    #             index_2 = index_1 + L1_size_compressed
    #             data_recons_event = self.data_enc[i-first,index_1:index_2]
    #             recons_event = sigmoid(np.dot(data_recons_event,self.COMP['DEC_weights_A']) + self.COMP['DEC_bias_A'].T)
    #             recons_event = recons_event*(self.COMP['maxA'].transpose()-self.COMP['minA'].transpose())\
    #                            + self.COMP['minA'].transpose()
    #             recons_event = recons_event.reshape((self.COMP['DEC_weights_A'].shape[1]//self.n_rows),self.n_rows)
    #
    #         else:
    #             L1_size_compressed = self.COMP['DEC_weights_B'].shape[0]
    #             index_2 = index_1 + L1_size_compressed
    #             data_recons_event = self.data_enc[i-first,index_1:index_2]
    #             recons_event = sigmoid(np.dot(data_recons_event,self.COMP['DEC_weights_B']) + self.COMP['DEC_bias_B'].T)
    #             recons_event = recons_event*(self.COMP['maxB'].transpose()-self.COMP['minB'].transpose())\
    #                            + self.COMP['minB'].transpose()
    #             recons_event = recons_event.reshape((self.COMP['DEC_weights_B'].shape[1]//self.n_rows),self.n_rows)
    #
    #         recons_event = recons_event.T
    #
    #         for sipm_id in L1_SiPM:
    #             # data_recons_event is now a matrix with same shape as L1_SiPM (see below)
    #             self.data_recons[i-first,sipm_id] = recons_event[np.where(L1_SiPM==sipm_id)]
    #
    #         # We apply the same threshold as for original data
    #         self.data_recons = (self.data_recons > self.TE2) * self.data_recons
    #
    #
    #
    # def enc_offset(self):
    #
    #     def sigmoid(x, derivative=False):
    #         return x*(1-x) if derivative else 1/(1+np.exp(-x))
    #
    #     DATA_IN = np.zeros((1,self.n_sensors))
    #
    #     data_enc_event = np.array([[]])
    #
    #     for L1 in self.L1:
    #         # L1 NUMBER
    #         L1_SiPM = np.array([],dtype='int').reshape(self.n_rows,0)
    #         for asic in L1:
    #             L1_SiPM = np.hstack((L1_SiPM,np.array(asic).reshape((self.n_rows,-1),
    #                                 order='F')))
    #         # Data PROCESSING
    #         data = DATA_IN[0,L1_SiPM.T]
    #         data = data.reshape(1,-1)[0]
    #
    #         # data = (data-self.COMP['minA'].transpose())/ \
    #         #        (self.COMP['maxA'].transpose()-self.COMP['minA'].transpose())
    #         data_enc_aux = sigmoid(np.dot(data,self.COMP['ENC_weights_A']) + self.COMP['ENC_bias_A'].T)
    #
    #
    #         # cond_TENC = (data_enc_aux > 0.038)
    #         # data_enc_aux = data_enc_aux*cond_TENC
    #
    #         data_enc_event = np.hstack((data_enc_event,data_enc_aux))
    #
    #     return data_enc_event
    #
    #
    #
    #
    # def encoder5mm(self,event,first,enc_threshold,diff_threshold):
    #
    #     def sigmoid(x, derivative=False):
    #         return x*(1-x) if derivative else 1/(1+np.exp(-x))
    #
    #     i = event
    #     data_enc_event = np.array([[]])
    #
    #     for L1 in self.L1:
    #         # L1 NUMBER
    #         L1_SiPM = np.array([],dtype='int').reshape(self.n_rows,0)
    #         for asic in L1:
    #             L1_SiPM = np.hstack((L1_SiPM,np.array(asic).reshape((self.n_rows,-1),
    #                                 order='F')))
    #         # Data PROCESSING
    #         data = self.out_table[i-first,L1_SiPM.T]
    #         data = data.reshape(1,-1)[0]
    #
    #         # data = (data-self.COMP['minA'].transpose())/ \
    #         #        (self.COMP['maxA'].transpose()-self.COMP['minA'].transpose())
    #         data_enc_aux = sigmoid(np.dot(data,self.COMP['ENC_weights_A']) + self.COMP['ENC_bias_A'].T)
    #         data_enc_event = np.hstack((data_enc_event,data_enc_aux))
    #
    #     #data_enc_event = data_enc_event - enc_threshold
    #     cond_TENC = data_enc_event > diff_threshold
    #     data_enc_event = data_enc_event*cond_TENC
    #
    #     # Store compressed information for every event
    #     self.data_enc = np.vstack((self.data_enc,data_enc_event))
    #
    #     ####################################################################
    #     ###            Event reconstruction after encoding               ###
    #     ####################################################################
    #
    #     index_2 = 0
    #
    #     for L1 in self.L1:
    #         index_1 = index_2
    #
    #         # Build L1_SiPM matrix
    #         L1_SiPM = np.array([],dtype='int').reshape(self.n_rows,0)
    #         for asic in L1:
    #             L1_SiPM = np.hstack((L1_SiPM,np.array(asic).reshape((self.n_rows,-1),
    #                                 order='F')))
    #
    #         #if (L1_SiPM.shape[1]==(self.COMP['DEC_weights_A'].shape[1]//self.n_rows)):
    #         L1_size_compressed = self.COMP['DEC_weights_A'].shape[0]
    #         index_2 = index_1 + L1_size_compressed
    #         data_recons_event = self.data_enc[i-first,index_1:index_2] + enc_threshold[0,index_1:index_2]
    #         recons_event = np.dot(data_recons_event,self.COMP['DEC_weights_A']) + self.COMP['DEC_bias_A'].T
    #         # recons_event = recons_event*(self.COMP['maxA'].transpose()-self.COMP['minA'].transpose())\
    #         #                + self.COMP['minA'].transpose()
    #         recons_event = recons_event.reshape((self.COMP['DEC_weights_A'].shape[1]//self.n_rows),self.n_rows)
    #
    #         recons_event = recons_event.T
    #
    #         for sipm_id in L1_SiPM:
    #             # data_recons_event is now a matrix with same shape as L1_SiPM (see below)
    #             self.data_recons[i-first,sipm_id] = recons_event[np.where(L1_SiPM==sipm_id)]
    #
    #         # We apply the same threshold as for original data
    #         self.data_recons = (self.data_recons > self.TE2) * self.data_recons


    def process(self,event_range,enc_threshold,diff_threshold):

        n_sensors = self.n_sensors

        low_limit = 0
        low_limit_tof = 0
        count = 0
        count_a = 0

        kwargs = {'n_rows':self.n_rows,
                  'COMP':self.COMP,
                  'TE2':self.TE2,
                  'n_sensors':self.n_sensors}

        ET = ENC.encoder_tools(**kwargs)

        first = event_range[0]
        for i in event_range: #range(0,800): #self.n_events):

            high_limit      = self.extents[i,1]
            high_limit_tof  = self.extents[i,2]
            event_wave = self.waves[low_limit:high_limit+1,:]
            event_tof  = self.tof[low_limit_tof:high_limit_tof+1,:]

            # Apply filtering with TE1 and TE2 thresholds
            for j in self.sensors:
                condition   = (event_wave[:,0] == j)
                condition_tof = (event_tof[:,0] == -j)
                L1_index = self.sipmtoL1[j-self.sensors[0]]
                # Paola's style

                sensor_data = np.sum(event_wave[condition,2])

                if np.sum(condition_tof)==False:
                    sensor_data_tof = 0
                else:
                    sensor_data_tof = np.amin(event_tof[condition_tof,1])*self.time_bin

                if (sensor_data > self.TE2):
                    self.out_table[i-first,count_a]     = sensor_data
                    self.out_table_tof[i-first,count_a] = sensor_data_tof

                if ((sensor_data > self.TE1) and (sensor_data <= self.TE2)):
                    self.subth_QDC_L1[i-first,L1_index] = self.subth_QDC_L1[i-first,L1_index] + sensor_data
                    if self.subth_TDC_L1[i-first,L1_index]==0:
                        self.subth_TDC_L1[i-first,L1_index] = sensor_data_tof
                    else:
                        self.subth_TDC_L1[i-first,L1_index] = np.amin(np.array([sensor_data_tof,
                                                                    self.subth_TDC_L1[i-first,L1_index]]))
                count_a += 1

            low_limit = high_limit+1
            low_limit_tof = high_limit_tof+1
            count_a = 0

            ####################################################################
            ###                    AUTOENCODER PROCESSING                    ###
            ####################################################################

            #self.encoder5mm(i,first,enc_threshold,diff_threshold)

            # ENCODER WORKS per L1 basis
            data_enc_event = np.array([[]])
            for L1 in self.L1:
                # Find SiPM included in L1
                L1_SiPM = np.array([],dtype='int').reshape(self.n_rows,0)
                for asic in L1:
                    L1_SiPM = np.hstack((L1_SiPM,np.array(asic).reshape((self.n_rows,-1),
                                        order='F')))
                # Read data from out_table
                data = self.out_table[i-first,L1_SiPM.T]
                data = data.reshape(1,-1)[0]

                data_enc_L1    = ET.encoder(L1,data,diff_threshold)
                data_enc_event = np.hstack((data_enc_event,data_enc_L1))

            # Store compressed information for every event
            self.data_enc = np.vstack((self.data_enc,data_enc_event))


            index_1 = 0
            for L1 in self.L1:
                # Build L1_SiPM matrix
                L1_SiPM = np.array([],dtype='int').reshape(self.n_rows,0)
                for asic in L1:
                    L1_SiPM = np.hstack((L1_SiPM,np.array(asic).reshape((self.n_rows,-1),
                                        order='F')))
                data_enc_aux = self.data_enc[i-first,:]
                index_1,recons_event = ET.decoder(L1_SiPM, data_enc_aux, index_1)

                for sipm_id in L1_SiPM:
                    # data_recons_event is now a matrix with same shape as L1_SiPM (see below)
                    # We apply the same threshold as for original data
                    self.data_recons[i-first,sipm_id] = recons_event[np.where(L1_SiPM==sipm_id)]



        # In the end we apply the same threshold for reconstructed
        #self.data_recons = self.data_recons * (self.data_recons > self.TE2)



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
                self.sipm_iter1A[i,0] = np.argmin(np.sqrt(np.sum(np.square(self.sensors_t[:,1:]-self.gamma1_i1[i,:]),axis=1))) + self.sensors[0]

            if (self.gamma2_i1[i,:].all() == 0):
                self.sipm_iter1B[i,0] = 0
            else:
                self.sipm_iter1B[i,0] = np.argmin(np.sqrt(np.sum(np.square(self.sensors_t[:,1:]-self.gamma2_i1[i,:]),axis=1))) + self.sensors[0]


    def add_event_batch(self):
        self.out_table_B = np.vstack((self.out_table_B,self.out_table))
        self.out_table_tof_B = np.vstack((self.out_table_tof_B,self.out_table_tof))
        self.data_enc_B = np.vstack((self.data_enc_B,self.data_enc))
        self.data_recons_B = np.vstack((self.data_recons_B,self.data_recons))
        self.subth_QDC_L1_B = np.vstack((self.subth_QDC_L1_B,self.subth_QDC_L1))
        self.subth_TDC_L1_B = np.vstack((self.subth_TDC_L1_B,self.subth_TDC_L1))
        # self.gamma1_i1_B = np.vstack((self.gamma1_i1_B,self.gamma1_i1))
        # self.gamma2_i1_B = np.vstack((self.gamma2_i1_B,self.gamma2_i1))
        # self.sipm_iter1A_B = np.vstack((self.sipm_iter1A_B,self.sipm_iter1A))
        # self.sipm_iter1B_B = np.vstack((self.sipm_iter1B_B,self.sipm_iter1B))


def DAQ_out(file_number,path,jsonfilename,encoder_data,Tenc):

    TEST_c = DAQ_MODEL(path,jsonfilename,file_number,encoder_data)
    TEST_c.read_files()
    #enc_threshold = TEST_c.enc_offset()
    enc_threshold = 0

    group = 100
    e_vec = np.arange(0,TEST_c.events_infile,group)
    if (e_vec[-1]<(TEST_c.events_infile-1)):
        e_vec = np.concatenate((e_vec,[TEST_c.events_infile-1]))
    low  = e_vec[:-1]
    high = e_vec[1:]
    i=0
    for x,x_1 in zip(low,high):
        TEST_c.read_data(np.arange(x,x_1))
        #diff_threshold allows to send only useful information
        TEST_c.process(np.arange(x,x_1),enc_threshold,Tenc)
        TEST_c.add_event_batch()
        print ("BATCH %d DONE" % i)
        i+=1

    TEST_c.process_table()
    TEST_c.h5file.close()
    TEST_c.write(iter=True)



if __name__ == "__main__":

    json_file = "test"
    path      = "/home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/5mm_pitch/"

    SIM_JSON     = CFG.SIM_DATA(filename = path + json_file +".json",read=True)
    encoder_file = SIM_JSON.data['ENVIRONMENT']['AUTOENCODER_file_name']
    Tenc         = SIM_JSON.data['L1']['Tenc']
    # AUTOENCODER DATA READING (Trick to read easily and keep names)
    COMP={}
    with pd.HDFStore(path + encoder_file) as store:
        keys = store.keys()
    COMP = {i[1:]:np.array(pd.read_hdf(path + encoder_file,key=i[1:])) for i in keys}

    kargs = {'path'         :path,
             'jsonfilename' :json_file,
             'encoder_data' :COMP,
             'Tenc'         :Tenc}
    TRANS_map = partial(DAQ_out, **kargs)

    # Multiprocess Work
    tic = time.time()
    pool_size = mp.cpu_count() // 2
    pool = mp.Pool(processes=pool_size)
    #Range of Files to Translate
    pool.map(TRANS_map, [i for i in SIM_JSON.data['ENVIRONMENT']['n_files']])
    pool.close()
    pool.join()

    toc = time.time()
    print("You made me waste %d seconds of my precious time in this simulation" % (toc-tic))

    #DAQ_out(0,**kargs)
