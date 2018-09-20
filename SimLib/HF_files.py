import simpy
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from simpy.events import AnyOf, AllOf, Event
import sys
sys.path.append("../PETALO_DAQ_infinity/SimLib")
sys.path.append("../PETALO_analysis")
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import fit_library
import config_sim as CFG


class DAQ_IO(object):
    """ H5 access for DAQ.
        Methods
        write       : Writes H5 with output dataframes from DAQ (see DATA FRAME)
                      Also attaches a table with sensor positions
        read        : Reads DAQ output frames from H5 file
        write_out   : Writes processed output frames. The output format is the
                      same of the input processed H5 (Table with Sensors as columns
                      and events as rows)
        Parameters

    """

    def __init__(self,path,daq_filename,ref_filename,daq_outfile):
        self.path = path
        self.out_filename = daq_filename
        self.ref_filename = ref_filename
        self.daq_outfile = daq_outfile
        os.chdir(self.path)
        self.sensors_xyz = np.array( pd.read_hdf(self.ref_filename,
                                    key='sensors'),
                                    dtype = 'float32')

    def write(self,data):
        os.chdir(self.path)
        with pd.HDFStore(self.out_filename) as store:
            panel_array = pd.DataFrame( data = data,
                                        columns = ['data','event','sensor_id',
                                                    'asic_id','in_time','out_time'])
            sensors_array = pd.DataFrame( data=self.sensors_xyz,
                                          columns=['sensor','x','y','z'])
            # complevel and complib are not compatible with MATLAB
            store.put('MC',panel_array)
            store.put('sensors',sensors_array)
            store.close()

    # def read(self):
    #     os.chdir(self.path)
    #     data = np.array(pd.read_hdf(self.out_filename,key='MC'), dtype='int32')
    #     sensors = np.array(np.array(pd.read_hdf(self.out_filename,key='sensors'),
    #               dtype='int32'))
    #     return data,sensors

    def read(self):
        data = np.array(pd.read_hdf(self.path+self.out_filename,
                                    key='MC'), dtype='int32')
        sensors = np.array(np.array(pd.read_hdf(self.path+self.out_filename,
                                    key='sensors'),dtype='int32'))
        return data,sensors


    def write_out(self,data,topology={},logs={}):
        os.chdir(self.path)
        with pd.HDFStore(self.daq_outfile,
                        complevel=9, complib='zlib') as store:
            self.panel_array = pd.DataFrame( data=data,
                                             columns=self.sensors_xyz[:,0])

            self.sensors_array = pd.DataFrame( data=self.sensors_xyz,
                                                columns=['sensor','x','y','z'])
            topo_data = np.array(list(topology.values())).reshape(1,len(list(topology.values())))
            logA         = np.array(logs['logA'])
            logB         = np.array(logs['logB'])
            log_channels = np.array(logs['log_channels'])
            log_outlink = np.array(logs['log_outlink'])
            log_in_time = np.array(logs['in_time'])
            log_out_time = np.array(logs['out_time'])

            topo = pd.DataFrame(data = topo_data,columns = list(topology.keys()))
            logA = pd.DataFrame(data = logA)
            logB = pd.DataFrame(data = logB)
            log_channels = pd.DataFrame(data = log_channels)
            log_outlink  = pd.DataFrame(data = log_outlink)
            log_in_time  = pd.DataFrame(data = log_in_time)
            log_out_time  = pd.DataFrame(data = log_out_time)
            lost_data = np.array(list(logs['lost'].values())).reshape(1,-1)
            lost = pd.DataFrame(data = lost_data,columns = list(logs['lost'].keys()))
            compress = pd.DataFrame(data = logs['compress'])
            tstamp_event = pd.DataFrame(data = logs['tstamp_event'])
            timestamp = pd.DataFrame(data=logs['timestamp'])
            # complevel and complib are not compatible with MATLAB
            store.put('MC',self.panel_array)
            store.put('sensors',self.sensors_array)
            store.put('topology',topo)
            store.put('logA',logA)
            store.put('logB',logB)
            store.put('log_channels',log_channels)
            store.put('log_outlink',log_outlink)
            store.put('in_time',log_in_time)
            store.put('out_time',log_out_time)
            store.put('lost',lost)
            store.put('compress',compress)
            store.put('tstamp_event',tstamp_event)
            store.put('timestamp',timestamp)
            store.close()


class hdf_access(object):
    """ A utility class to access data in hf5 format.
        read method is used to load data from a preprocessed file.
        The file format is a table with each column is a sensor and
        each row an event
    """

    def __init__(self,path,file_name):
        self.path = path
        self.file_name = file_name

    def read(self):
        os.chdir(self.path)
        self.data = pd.read_hdf(self.file_name,key='MC')

        # Reads translated hf files (table with sensor/charge per event)
        self.sensors = np.array(self.data.columns)
        self.data = np.array(self.data, dtype = 'int32')
        self.events = self.data.shape[0]

        #returns data array, sensors vector, and number of events
        return self.data,self.sensors,self.events

    def read_DAQ_fast(self):
        file_name = self.path+self.file_name
        with pd.HDFStore(file_name) as hdf:
            out={}
            for i in hdf.keys():
                out[i] = pd.read_hdf(file_name,key=i)
        return out

class hdf_compose(object):
    """ A utility class to access preprocessed data from MCs in hf5 format.
            param
            files           : Array of files
            n_sensors       : Number of sensors (all of them)
            Output
            composed data
            sensor array
            number of events
    """

    def __init__(self,path,file_name,files,n_sensors):
        self.path       = path
        self.file_name  = file_name
        self.files      = files
        self.n_sensors  = n_sensors
        self.data       = np.array([]).reshape(0,self.n_sensors)
        self.data_aux   = np.array([]).reshape(0,self.n_sensors)

    def compose(self):

        hf = hdf_access(self.path,self.file_name + str(self.files[0]) + ".h5")
        self.data_aux,self.sensors,self.events = hf.read()
        self.data = np.pad( self.data,
                            ((0,self.events),(0,0)),
                            mode='constant',
                            constant_values=0)
        self.data[-self.events:,:] = self.data_aux

        for i in self.files:
            hf = hdf_access(self.path,self.file_name + str(i) + ".h5")
            self.data_aux,self.fake,self.events = hf.read()
            self.data = np.pad( self.data,
                                ((0,self.events),(0,0)),
                                mode='constant',
                                constant_values=0)
            self.data[-self.events:,:] = self.data_aux


        return self.data, self.sensors, self.data.shape[0]


class infinity_graphs(object):
    """ Data Analysis and Graphs generation
    """
    def __init__(self,config_file,data_path):
        self.config_file = config_file
        self.data_path   = data_path

    def __call__(self):

        logA         = np.array([]).reshape(0,2)
        logB         = np.array([]).reshape(0,2)
        log_channels = np.array([]).reshape(0,2)
        log_outlink  = np.array([]).reshape(0,2)
        in_time      = np.array([]).reshape(0,1)
        out_time     = np.array([]).reshape(0,1)
        lost         = np.array([]).reshape(0,4)
        compress     = np.array([]).reshape(0,1)

        for i in self.config_file:
            config_file2 = self.data_path + i + ".json"
            CG = CFG.SIM_DATA(filename = config_file2,read = True)
            CG = CG.data
            filename = CG['ENVIRONMENT']['out_file_name']+"_"+ i + ".h5"
            filename = self.data_path + filename

            logA         = np.vstack([logA,np.array(pd.read_hdf(filename,key='logA'))])
            logB         = np.vstack([logB,np.array(pd.read_hdf(filename,key='logB'))])
            log_channels = np.vstack([log_channels,np.array(pd.read_hdf(filename,key='log_channels'))])
            log_outlink  = np.vstack([log_outlink,np.array(pd.read_hdf(filename,key='log_outlink'))])
            in_time      = np.vstack([in_time,np.array(pd.read_hdf(filename,key='in_time'))])
            out_time     = np.vstack([out_time,np.array(pd.read_hdf(filename,key='out_time'))])
            lost         = np.vstack([lost,np.array(pd.read_hdf(filename,key='lost'))])
            compress     = np.vstack([compress,np.array(pd.read_hdf(filename,key='compress'))])


        latency_L1 = logA[:,1]
        latency    = out_time-in_time

        print ("LOST DATA PRODUCER -> CH      = %d" % (lost[:,0].sum()))
        print ("LOST DATA CHANNELS -> OUTLINK = %d" % (lost[:,1].sum()))
        print ("LOST DATA OUTLINK  -> L1      = %d" % (lost[:,2].sum()))
        print ("LOST DATA L1A -> L1B          = %d" % (lost[:,3].sum()))

        WC_CH_FIFO    = float(max(log_channels[:,0])/CG['TOFPET']['IN_FIFO_depth'])*100
        WC_OLINK_FIFO = float(max(log_outlink[:,0])/CG['TOFPET']['OUT_FIFO_depth'])*100
        WC_L1_A_FIFO  = float(max(logA[:,0])/CG['L1']['FIFO_L1a_depth'])*100
        WC_L1_B_FIFO  = float(max(logB[:,0])/CG['L1']['FIFO_L1b_depth'])*100


        print ("\n \n BYE \n \n")

        fit = fit_library.gauss_fit()
        fig = plt.figure(figsize=(20,10))

        fit(log_channels[:,0],CG['TOFPET']['IN_FIFO_depth'])
        fit.plot(axis = fig.add_subplot(341),
                title = "ASICS Channel Input analog FIFO (4)",
                xlabel = "FIFO Occupancy",
                ylabel = "Hits",
                res = False, fit = False)
        fig.add_subplot(341).set_yscale('log')
        fig.add_subplot(341).text(0.99,0.97,(("ASIC Input FIFO reached %.1f %%" % \
                                                (WC_CH_FIFO))),
                                                fontsize=8,
                                                verticalalignment='top',
                                                horizontalalignment='right',
                                                transform=fig.add_subplot(341).transAxes)

        fit(log_outlink[:,0],CG['TOFPET']['OUT_FIFO_depth'])
        fit.plot(axis = fig.add_subplot(342),
                title = "ASICS Channels -> Outlink",
                xlabel = "FIFO Occupancy",
                ylabel = "Hits",
                res = False, fit = False)
        fig.add_subplot(342).set_yscale('log')
        fig.add_subplot(342).text(0.99,0.97,(("ASIC Outlink FIFO reached %.1f %%" % \
                                                (WC_OLINK_FIFO))),
                                                fontsize=8,
                                                verticalalignment='top',
                                                horizontalalignment='right',
                                                transform=fig.add_subplot(342).transAxes)

        fit(logA[:,0],CG['L1']['FIFO_L1a_depth'])
        fit.plot(axis = fig.add_subplot(346),
                title = "ASICS -> L1A (FIFOA)",
                xlabel = "FIFO Occupancy",
                ylabel = "Hits",
                res = False, fit = False)
        fig.add_subplot(346).set_yscale('log')
        fig.add_subplot(346).text(0.99,0.97,(("L1_A FIFO reached %.1f %%" % \
                                                (WC_L1_A_FIFO))),
                                                fontsize=8,
                                                verticalalignment='top',
                                                horizontalalignment='right',
                                                transform=fig.add_subplot(346).transAxes)

        fit(logB[:,0],CG['L1']['FIFO_L1b_depth'])
        fit.plot(axis = fig.add_subplot(345),
                title = "L1 OUTPUT (FIFOB)",
                xlabel = "FIFO Occupancy",
                ylabel = "Hits",
                res = False, fit = False)
        fig.add_subplot(345).set_yscale('log')
        fig.add_subplot(345).text(0.99,0.97,(("L1_B FIFO reached %.1f %%" % \
                                                (WC_L1_B_FIFO))),
                                                fontsize=8,
                                                verticalalignment='top',
                                                horizontalalignment='right',
                                                transform=fig.add_subplot(345).transAxes)
        fit(latency,50)
        fit.plot(axis = fig.add_subplot(343),
                title = "Total Data Latency",
                xlabel = "Latency in nanoseconds",
                ylabel = "Hits",
                res = False)
        fig.add_subplot(343).text(0.99,0.8,(("WORST LATENCY = %d ns" % \
                                                (max(latency)))),
                                                fontsize=7,
                                                verticalalignment='top',
                                                horizontalalignment='right',
                                                transform=fig.add_subplot(343).transAxes)

        fit(latency_L1,50)
        fit.plot(axis = fig.add_subplot(349),
                title = "L1 input Data Latency",
                xlabel = "Latency in nanoseconds",
                ylabel = "Hits",
                res = False)
        fig.add_subplot(349).text(0.99,0.8,(("WORST LATENCY = %d ns" % \
                                                (max(latency_L1)))),
                                                fontsize=7,
                                                verticalalignment='top',
                                                horizontalalignment='right',
                                                transform=fig.add_subplot(349).transAxes)

        new_axis = fig.add_subplot(347)
        x_data = fit.bin_centers
        y_data = np.add.accumulate(fit.hist_fit)/np.max(np.add.accumulate(fit.hist_fit))
        new_axis.plot(x_data,y_data)
        new_axis.set_ylim((0.9,1.0))
        new_axis.set_xlabel("Latency in nanoseconds")
        new_axis.set_ylabel("Percentage of Recovered Data")
        new_axis.text(0.05,0.9,(("LOST DATA PRODUCER -> CH           = %d\n" + \
                                 "LOST DATA CHANNELS -> OUTLINK  = %d\n" + \
                                 "LOST DATA OUTLINK -> L1                = %d\n" + \
                                 "LOST DATA L1A -> L1B                      = %d\n") % \
                                (lost[:,0].sum(),
                                 lost[:,1].sum(),
                                 lost[:,2].sum().sum(),
                                 lost[:,3].sum().sum())
                                ),
                                fontsize=8,
                                verticalalignment='top',
                                horizontalalignment='left',
                                transform=new_axis.transAxes)


        fit(compress,int(np.max(compress)))
        fit.plot(axis = fig.add_subplot(344),
                title = "Data Frame Length (Compression)",
                xlabel = "Number of QDC fields",
                ylabel = "Hits",
                res = False,
                fit = False)

        # TOTAL NUMBER OF BITS vs COMPRESS EFFICIENCY
        A = np.arange(0,np.max(compress))
        D_data = 1 + 7*(A>0) + A * 23 + 10     #see DAQ_infinity
        D_save = (A-1)*10
        B_data = np.multiply(D_data,fit.hist)
        B_save = np.multiply(D_save,fit.hist)
        B_save[0]=0
        B_save[1]=0
        new_axis_2 = fig.add_subplot(348)
        x_data = fit.bin_centers
        new_axis_2.bar(x_data,B_data,color='r')
        new_axis_2.bar(x_data,B_save,color='b')
        new_axis_2.set_title("Data sent vs frame length")
        new_axis_2.set_xlabel("Length of frame in QDC data")
        new_axis_2.set_ylabel("Red - Data sent (bits) / Blue - Data saved (bits)")

        new_axis_2.text(0.99,0.97,(("TOTAL DATA SENT = %d bits\n" + \
                                 "DATA REDUCTION  = %d bits\n" + \
                                 "COMPRESS RATIO = %f \n") % \
                                (np.sum(B_data),np.sum(B_save),float(np.sum(B_save))/float(np.sum(B_save)+np.sum(B_data)))),
                                fontsize=8,
                                verticalalignment='top',
                                horizontalalignment='right',
                                transform=new_axis_2.transAxes)


        fig.tight_layout()

        #plt.savefig(CG['ENVIRONMENT']['out_file_name']+"_"+ filename + ".pdf")
        plt.savefig(filename + ".pdf")


def main():
    A = infinity_graphs(["OF_4mm_min"],"/home/viherbos/DAQ_DATA/NEUTRINOS/PETit-ring/4mm_pitch/")
    A()
    # start = time.time()
    #
    # files = [0,1,2,3,4,5,6,8]
    #
    # TEST_c = hdf_compose(  "/home/viherbos/DAQ_DATA/NEUTRINOS/RING/",
    #                        "p_FRSET_", files, 1536)
    # a,b,c = TEST_c.compose()
    #
    # time_elapsed = time.time() - start
    #
    # print ("It took %d seconds to compose %d files" % (time_elapsed,
    #                                                    len(files)))


if __name__ == "__main__":
    main()
