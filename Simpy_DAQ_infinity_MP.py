import simpy
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from simpy.events import AnyOf, AllOf, Event
import sys
sys.path.append("../PETALO_analysis/")
import fit_library
import HF_translator as HFT
import os
import multiprocessing as mp
from functools import partial
from SimLib import DAQ_infinity as DAQ
from SimLib import HF_files as HF
import time
from SimLib import config_sim as CFG
from SimLib import pet_graphics as PG
import pandas as pd
import math
import argparse


def L1_sch(SiPM_Matrix_Slice, sim_info):

    data_out   = []
    param = sim_info['Param']
    DATA  = sim_info['DATA']

    env = simpy.Environment()

    n_asics = len(SiPM_Matrix_Slice)

    L1 = DAQ.L1( env        = env,
                 out_stream = data_out,
                 param      = param,
                 L1_id      = 0)

    block_size = param.P['TOFPET']['n_channels']

    ASICS_L1 = [DAQ.FE_asic(
                    env     = env,
                    param   = param,
                    data    = DATA[:,SiPM_Matrix_Slice[i]],
                    timing  = sim_info['timing'],
                    sensors = sim_info['Param'].sensors[SiPM_Matrix_Slice[i]],
                    asic_id = i)
                for i in range(n_asics)]

    for i in range(len(ASICS_L1)):
        ASICS_L1[i].Link.out = L1

    # Run Simulation for a very long time to force flush of FIFOs
    env.run(until = 100E9)

    OUTPUT_L1     = L1()
    OUTPUT_ASICS  = [ASICS_L1[i]() for i in range(n_asics)]

    print "L1 finished"

    return {'L1_out':OUTPUT_L1, 'ASICS_out':OUTPUT_ASICS}




def DAQ_sim(sim_info):
    param = sim_info['Param']
    # Work out number of SiPMs based on geometry data
    n_sipms_I = param.P['TOPOLOGY']['sipm_int_row']*param.P['TOPOLOGY']['n_rows']
    n_sipms_O = param.P['TOPOLOGY']['sipm_ext_row']*param.P['TOPOLOGY']['n_rows']
    n_sipms     = n_sipms_I + n_sipms_O
    # Number of ASICs calculation: Inner Face + Outer Face // full + partial
    n_asics_I = int(math.ceil(float(n_sipms_I) / float(param.P['TOFPET']['n_channels'])))
    n_asics_f_I  = n_sipms_I // param.P['TOFPET']['n_channels']  # Fully used
    n_asics_p_I = n_asics_I - n_asics_f_I                       # Partially used
    n_asics_O = int(math.ceil(float(n_sipms_O) / float(param.P['TOFPET']['n_channels'])))
    n_asics_f_O  = n_sipms_O // param.P['TOFPET']['n_channels']
    n_asics_p_O   = n_asics_O - n_asics_f_O      # Number of not fully used ASICs (0 or 1)
    n_asics = n_asics_I + n_asics_O
    # L1 are required with max number of ASICs in param.P['L1']['n_asics']
    # // full + part
    n_L1 = int(math.ceil(float(n_asics) / float(param.P['L1']['n_asics'])))
    n_L1_f = n_asics // param.P['L1']['n_asics']
    n_L1_p = n_L1 - n_L1_f

    print ("Number of SiPM : %d \nNumber of ASICS : %d " % (n_sipms,n_asics))
    print ("Number of L1 : %d " % (n_L1))

    SiMP_Matrix_I = np.reshape(np.arange(0,n_sipms_I),
                                (param.P['TOPOLOGY']['n_rows'],
                                param.P['TOPOLOGY']['sipm_int_row']))
    SiMP_Matrix_O = np.reshape(np.arange(n_sipms_I,n_sipms),
                                (param.P['TOPOLOGY']['n_rows'],
                                param.P['TOPOLOGY']['sipm_ext_row']))
    # SiPM matrixs Inner face and Outer face

    # Generation of Iterable for pool.map
    L1_Slice=[]
    count = 0

    SiPM_ASIC_Slice=[]
    # Generate Slice of ASICs (SiPM) for L1
    for i in range(n_asics_I):
        if (count < param.P['L1']['n_asics']-1):
            SiPM_ASIC_Slice.append(np.reshape(SiMP_Matrix_I[:,i*4:(i+1)*4],-1))
            count += 1
        else:
            SiPM_ASIC_Slice.append(np.reshape(SiMP_Matrix_I[:,i*4:(i+1)*4],-1))
            L1_Slice.append(SiPM_ASIC_Slice)
            SiPM_ASIC_Slice=[]
            count = 0

    # if (n_asics_p_I == 1):
    #     L1_Slice.append(SiPM_ASIC_Slice)


    for i in range(n_asics_O):
        if (count < param.P['L1']['n_asics']-1):
            SiPM_ASIC_Slice.append(np.reshape(SiMP_Matrix_O[:,i*4:(i+1)*4],-1))
            count += 1
        else:
            SiPM_ASIC_Slice.append(np.reshape(SiMP_Matrix_O[:,i*4:(i+1)*4],-1))
            L1_Slice.append(SiPM_ASIC_Slice)
            SiPM_ASIC_Slice=[]
            count = 0

    if (n_L1_p == 1):
        L1_Slice.append(SiPM_ASIC_Slice)

    print ("Number of Instanciated L1 = %d" % (len(L1_Slice)))
    for i in range(len(L1_Slice)):
        print ("L1 number %d has %d ASICs" % (i,len(L1_Slice[i])))

    # Multiprocess Pool Management

    kargs = {'sim_info':sim_info}
    DAQ_map = partial(L1_sch, **kargs)

    start_time = time.time()
    # Multiprocess Work
    pool_size = mp.cpu_count() #// 2
    pool = mp.Pool(processes=pool_size)

    pool_output = pool.map(DAQ_map, [i for i in L1_Slice])

    pool.close()
    pool.join()
    #pool_output = DAQ_map(L1_Slice[0])

    elapsed_time = time.time()-start_time
    print ("IT TOOK SKYNET %d SECONDS TO RUN THIS SIMULATION" % elapsed_time)

    topology = {'n_sipms_I':n_sipms_I, 'n_sipms_O':n_sipms_O, 'n_sipms': n_sipms,
            'n_asics_I':n_asics_I, 'n_asics_f_I':n_asics_f_I,'n_asics_p_I':n_asics_p_I,
            'n_asics_O':n_asics_O, 'n_asics_f_O':n_asics_f_O,'n_asics_p_O':n_asics_p_O,
            'n_asics':n_asics, 'n_L1':n_L1, 'n_L1_f':n_L1_f, 'n_L1_p':n_L1_p}

    return pool_output,topology



def DAQ_OUTPUT_processing(SIM_OUT,n_L1,n_asics):
    data, in_time, out_time, lostL1a, lostL1b = [],[],[],[],[]
    lost_producers= np.array([]).reshape(0,1)
    lost_channels = np.array([]).reshape(0,1)
    lost_outlink = np.array([]).reshape(0,1)
    SIM_OUT_L1      = np.array(SIM_OUT['L1_out'])
    SIM_OUT_ASICs   = np.array(SIM_OUT['ASICS_out'])
    logA = np.array([]).reshape(0,2)
    logB = np.array([]).reshape(0,2)
    log_channels = np.array([]).reshape(0,2)
    log_outlink = np.array([]).reshape(0,2)

    # Gather information from ASICS layer
    for j in range(n_asics):
        lost_producers = np.vstack([lost_producers,
                                    SIM_OUT_ASICs[j]['lost_producers']])
        lost_channels = np.vstack([lost_channels,
                                    SIM_OUT_ASICs[j]['lost_channels']])
        lost_outlink  = np.vstack([lost_outlink,
                                    SIM_OUT_ASICs[j]['lost_outlink']])
        log_channels  = np.vstack([log_channels,
                                    SIM_OUT_ASICs[j]['log_channels']])
        log_outlink   = np.vstack([log_outlink,
                                    SIM_OUT_ASICs[j]['log_outlink']])

    # Gather information from L1 layer
    for j in range(n_L1):
        #lostL1a.append(SIM_OUT_L1[j]['lostL1a'])
        lostL1b.append(SIM_OUT_L1[j]['lostL1b'])
        logA=np.vstack([logA,SIM_OUT_L1[j]['logA']])
        logB=np.vstack([logB,SIM_OUT_L1[j]['logB']])

        for i in range(len(SIM_OUT_L1[j]['data_out'])):
            #if SIM_OUT[j]['data_out'][i]['data'][0] > 0:
            data.append(SIM_OUT_L1[j]['data_out'][i]['data'])
            in_time.append(SIM_OUT_L1[j]['data_out'][i]['in_time'])
            out_time.append(SIM_OUT_L1[j]['data_out'][i]['out_time'])


    A = np.array(data)
    sort = np.array([i[1] for i in A])
    A = A[np.argsort(sort)]

    n_TDC = np.array([])
    i_TDC = np.array([])
    TDC = np.array([A[i][1] for i in range(len(A))])


    prev=0
    for i in TDC:
        if (i != prev):
            cond = np.array((TDC == i))
            n_TDC = np.concatenate((n_TDC,[np.sum(cond)]),axis=0)
            i_TDC = np.concatenate((i_TDC,[i]),axis=0)
            prev = i

    # Data table building
    event = 0
    A_index = 0

    data = np.zeros((n_events,n_sipms),dtype='int32')
    for i in i_TDC:
        for j in range(int(n_TDC[event])):
            for l in range(int(A[A_index][0])):
                data[event,int(A[A_index][2*l+2])-1000] = A[A_index][2*l+3]

            A_index += 1

        event += 1

    n_words = np.zeros(len(A))
    # Buffer compression statistic
    for i in range(len(A)):
        n_words[i] = A[i][0]


    output = {'data': data,
              'L1': {'in_time': in_time, 'out_time': out_time,
                     'lostL1b': lostL1b, 'logA': logA, 'logB': logB},
              'ASICS':{ 'lost_producers':lost_producers,
                        'lost_channels':lost_channels,
                        'lost_outlink':lost_outlink,
                        'log_channels':log_channels,
                        'log_outlink':log_outlink},
              'compress': n_words
            }


    return output



if __name__ == '__main__':

    # Argument parser for config file name
    parser = argparse.ArgumentParser(description='PETALO Infinity DAQ Simulator.')
    parser.add_argument("-f", "--json_file", action="store_true",
                        help="Simulate with configuration stored in json file")
    parser.add_argument('arg1', metavar='N', nargs='?', help='')
    parser.add_argument("-d", "--directory", action="store_true",
                        help="Work directory")
    parser.add_argument('arg2', metavar='N', nargs='?', help='')
    args = parser.parse_args()

    if args.json_file:
        file_name = ''.join(args.arg1)
    else:
        file_name = "sim_config"
    if args.directory:
        path = ''.join(args.arg2)
    else:
        path="./"

    config_file = file_name + ".json"

    CG = CFG.SIM_DATA(filename = path + config_file, read = True)
    CG = CG.data
    # Read data from json file

    n_sipms_int = CG['TOPOLOGY']['sipm_int_row']*CG['TOPOLOGY']['n_rows']
    n_sipms_ext = CG['TOPOLOGY']['sipm_ext_row']*CG['TOPOLOGY']['n_rows']
    n_sipms     = n_sipms_int + n_sipms_ext

    n_files = CG['ENVIRONMENT']['n_files']
    # Number of files to group for data input
    A = HF.hdf_compose( CG['ENVIRONMENT']['path_to_files'],
                        CG['ENVIRONMENT']['file_name'],
                        range(n_files),n_sipms)
    DATA,sensors,n_events = A.compose()

    # Number of events for simulation
    n_events = CG['ENVIRONMENT']['n_events']
    DATA = DATA[0:n_events,:]
    print (" %d EVENTS IN %d H5 FILES" % (n_events,n_files))

    # SHOW = PG.DET_SHOW(CG.data)
    # os.chdir("/home/viherbos/DAQ_DATA/NEUTRINOS/RING/")
    # filename = "p_FRSET_0.h5"
    # positions = np.array(pd.read_hdf(filename,key='sensors'))
    # data = np.array(pd.read_hdf(filename,key='MC'), dtype = 'int32')
    # SHOW(positions,data,0,True,False)


    Param = DAQ.parameters(CG,sensors,n_events)

    timing = np.random.randint(0,int(1E9/Param.P['ENVIRONMENT']['ch_rate']),
                                 size=n_events)

    # All sensors are given the same timestamp in an events
    sim_info = {'DATA': DATA, 'timing': timing, 'Param': Param }

    # Call Simulation Function
    pool_out,topology = DAQ_sim(sim_info)

    # Translate Simulation Output into an array for Data recovery
    SIM_OUT = {'L1_out':[], 'ASICS_out':[]}
    for i in range(len(pool_out)):
        SIM_OUT['L1_out'].append(pool_out[i]['L1_out'])
        for j in range(len(pool_out[i]['ASICS_out'])):
            SIM_OUT['ASICS_out'].append(pool_out[i]['ASICS_out'][j])
    # Data Output recovery
    out = DAQ_OUTPUT_processing(SIM_OUT,topology['n_L1'],topology['n_asics'])

    #//////////////////////////////////////////////////////////////////
    #///                     DATA ANALYSIS AND GRAPHS               ///
    #//////////////////////////////////////////////////////////////////

    latency = np.array(out['L1']['out_time'])-np.array(out['L1']['in_time'])

    print ("LOST DATA PRODUCER -> CH      = %d" % (out['ASICS']['lost_producers'].sum()))
    print ("LOST DATA CHANNELS -> OUTLINK = %d" % (out['ASICS']['lost_channels'].sum()))
    print ("LOST DATA OUTLINK  -> L1      = %d" % (out['ASICS']['lost_outlink'].sum()))
    print ("LOST DATA L1A -> L1B          = %d" % (np.array(out['L1']['lostL1b']).sum()))

    WC_CH_FIFO    = float(max(out['ASICS']['log_channels'][:,0])/CG['TOFPET']['IN_FIFO_depth'])*100
    WC_OLINK_FIFO = float(max(out['ASICS']['log_outlink'][:,0])/CG['TOFPET']['OUT_FIFO_depth'])*100
    WC_L1_A_FIFO  = float(max(out['L1']['logA'][:,0])/CG['L1']['FIFO_L1a_depth'])*100
    WC_L1_B_FIFO  = float(max(out['L1']['logB'][:,0])/CG['L1']['FIFO_L1b_depth'])*100


    print ("\n \n BYE \n \n")

    fit = fit_library.gauss_fit()
    fig = plt.figure(figsize=(20,8))

    fit(out['ASICS']['log_channels'][:,0],CG['TOFPET']['IN_FIFO_depth'])
    fit.plot(axis = fig.add_subplot(241),
            title = "ASICS Channel Input analog FIFO (4)",
            xlabel = "FIFO Occupancy",
            ylabel = "Hits",
            res = False, fit = False)
    fig.add_subplot(241).set_yscale('log')
    fig.add_subplot(241).text(0.99,0.97,(("ASIC Input FIFO reached %.1f %%" % \
                                            (WC_CH_FIFO))),
                                            fontsize=8,
                                            verticalalignment='top',
                                            horizontalalignment='right',
                                            transform=fig.add_subplot(241).transAxes)

    fit(out['ASICS']['log_outlink'][:,0],CG['TOFPET']['OUT_FIFO_depth'])
    fit.plot(axis = fig.add_subplot(242),
            title = "ASICS Channels -> Outlink",
            xlabel = "FIFO Occupancy",
            ylabel = "Hits",
            res = False, fit = False)
    fig.add_subplot(242).set_yscale('log')
    fig.add_subplot(242).text(0.99,0.97,(("ASIC Outlink FIFO reached %.1f %%" % \
                                            (WC_OLINK_FIFO))),
                                            fontsize=8,
                                            verticalalignment='top',
                                            horizontalalignment='right',
                                            transform=fig.add_subplot(242).transAxes)

    fit(out['L1']['logA'][:,0],CG['L1']['FIFO_L1a_depth'])
    fit.plot(axis = fig.add_subplot(246),
            title = "ASICS -> L1A (FIFOA)",
            xlabel = "FIFO Occupancy",
            ylabel = "Hits",
            res = False, fit = False)
    fig.add_subplot(246).set_yscale('log')
    fig.add_subplot(246).text(0.99,0.97,(("L1_A FIFO reached %.1f %%" % \
                                            (WC_L1_A_FIFO))),
                                            fontsize=8,
                                            verticalalignment='top',
                                            horizontalalignment='right',
                                            transform=fig.add_subplot(246).transAxes)

    fit(out['L1']['logB'][:,0],CG['L1']['FIFO_L1b_depth'])
    fit.plot(axis = fig.add_subplot(245),
            title = "L1 OUTPUT (FIFOB)",
            xlabel = "FIFO Occupancy",
            ylabel = "Hits",
            res = False, fit = False)
    fig.add_subplot(245).set_yscale('log')
    fig.add_subplot(245).text(0.99,0.97,(("L1_B FIFO reached %.1f %%" % \
                                            (WC_L1_B_FIFO))),
                                            fontsize=8,
                                            verticalalignment='top',
                                            horizontalalignment='right',
                                            transform=fig.add_subplot(245).transAxes)
    fit(latency,50)
    fit.plot(axis = fig.add_subplot(243),
            title = "Data Latency",
            xlabel = "Latency in nanoseconds",
            ylabel = "Hits",
            res = False)
    fig.add_subplot(243).text(0.99,0.8,(("WORST LATENCY = %d ns" % \
                                            (max(latency)))),
                                            fontsize=7,
                                            verticalalignment='top',
                                            horizontalalignment='right',
                                            transform=fig.add_subplot(243).transAxes)
    new_axis = fig.add_subplot(247)
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
                            (out['ASICS']['lost_producers'].sum(),
                             out['ASICS']['lost_channels'].sum(),
                             out['ASICS']['lost_outlink'].sum(),
                             np.array(out['L1']['lostL1b']).sum())
                            ),
                            fontsize=8,
                            verticalalignment='top',
                            horizontalalignment='left',
                            transform=new_axis.transAxes)


    fit(out['compress'],int(np.max(out['compress'])))
    fit.plot(axis = fig.add_subplot(244),
            title = "Data Frame Lenght (Compression)",
            xlabel = "Number of QDC fields",
            ylabel = "Hits",
            res = False,
            fit = False)

    # TOTAL NUMBER OF BITS vs COMPRESS EFFICIENCY
    A = np.arange(0,int(np.max(out['compress'])),1)
    D_data = A * 26 + + 8 + 10 + 8     #see DAQ_infinity
    D_save = (A-1)*10
    B_data = np.multiply(D_data,fit.hist)
    B_save = np.multiply(D_save,fit.hist)
    B_save[0]=0
    B_save[1]=0
    new_axis_2 = fig.add_subplot(248)
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

    # Write output to file
    DAQ_dump = HF.DAQ_IO(CG['ENVIRONMENT']['path_to_files'],
                    CG['ENVIRONMENT']['file_name'],
                    CG['ENVIRONMENT']['file_name']+"0.h5",
                    CG['ENVIRONMENT']['out_file_name']+"_"+ file_name + ".h5")
    logs = {  'logA':out['L1']['logA'],
              'logB':out['L1']['logB'],
              'log_channels':out['ASICS']['log_channels'],
              'log_outlink': out['ASICS']['log_outlink'],
              'in_time': out['L1']['in_time'],
              'out_time': out['L1']['out_time'],
              'lost':{  'producers':out['ASICS']['lost_producers'].sum(),
                        'channels' :out['ASICS']['lost_channels'].sum(),
                        'outlink'  :out['ASICS']['lost_outlink'].sum(),
                        'L1b'      :np.array(out['L1']['lostL1b']).sum()
                      },
              'compress':out['compress']
            }

    DAQ_dump.write_out(out['data'],topology,logs)


    plt.savefig(CG['ENVIRONMENT']['out_file_name']+"_"+ file_name + ".pdf")
    #plt.show()
