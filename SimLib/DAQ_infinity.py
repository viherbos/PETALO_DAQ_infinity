import simpy
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from simpy.events import AnyOf, AllOf, Event
import sys
#import HF_translator as HFT
import os
import pandas as pd
import math

""" LIBRARY FOR INFINITY DAQ """

class Full(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class parameters(object):

    def __init__(self,data,sensors,n_events):
        self.P          = data
        self.sensors    = sensors
        self.events     = n_events


class ch_frame(object):

    def __init__(self,data,event,sensor_id,asic_id,in_time,out_time):
        self.data = data
        self.sensor_id = sensor_id
        self.event = event
        self.asic_id = asic_id
        self.in_time = in_time
        self.out_time = out_time

    def get_np_array(self):
        return np.array([self.data, self.event, self.sensor_id, self.asic_id,
                self.in_time, self.out_time])

    def put_np_array(self, nparray):
        aux_list = {'data'      :   nparray[0], 'event'     :   nparray[1],
                    'sensor_id' :   nparray[2], 'asic_id'   :   nparray[3],
                    'in_time'   :   nparray[4], 'out_time'  :   nparray[5]}

        self.data       = aux_list['data']
        self.sensor_id  = aux_list['sensor_id']
        self.event      = aux_list['event']
        self.asic_id    = aux_list['asic_id']
        self.in_time    = aux_list['in_time']
        self.out_time   = aux_list['out_time']

    def __repr__(self):
        return "data: {}, event: {}, sensor_id: {}, asic_id: {} in_time:{} out_time:{}".\
            format( self.data, self.event, self.sensor_id, self.asic_id,
                    self.in_time, self.out_time)


class L1_outframe(object):

    def __init__(self,data,event,asic_id,in_time,out_time):
        self.data = data
        # Lenght of data is not constant, depends on the number of channels being sent
        # DATA fiels: n_CH | TDC | SENSOR1 | QDC1 | SENSOR2 | QDC2 | ... | B_QDC
        self.event = event
        self.asic_id = asic_id
        self.in_time = in_time
        self.out_time = out_time

    def get_np_array(self):
        B = [ self.event, self.asic_id, self.in_time, self.out_time]
        return np.concatenate((self.data,B),axis=0)

    def get_dict(self):
        return {'data'      :self.data,
        # DATA fiels: n_CH | TDC | SENSOR1 | QDC1 | SENSOR2 | QDC2 | ... | B_QDC
                'event'     :self.event,
                'asic_id'   :self.asic_id,
                'in_time'   :self.in_time,
                'out_time'  :self.out_time}

    def __repr__(self):
        return "data: {}, event: {}, asic_id: {} in_time:{} out_time:{}".\
            format(self.data,self.event,self.asic_id,self.in_time,self.out_time)



class producer(object):
    """ Sends data to a given channel. DATA has 3 elements:
            Charge, in_time, out_time(0)
        Parameters
        env     : Simpy environment
        counter : Event counter
        lost    : FIFO drops counter (Channel Input FIFO)
        TE      : Energy threshold for channel filtering
        timing  : reads delay from previously generated vector
    """

    def __init__(self,env,data,timing,param,sensor_id,asic_id):
        self.env = env
        self.out = None
        # Connection with receptor
        self.action = env.process(self.run())
        self.counter = 0
        self.lost = 0
        self.data = data
        self.timing = timing
        self.TE = param.P['TOFPET']['TE']
        self.sensor_id = sensor_id
        self.asic_id = asic_id


    def run(self):
        while self.counter < len(self.data):

            yield self.env.timeout(int(self.timing[self.counter]))
            #print_stats(env,self.out.res)

            try:
                if self.data[self.counter]>self.TE:
                    self.DATA = ch_frame(data     = self.data[self.counter],
                                        event     = self.counter,
                                        sensor_id = self.sensor_id,
                                        asic_id   = self.asic_id,
                                        in_time   = self.env.now,
                                        out_time  = 0)
                    #np.array([self.data[self.counter],self.env.now,0])
                    # PACKET FRAME: [SENSOR_DATA, IN_TIME, OUT_TIME]
                    self.lost = self.out.put(self.DATA.get_np_array(),self.lost)
                self.counter += 1
                # Drop data. FIFO is FULL so data is lost
            except IndexError:
                print "List Empty"

    def __call__(self):
        output = {  'lost'   : self.lost
                    }
        return output


class FE_channel(object):
    """ ASIC channel model.
        Method
        put     : Input FIFO storing method
        Parameters
        env     : Simpy environment
        FIFO_size : Size of input FIFO (4)
        lost    : FIFO drops counter (output FIFO)
        gain    : channel QDC gain
        timing  : reads delay from previously generated vector
        latency : Wilkinson ADC latency (in terms of amplitude)
        log     : Stores log of items and time in input FIFO
    """

    def __init__(self,env,param,sensor_id):
        self.env = env
        self.FIFO_size = param.P['TOFPET']['IN_FIFO_depth']
        self.res = simpy.Store(self.env,capacity=self.FIFO_size)
        self.action = env.process(self.run())
        self.out = None
        self.latency = param.P['TOFPET']['MAX_WILKINSON_LATENCY']
        self.index = 0
        self.lost = 0
        self.gain = param.P['TOFPET']['TGAIN']
        self.log = np.array([]).reshape(0,2)
        self.sensor_id = sensor_id

    def print_stats(self):
        self.log=np.vstack([self.log,[len(self.res.items),self.env.now]])
        # FIFO Statistics

    def put(self,data,lost):
        try:
            if (len(self.res.items)<self.FIFO_size):
                self.res.put(data)
                self.print_stats()
                return lost
            else:
                raise Full('Channel FIFO is FULL')
        except Full as e:
            print ("TIME: %s // %s" % (self.env.now,e.value))
            return (lost+1)

    def run(self):
        while True:
            self.packet = yield self.res.get()
            self.msg = self.packet[0]
            self.wilk_delay = int((self.latency/1024)*self.msg*self.gain)
            if self.wilk_delay > self.latency:
                self.wilk_delay = self.latency
            yield self.env.timeout(self.wilk_delay)
            # Latency depends on Amplitude and FIFO status (!!!)
            # Analize dynamic range
            self.lost = self.out.put(self.packet,self.lost)

    def __call__(self):
        output = {  'lost'   : self.lost,
                    'log'    : self.log
                    }
        return output

class FE_outlink(object):
    """ ASIC Outlink model.
        Method
        put             : Output link FIFO storing method

        Parameters
        env             : Simpy environment
        FIFO_out_size   : Size of output FIFO
        latency         : Latency depends on output link speed
        log             : Stores time and number of FIFO elements
    """

    def __init__(self,env,param,asic_id):
        self.env = env
        self.FIFO_out_size = param.P['TOFPET']['OUT_FIFO_depth']
        self.res = simpy.Store(self.env,capacity=self.FIFO_out_size)
        self.action = env.process(self.run())
        self.latency = int(1E9/param.P['TOFPET']['outlink_rate'])
        self.log = np.array([]).reshape(0,2)
        self.asic_id = asic_id
        self.out = None
        self.lost = 0

    def print_stats(self):
        self.log=np.vstack([self.log,[len(self.res.items),self.env.now]])
        # FIFO Statistics

    def put(self,data,lost):
        try:
            if (len(self.res.items)<self.FIFO_out_size):
                self.res.put(data)
                self.print_stats()
                return lost
            else:
                raise Full('OUT LINK FIFO is FULL')
        except Full as e:
            print ("TIME: %s // %s" % (self.env.now,e.value))
            return (lost+1)

    def run(self):
        while True:
            yield self.env.timeout(self.latency)
            packet = yield self.res.get()
            self.lost = self.out.put(packet,self.lost)

    def __call__(self):
        output = {  'lost'   : self.lost,
                    'log'    : self.log
                    }
        return output


class FE_asic(object):
    """ ASIC model.
        Method

        Parameters
        sensor_id : Array with the positions of the sensors being used (param.sensors)
    """
    def __init__(self,env,param,data,timing,sensors,asic_id):
        self.env        = env
        self.param      = param
        self.DATA       = data
        self.timing     = timing
        self.sensors    = sensors
        self.asic_id    = asic_id
        self.n_ch       = len(sensors)


        # System Instanciation and Wiring
        self.Producer = [producer(   self.env,
                                data       = self.DATA[:,i],
                                timing     = self.timing,
                                param      = self.param,
                                sensor_id  = self.sensors[i],
                                asic_id    = self.asic_id)
                                            for i in range(self.n_ch)]
        self.Channels = [FE_channel( self.env,
                                param = self.param,
                                sensor_id = self.sensors[i])
                                            for i in range(self.n_ch)]
        self.Link     = FE_outlink(  self.env,
                                self.param,
                                asic_id = self.asic_id)

        for i in range(self.n_ch):
            self.Producer[i].out = self.Channels[i]
            self.Channels[i].out = self.Link


    def __call__(self):
        lost_producers = np.array([]).reshape(0,1)
        lost_channels  = np.array([]).reshape(0,1)
        log_channels   = np.array([]).reshape(0,2)

        for i in self.Producer:
            lost_producers = np.vstack([lost_producers, i()['lost']])

        for i in self.Channels:
            lost_channels = np.vstack([lost_channels, i()['lost']])
            log_channels  = np.vstack([log_channels, i()['log']])


        output = {  'lost_producers' : lost_producers,
                    'lost_channels'  : lost_channels,
                    'lost_outlink'   : self.Link()['lost'],
                    'log_channels'   : log_channels,
                    'log_outlink'    : self.Link()['log']
                    }
        return output



class L1(object):
    """ L1 model.
        Methods

        Parameters

    """
    def __init__(self,env,out_stream,param,L1_id):
        self.env        = env
        self.L1_id      = L1_id
        self.param      = param
        self.out_stream = out_stream
        self.latency    = int(1E9/param.P['L1']['L1_outrate'])
        self.fifoA      = simpy.Store(self.env,
                                    capacity=param.P['L1']['FIFO_L1a_depth'])
        self.fifoB      = simpy.Store(self.env,
                                    capacity=param.P['L1']['FIFO_L1b_depth'])
        self.buffer     = np.array([]).reshape(0,6)
        self.flag       = False
        self.frame_count = 0
        self.lostB      = 0
        self.action1    = env.process(self.runA())
        self.action2    = env.process(self.runB())
        self.logA = np.array([]).reshape(0,2)
        self.logB = np.array([]).reshape(0,2)

    def print_statsA(self):
        self.logA=np.vstack([self.logA,[len(self.fifoA.items),self.env.now]])
        # FIFO Statistics
    def print_statsB(self):
        self.logB=np.vstack([self.logB,[len(self.fifoB.items),self.env.now]])
        # FIFO Statistics

    def process_frames(self):
        out=[]
        while (self.buffer.shape[0]>0):
            time = self.buffer[0,4]
            cond = np.array(self.buffer[:,4]==time)
            buffer_sel = self.buffer[cond,:]
            #Select those with same IN_TIME

            data_frame = [-1,time]
            sum_QDC = 0
            n_ch = 0
            for i in buffer_sel[:,:]:
                if i[0] > self.param.P['L1']['TE']:
                    data_frame.append(i[2])
                    data_frame.append(i[0])
                    n_ch +=1
                else:
                    sum_QDC += i[0]
            if (n_ch == 0):
                data_frame.append(11111)
            # This is for any SiPM in the ASIC
            data_frame.append(sum_QDC)

            data_frame[0] = n_ch

            # Build Data frame

            out.extend([{'data'      :data_frame,
                        'event'     :self.buffer[0,1],
                        'asic_id'   :self.buffer[0,3],
                        'in_time'   :time,
                        'out_time'  :0}])

            #take all the used data out of the buffer
            cond_not = np.invert(cond)
            self.buffer = self.buffer[cond_not]

        return out


    def put(self,data,lost):
        try:
            if (len(self.fifoA.items)<(self.param.P['L1']['FIFO_L1a_depth'])):
                self.fifoA.put(data)
                self.print_statsA()
                return lost
            else:
                raise Full('L1 FIFO A is FULL')
        except Full as e:
            print ("TIME: %s // %s" % (self.env.now,e.value))
            return (lost+1)



    def runA(self):
        while True:
            frame = yield self.fifoA.get()

            if (self.frame_count < self.param.P['L1']['buffer_size']):
                self.buffer = np.pad(self.buffer,((1,0),(0,0)),mode='constant')
                self.buffer[0,:] = frame
                self.frame_count += 1

            if (self.frame_count == self.param.P['L1']['buffer_size']):
                out = self.process_frames()
                for i in out:
                    yield self.env.timeout(1.0E9/self.param.P['L1']['frame_process'])
                    self.lostB = self.putB(i,self.lostB)

                self.frame_count = 0
                #self.flag = False
                self.buffer = np.array([]).reshape(0,6)


    def putB(self,data,lost):
        try:
            if (len(self.fifoB.items)<(self.param.P['L1']['FIFO_L1b_depth'])):
                self.fifoB.put(data)
                self.print_statsB()
                return lost
            else:
                raise Full('L1 FIFO B is FULL')
        except Full as e:
            print ("TIME: %s // %s" % (self.env.now,e.value))
            return (lost+1)


    def runB(self):
        while True:
            msg = yield self.fifoB.get()
        # 8 bits n_CH | 10 bits TDC | n_CH * (16 bits + 10 bits) | 8 bits B_QDC
            # When no active channels -> send a sensor in the center of ASIC
            delay = float((msg['data'][0]*26 + 8 + 10 + 8))*(1.0E9/self.param.P['L1']['L1_outrate'])
            yield self.env.timeout(int(delay))
            msg['out_time'] = self.env.now
            self.out_stream.append(msg)


    def __call__(self):
        output = {  'lostL1b':self.lostB,
                    'logA'   :self.logA,
                    'logB'   :self.logB,
                    'data_out':self.out_stream}

        return output



def SiPM_Mapping(param, style):

    # Work out number of SiPMs based on geometry data
    n_sipms_I = param['TOPOLOGY']['sipm_int_row']*param['TOPOLOGY']['n_rows']
    n_sipms_O = param['TOPOLOGY']['sipm_ext_row']*param['TOPOLOGY']['n_rows']
    n_sipms     = n_sipms_I + n_sipms_O
    # Number of ASICs calculation: Inner Face + Outer Face // full + partial
    n_asics_I = int(math.ceil(float(n_sipms_I) / float(param['TOFPET']['n_channels'])))
    n_asics_f_I  = n_sipms_I // param['TOFPET']['n_channels']  # Fully used
    n_asics_p_I = n_asics_I - n_asics_f_I                       # Partially used
    n_asics_O = int(math.ceil(float(n_sipms_O) / float(param['TOFPET']['n_channels'])))
    n_asics_f_O  = n_sipms_O // param['TOFPET']['n_channels']
    n_asics_p_O   = n_asics_O - n_asics_f_O      # Number of not fully used ASICs (0 or 1)
    n_asics = n_asics_I + n_asics_O
    # L1 are required with max number of ASICs in param.P['L1']['n_asics']
    # // full + part
    n_L1 = int(math.ceil(float(n_asics) / float(param['L1']['n_asics'])))
    n_L1_f = n_asics // param['L1']['n_asics']
    n_L1_p = n_L1 - n_L1_f

    print ("Number of SiPM : %d \nNumber of ASICS : %d " % (n_sipms,n_asics))
    print ("Number of L1 : %d " % (n_L1))

    SiPM_Matrix_I = np.reshape(np.arange(0,n_sipms_I),
                                (param['TOPOLOGY']['n_rows'],
                                param['TOPOLOGY']['sipm_int_row']))
    SiPM_Matrix_O = np.reshape(np.arange(n_sipms_I,n_sipms),
                                (param['TOPOLOGY']['n_rows'],
                                param['TOPOLOGY']['sipm_ext_row']))
    # SiPM matrixs Inner face and Outer face

    topology = {'n_sipms_I':n_sipms_I, 'n_sipms_O':n_sipms_O, 'n_sipms': n_sipms,
            'n_asics_I':n_asics_I, 'n_asics_f_I':n_asics_f_I,'n_asics_p_I':n_asics_p_I,
            'n_asics_O':n_asics_O, 'n_asics_f_O':n_asics_f_O,'n_asics_p_O':n_asics_p_O,
            'n_asics':n_asics, 'n_L1':n_L1, 'n_L1_f':n_L1_f, 'n_L1_p':n_L1_p}

    # if style == "striped":
    #     L1_Slice=[]
    #     count = 0
    #     SiPM_ASIC_Slice=[]
    #     # Generate Slice of ASICs (SiPM) for L1
    #     for i in range(topology['n_asics_I']):
    #         if (count < param.P['L1']['n_asics']-1):
    #             SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_I[:,i*4:(i+1)*4],-1))
    #             count += 1
    #         else:
    #             SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_I[:,i*4:(i+1)*4],-1))
    #             L1_Slice.append(SiPM_ASIC_Slice)
    #             SiPM_ASIC_Slice=[]
    #             count = 0
    #     for i in range(topology['n_asics_O']):
    #         if (count < param.P['L1']['n_asics']-1):
    #             SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_O[:,i*4:(i+1)*4],-1))
    #             count += 1
    #         else:
    #             SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_O[:,i*4:(i+1)*4],-1))
    #             L1_Slice.append(SiPM_ASIC_Slice)
    #             SiPM_ASIC_Slice=[]
    #             count = 0
    #
    #     if (topology['n_L1_p'] == 1):
    #         L1_Slice.append(SiPM_ASIC_Slice)


    if style == "mixed":
        L1_Slice=[]
        SiPM_ASIC_Slice=[]
        count = 0

        for i in range(topology['n_asics_I']):
            if (count < param['L1']['n_asics']-2):
                SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_I[:,i*4:(i+1)*4],-1))
                SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_O[:,i*4:(i+1)*4],-1))
                count += 2
            else:
                SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_I[:,i*4:(i+1)*4],-1))
                SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_O[:,i*4:(i+1)*4],-1))
                L1_Slice.append(SiPM_ASIC_Slice)
                SiPM_ASIC_Slice=[]
                count = 0

        for i in range(topology['n_asics_O']-topology['n_asics_I']):
            if (count < param['L1']['n_asics']-1):
                SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_O[:,i*4:(i+1)*4],-1))
                count += 1
            else:
                SiPM_ASIC_Slice.append(np.reshape(SiPM_Matrix_O[:,i*4:(i+1)*4],-1))
                L1_Slice.append(SiPM_ASIC_Slice)
                SiPM_ASIC_Slice=[]
                count = 0

        if (topology['n_L1_p'] == 1):
            L1_Slice.append(SiPM_ASIC_Slice)


    if style == "striped_old":
        L1_Slice=[]
        L1_aux_Slice=[]
        count_ch   = 0
        count_asic = 0
        count_L1   = 0
        ASIC_Slice=[]
        SiPM_Slice=[]

        # Generate Slice of ASICs (SiPM) for L1
        for k in range(param['TOPOLOGY']['sipm_int_row']):
            for j in range(param['TOPOLOGY']['n_rows']):
                SiPM_Slice.append(SiPM_Matrix_I[j,k])
                count_ch += 1
                if count_ch == param['TOFPET']['n_channels']:
                    ASIC_Slice.append(SiPM_Slice)
                    SiPM_Slice = []
                    count_ch = 0
        if (count_ch > 0):
            ASIC_Slice.append(SiPM_Slice)

        count_ch = 0
        SiPM_Slice=[]
        for k in range(param['TOPOLOGY']['sipm_ext_row']):
            for j in range(param['TOPOLOGY']['n_rows']):
                SiPM_Slice.append(SiPM_Matrix_O[j,k])
                count_ch += 1
                if count_ch == param['TOFPET']['n_channels']:
                    ASIC_Slice.append(SiPM_Slice)
                    SiPM_Slice = []
                    count_ch = 0
        if (count_ch > 0):
            ASIC_Slice.append(SiPM_Slice)

        # Number of ASICs
        print ("CHECK Number of ASICS = %d" % (len(ASIC_Slice)))


        for i in range(len(ASIC_Slice)):
            L1_aux_Slice.append(ASIC_Slice[i])
            count_asic += 1
            if count_asic == param['L1']['n_asics']:
                L1_Slice.append(L1_aux_Slice)
                L1_aux_Slice = []
                count_asic = 0
        if count_asic > 0:
            L1_Slice.append(L1_aux_Slice)



    if style == "striped":
        L1_Slice=[]
        L1_aux_Slice=[]
        count_ch   = 0
        count_asic = 0
        count_L1   = 0
        ASIC_Slice=[]
        SiPM_Slice=[]

        # Generate Slice of ASICs (SiPM) for L1
        for k in range(param['TOPOLOGY']['sipm_int_row']):
            for j in range(param['TOPOLOGY']['n_rows']):
                SiPM_Slice.append(SiPM_Matrix_I[j,k])
                count_ch += 1
                if count_ch == param['TOFPET']['n_channels']:
                    ASIC_Slice.append(SiPM_Slice)
                    SiPM_Slice = []
                    count_ch = 0
        if (count_ch > 0):
            ASIC_Slice.append(SiPM_Slice)

        count_ch = 0
        SiPM_Slice=[]
        for k in range(param['TOPOLOGY']['sipm_ext_row']):
            for j in range(param['TOPOLOGY']['n_rows']):
                SiPM_Slice.append(SiPM_Matrix_O[j,k])
                count_ch += 1
                if count_ch == param['TOFPET']['n_channels']:
                    ASIC_Slice.append(SiPM_Slice)
                    SiPM_Slice = []
                    count_ch = 0
        if (count_ch > 0):
            ASIC_Slice.append(SiPM_Slice)

        # Number of ASICs
        print ("CHECK Number of ASICS = %d" % (len(ASIC_Slice)))


        for i in ASIC_Slice:
            L1_aux_Slice.append(i)
            count_asic += 1
            if count_asic == param['L1']['n_asics']:
                L1_Slice.append(L1_aux_Slice)
                L1_aux_Slice = []
                count_asic = 0
        if count_asic > 0:
            L1_Slice.append(L1_aux_Slice)



    print ("Number of Instanciated L1 = %d" % (len(L1_Slice)))
    for i in range(len(L1_Slice)):
        print ("L1 number %d has %d ASICs" % (i,len(L1_Slice[i])))


    return L1_Slice, SiPM_Matrix_I, SiPM_Matrix_O, topology
