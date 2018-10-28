import numpy as np



class encoder_tools2(object):

    def __init__(self,**param):
        self.n_rows      = param['n_rows']
        self.COMP        = param['COMP']
        #self.L1          = param['L1']
        self.TE2         = param['TE2']
        self.n_sensors   = param['n_sensors']


    def sigmoid(self, x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))


    def encoder(self,L1,data,diff_threshold):

        if (data.shape[0]==self.COMP['ENC_weights_A'].shape[0]):
            data = (data-self.COMP['minA'].transpose())/ \
                   (self.COMP['maxA'].transpose()-self.COMP['minA'].transpose())
            data_enc_aux = self.sigmoid(np.dot(data,self.COMP['ENC_weights_A']) + self.COMP['ENC_bias_A'].T)
        else:
            data = (data-self.COMP['minB'].transpose())/ \
                   (self.COMP['maxB'].transpose()-self.COMP['minB'].transpose())
            data_enc_aux = self.sigmoid(np.dot(data,self.COMP['ENC_weights_B']) + self.COMP['ENC_bias_B'].T)

        cond_TENC = (data_enc_aux > diff_threshold)
        data_enc_aux = data_enc_aux*cond_TENC

        return data_enc_aux



    def decoder(self,L1_SiPM,data_enc,index_1):

        if (L1_SiPM.shape[1]==(self.COMP['DEC_weights_A'].shape[1]//self.n_rows)):
            L1_size_compressed = self.COMP['DEC_weights_A'].shape[0]
            index_2 = index_1 + L1_size_compressed
            data_recons_event = data_enc[index_1:index_2]
            recons_event = self.sigmoid(np.dot(data_recons_event,self.COMP['DEC_weights_A']) + self.COMP['DEC_bias_A'].T)
            recons_event = recons_event*(self.COMP['maxA'].transpose()-self.COMP['minA'].transpose())\
                           + self.COMP['minA'].transpose()
            recons_event = recons_event.reshape((self.COMP['DEC_weights_A'].shape[1]//self.n_rows),self.n_rows)

        else:
            L1_size_compressed = self.COMP['DEC_weights_B'].shape[0]
            index_2 = index_1 + L1_size_compressed
            data_recons_event = data_enc[index_1:index_2]
            recons_event = self.sigmoid(np.dot(data_recons_event,self.COMP['DEC_weights_B']) + self.COMP['DEC_bias_B'].T)
            recons_event = recons_event*(self.COMP['maxB'].transpose()-self.COMP['minB'].transpose())\
                           + self.COMP['minB'].transpose()
            recons_event = recons_event.reshape((self.COMP['DEC_weights_B'].shape[1]//self.n_rows),self.n_rows)

        index_1 = index_2

        return index_1,recons_event.T


class encoder_tools(object):

    def __init__(self,**param):
        self.n_rows      = param['n_rows']
        self.COMP        = param['COMP']
        #self.L1          = param['L1']
        self.TE2         = param['TE2']
        self.n_sensors   = param['n_sensors']


    def sigmoid(self, x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))


    def encoder(self,L1,data,diff_threshold):

        data_enc_aux = self.sigmoid(np.dot(data,self.COMP['ENC_weights_A']) + self.COMP['ENC_bias_A'].T)

        cond_TENC = (data_enc_aux > diff_threshold)
        data_enc_aux = data_enc_aux*cond_TENC

        return data_enc_aux


    def decoder(self,L1_SiPM,data_enc,index_1):

        L1_size_compressed = self.COMP['DEC_weights_A'].shape[0]
        index_2 = index_1 + L1_size_compressed
        data_recons_event = data_enc[index_1:index_2] #+ enc_threshold[0,index_1:index_2]
        recons_event = np.dot(data_recons_event,self.COMP['DEC_weights_A']) + self.COMP['DEC_bias_A'].T
        recons_event = recons_event.reshape((self.COMP['DEC_weights_A'].shape[1]//self.n_rows),self.n_rows)

        index_1 = index_2

        return index_1,recons_event.T


class encoder_tools_N(object):

    def __init__(self,**param):
        self.n_rows      = param['n_rows']
        self.COMP        = param['COMP']
        #self.L1          = param['L1']
        self.TE2         = param['TE2']
        self.n_sensors   = param['n_sensors']


    def sigmoid(self, x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))


    def encoder(self,L1,data,diff_threshold):

        data = (data-self.COMP['minA'].transpose())/ \
               (self.COMP['maxA'].transpose()-self.COMP['minA'].transpose())
        data_enc_aux = self.sigmoid(np.dot(data,self.COMP['ENC_weights_A']) + self.COMP['ENC_bias_A'].T)

        cond_TENC = (data_enc_aux > diff_threshold)
        data_enc_aux = data_enc_aux*cond_TENC

        return data_enc_aux


    def decoder(self,L1_SiPM,data_enc,index_1):

        L1_size_compressed = self.COMP['DEC_weights_A'].shape[0]
        index_2 = index_1 + L1_size_compressed
        data_recons_event = data_enc[index_1:index_2] #+ enc_threshold[0,index_1:index_2]
        recons_event = self.sigmoid(np.dot(data_recons_event,self.COMP['DEC_weights_A']) + self.COMP['DEC_bias_A'].T)
        recons_event = recons_event*(self.COMP['maxA'].transpose()-self.COMP['minA'].transpose())\
                       + self.COMP['minA'].transpose()
        recons_event = recons_event.reshape((self.COMP['DEC_weights_A'].shape[1]//self.n_rows),self.n_rows)

        index_1 = index_2

        return index_1,recons_event.T




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
