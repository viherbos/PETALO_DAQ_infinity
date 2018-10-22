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

        # cond_TENC = (data_enc_aux > diff_threshold)
        # data_enc_aux = data_enc_aux*cond_TENC

        return data_enc_aux


    def decoder(self,L1_SiPM,data_enc,index_1):

        L1_size_compressed = self.COMP['DEC_weights_A'].shape[0]
        index_2 = index_1 + L1_size_compressed
        data_recons_event = data_enc[index_1:index_2] #+ enc_threshold[0,index_1:index_2]
        recons_event = np.dot(data_recons_event,self.COMP['DEC_weights_A']) + self.COMP['DEC_bias_A'].T
        recons_event = recons_event.reshape((self.COMP['DEC_weights_A'].shape[1]//self.n_rows),self.n_rows)

        index_1 = index_2

        return index_1,recons_event.T
