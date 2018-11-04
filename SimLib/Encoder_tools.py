import numpy as np
import pywt
import pywt.data


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
        self.THRESHOLD   = 0

    def sigmoid(self, x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))


    def encoder(self,L1,data,diff_threshold):

        data_enc_aux = self.sigmoid(np.dot(data,self.COMP['ENC_weights_A']) + self.COMP['ENC_bias_A'].T)

        cond_TENC = np.abs((data_enc_aux - self.THRESHOLD)) > diff_threshold*self.THRESHOLD
        data_enc_aux = (data_enc_aux-self.THRESHOLD)*cond_TENC

        return data_enc_aux


    def decoder(self,L1_SiPM,data_enc,index_1):

        L1_size_compressed = self.COMP['DEC_weights_A'].shape[0]
        index_2 = index_1 + L1_size_compressed
        data_recons_event = data_enc[index_1:index_2] #+ enc_threshold[0,index_1:index_2]
        recons_event = np.dot(data_recons_event,self.COMP['DEC_weights_A']) + self.COMP['DEC_bias_A'].T
        recons_event = recons_event.reshape((self.COMP['DEC_weights_A'].shape[1]//self.n_rows),self.n_rows)

        index_1 = index_2

        return index_1,recons_event.T


class encoder_tools_Z(object):

    def __init__(self,**param):
        self.n_rows      = param['n_rows']
        self.COMP        = param['COMP']
        #self.L1          = param['L1']
        self.TE2         = param['TE2']
        self.n_sensors   = param['n_sensors']
        self.THRESHOLD   = 0

    def sigmoid(self, x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))


    def encoder(self,L1,data,diff_threshold):

        data_enc_aux = self.sigmoid(np.dot(data/100,self.COMP['ENC_weights_A']) + self.COMP['ENC_bias_A'].T)

        cond_TENC = np.abs((data_enc_aux - self.THRESHOLD)) > diff_threshold*self.THRESHOLD
        data_enc_aux = (data_enc_aux-self.THRESHOLD)*cond_TENC

        return data_enc_aux


    def decoder(self,L1_SiPM,data_enc,index_1):

        L1_size_compressed = self.COMP['DEC_weights_A'].shape[0]
        index_2 = index_1 + L1_size_compressed

        data_recons_event = data_enc[index_1:index_2] + self.THRESHOLD

        recons_event = self.sigmoid(np.dot(data_recons_event,self.COMP['DEC_weights_A']) + self.COMP['DEC_bias_A'].T)
        recons_event = recons_event.reshape((self.COMP['DEC_weights_A'].shape[1]//self.n_rows),self.n_rows)

        index_1 = index_2

        return index_1,100*recons_event.T



class encoder_tools_N(object):

    def __init__(self,**param):
        self.n_rows      = param['n_rows']
        self.COMP        = param['COMP']
        #self.L1          = param['L1']
        self.TE2         = param['TE2']
        self.n_sensors   = param['n_sensors']
        self.THRESHOLD   = 0

    def sigmoid(self, x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))


    def encoder(self,L1,data,diff_threshold):

        data = (data-self.COMP['minA'].transpose())/ \
               (self.COMP['maxA'].transpose()-self.COMP['minA'].transpose())
        data_enc_aux = self.sigmoid(np.dot(data,self.COMP['ENC_weights_A']) + self.COMP['ENC_bias_A'].T)

        cond_TENC = np.abs((data_enc_aux - self.THRESHOLD)) > diff_threshold*self.THRESHOLD
        data_enc_aux = (data_enc_aux-self.THRESHOLD)*cond_TENC

        return data_enc_aux


    def decoder(self,L1_SiPM,data_enc,index_1):

        L1_size_compressed = self.COMP['DEC_weights_A'].shape[0]
        index_2 = index_1 + L1_size_compressed
        data_recons_event = data_enc[index_1:index_2] + self.THRESHOLD #enc_threshold[0,index_1:index_2]
        recons_event = self.sigmoid(np.dot(data_recons_event,self.COMP['DEC_weights_A']) + self.COMP['DEC_bias_A'].T)
        recons_event = recons_event*(self.COMP['maxA'].transpose()-self.COMP['minA'].transpose())\
                       + self.COMP['minA'].transpose()
        recons_event = recons_event.reshape((self.COMP['DEC_weights_A'].shape[1]//self.n_rows),self.n_rows)

        index_1 = index_2

        return index_1,recons_event.T


class encoder_tools_W(object):

    def __init__(self,**param):
        self.n_rows      = param['n_rows']
        self.COMP        = param['COMP']
        #self.L1          = param['L1']
        self.TE2         = param['TE2']
        self.n_sensors   = param['n_sensors']
        self.THRESHOLD   = 0
        self.L1_size_compressed = 0
        self.base = self.COMP['base']


    def encoder(self,L1,data,diff_threshold):

        aux_data = data.reshape(-1,self.n_rows).T

        for i in range(1):
            aux_data,(aux1,aux2,aux3) = pywt.dwt2(aux_data, self.base)

        cond_TENC = aux_data > diff_threshold
        aux_data = aux_data*cond_TENC

        self.L1_size_compressed = aux_data.shape

        return aux_data.reshape(1,-1)


    def decoder(self,L1_SiPM,data_enc,index_1):

        L1_size_compressed = self.L1_size_compressed[0]*self.L1_size_compressed[1]
        index_2 = index_1 + L1_size_compressed
        data_recons_event = data_enc[index_1:index_2].reshape(self.L1_size_compressed[0],-1)

        for i in range(1):
            data_recons_event = pywt.idwt2(
                            (data_recons_event,
                            (np.zeros(data_recons_event.shape),
                            np.zeros(data_recons_event.shape),
                            np.zeros(data_recons_event.shape))), self.base)

        recons_event = data_recons_event #.reshape(self.n_rows,-1)

        index_1 = index_2

        return index_1,recons_event


class encoder_tools_PW(object):

    def __init__(self,**param):
        self.n_rows      = param['n_rows']
        #self.L1          = param['L1']
        self.TE2         = param['TE2']
        self.n_sensors   = param['n_sensors']
        self.THRESHOLD   = 0
        self.L1_size_compressed = 0
        self.base = param['base']


    def encoder(self,L1,data):

        aux_data = data.reshape(-1,self.n_rows).T

        LL_aux,(LH_aux,HL_aux,HH_aux) = pywt.dwt2(aux_data, self.base)

        self.L1_size_compressed = LL_aux.shape

        return [LL_aux.reshape(1,-1),LH_aux.reshape(1,-1),HL_aux.reshape(1,-1)]


    def decoder(self,L1_SiPM,enc_LL,enc_LH,enc_HL,index_1):

        L1_size_compressed = self.L1_size_compressed[0]*self.L1_size_compressed[1]
        index_2 = index_1 + L1_size_compressed
        rec_LL = enc_LL[index_1:index_2].reshape(self.L1_size_compressed[0],-1)
        rec_LH = enc_LH[index_1:index_2].reshape(self.L1_size_compressed[0],-1)
        rec_HL = enc_HL[index_1:index_2].reshape(self.L1_size_compressed[0],-1)

        data_recons_event = pywt.idwt2(
                            [rec_LL,[rec_LH,rec_HL,np.zeros(rec_LL.shape)]], self.base)

        recons_event = data_recons_event #.reshape(self.n_rows,-1)

        index_1 = index_2

        return index_1,recons_event
