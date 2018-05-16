
# In[2]:
import argparse
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session( config=config )


# In[3]:
import math
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import numpy as np

from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential,Model,Input
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Multiply,Add,Lambda,Dot,Concatenate,Subtract,Flatten,Dropout
from keras.optimizers import Adamax,Adam,SGD
from keras.regularizers import l2
from keras.constraints import non_neg
from keras.initializers import VarianceScaling,TruncatedNormal,Ones,Zeros
from keras.layers.advanced_activations import LeakyReLU

import readline
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import os,re
import scipy.io as sio  
pandas2ri.activate()
ro.r('library(forecast)')

def execCmd(cmd):  
    r = os.popen(cmd)  
    text = r.read()  
    r.close()  
    return text 
    
# In[4]:

def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuCast.")
    parser.add_argument('--end', type=int, default=23,
                        help='end.')
    parser.add_argument('--day_pred', type=int, default=5,
                        help='day_pred')
    parser.add_argument('--method', type=str, default='sar',
                        help='holt')
    parser.add_argument('--times', type=str, default=0,
                        help='times')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    end = args.end
    day_pred = args.day_pred
    times = args.times
    # In[5]:

    data = pd.read_csv('data/CMU.csv',header=None,names=['Vr','Vi','I_real','I_imag','tem'] )

    data['hour'] = data.index%24
    data['date_ind'] = data.index//24
    train_data = data
    train_data['I_imag_max'] = train_data['I_imag'].max()
    train_data['I_real_max'] = train_data['I_real'].max()
    train_data['I_imag_min'] = train_data['I_imag'].min()
    train_data['I_real_min'] = train_data['I_real'].min()
    train_data['tem_min'] = train_data['tem'].min()
    train_data['tem_max'] = train_data['tem'].max()
    train_data['q'] = (train_data['I_imag']-train_data['I_imag_min'])/(train_data['I_imag_max']-train_data['I_imag_min'])
    train_data['p'] = (train_data['I_real']-train_data['I_real_min'])/(train_data['I_real_max']-train_data['I_real_min'])
    train_data['t'] = (train_data['tem']-train_data['tem_min'])/(train_data['tem_max']-train_data['tem_min'])
    # In[7]:
    train_data = train_data[(train_data.date_ind<end)]
    test_data = train_data[(train_data.date_ind >= end - day_pred)]
    train_data = train_data[(train_data.date_ind< end - day_pred)]
    n_epochs = 100
    n_days = int(train_data[['date_ind']].max()+1+day_pred)
    n_hours = int(train_data[['hour']].max()+1)
    k_factors = 16
    d_factors = 4
    lr = 0.0005
    batch_size = 32
    c=0
    alpha = 0.2
    max_iter_K=2
    # In[8]:


    def get_model(n_days, n_hours, d_factors=2, k_factors=4, c=0):
        Dinp = Input((1,))
        D = Embedding(n_days, d_factors, input_length=1, name='DayEmbedding', embeddings_regularizer=l2(c))(Dinp)
        D = Flatten()(D)

        Hinp = Input((1,))
        H = Embedding(n_hours, d_factors, input_length=1, name='HourEmbedding', embeddings_regularizer=l2(c))(Hinp)
        H = Flatten()(H)

        Tinp = Input((1,))
        #     Rinp = Input((1,))


        latent_layer = Concatenate()([D, H])
        latent_layer = Dense(k_factors * 2)(latent_layer)
        latent_layer = LeakyReLU(alpha=0.1)(latent_layer)
        latent_layer = Dense(k_factors)(latent_layer)
        latent_layer = LeakyReLU(alpha=0.1)(latent_layer)

        external_layer = Tinp
        external_layer = Dense(k_factors * 2)(external_layer)
        external_layer = LeakyReLU(alpha=0.1)(external_layer)
        external_layer = Dense(k_factors)(external_layer)
        external_layer = LeakyReLU(alpha=0.1)(external_layer)

        out_layer = Concatenate()([latent_layer, external_layer])
        out_layer = Dense(k_factors * 2)(out_layer)
        out_layer = LeakyReLU(alpha=0.1)(out_layer)
        out_layer = Dense(k_factors)(out_layer)
        out_layer = LeakyReLU(alpha=0.1)(out_layer)
        out_layer = Dense(2, name='out')(out_layer)
        out_layer = LeakyReLU(alpha=0.1)(out_layer)

        model = Model([Dinp, Hinp, Tinp], [out_layer])
        return model

    class TrainStop(Callback):
        def __init__(self, monitor='loss', thresh=0.0001, patience=2, least=10):
            super(TrainStop, self).__init__()
            self.monitor = monitor
            self.thresh = thresh
            self.patience = patience
            self.count = 0
            self.least = least
            self.epoch = 0

        def on_train_begin(self, logs={}):
            self.last_loss = np.inf

        def on_epoch_end(self, epoch, logs={}):
            self.epoch += 1
            if self.last_loss - logs.get('loss') >= self.thresh:
                self.last_loss = logs.get('loss')
            else:
                self.count += 1
                if self.count >= self.patience:
                    if self.epoch > self.least:
                        self.model.stop_training = True

    def mse(y_true, y_pred):
        return K.mean(K.square(y_true-y_pred))
    def rmse(y_true, y_pred):
        return mse(y_true, y_pred) ** 0.5


    # In[ ]:

    
    train = train_data
    dif = 0

    
    for epoch in range(max_iter_K):
	# fit factor modeling
        day = train[['date_ind']].as_matrix()
        if epoch>0:
            n_days = int(train[['date_ind_new']].max()+1+day_pred)
            day =  train[['date_ind_new']].as_matrix()

        hour = train[['hour']].as_matrix()
        tem = train[['t']].as_matrix()
        P = train[['p','q']].as_matrix()
        
        
        model = get_model(n_days, n_hours, d_factors=d_factors,k_factors = k_factors)
        if (epoch>0):
            model.set_weights(weights)
        
        adam = Adam(lr=lr)
        model.compile(loss='mse', optimizer=adam, metrics=[rmse])
        callbacks = [TrainStop()]
        history = model.fit([day, hour, tem], [P], batch_size=batch_size, epochs=n_epochs, verbose=0, callbacks=callbacks)
        
        # seasonal smoothing
        dayEmb_forecast = pd.DataFrame()
        for i in range(d_factors):
            pdf = pd.DataFrame(model.get_weights()[0])[i][:-day_pred]
            rdf = pandas2ri.py2ri(pdf)
            ro.globalenv['r_timeseries'] = rdf
            if args.method =='hw':
                freq = 2
                pred = ro.r('as.data.frame(hw(ts(r_timeseries,frequency ='+ str(freq) +'),h = ' + str(day_pred) + '))')
            elif args.method =='sar':
                pred = ro.r('as.data.frame(forecast(auto.arima(ts(r_timeseries,frequency = 7 )),h = ' + str(day_pred) + '))')
            else:
                pred = ro.r('as.data.frame(forecast(auto.arima(r_timeseries),h = ' + str(day_pred) + '))')
            dayEmb_forecast[i] = pred['Point Forecast']
        dayEmb_forecast = dayEmb_forecast.reset_index().drop(labels='index', axis=1)

        weights = model.get_weights()
        weights[0] = np.vstack([model.get_weights()[0][:-day_pred], dayEmb_forecast.as_matrix()])
        model.set_weights(weights)
        if (epoch==max_iter_K-1): continue
        
        # high-level pattern purification
        ws = pd.DataFrame(model.get_weights()[0])[list(range(d_factors))]
        ws.to_csv('autoplait/_elecdat/dayEmb',header=None,index=None,sep=' ')
        os.chdir('autoplait/')
        cmd = 'sh elec.sh '+str(d_factors)+' '+str(alpha)
        execCmd(cmd)
        os.chdir('../')

        candidate_set = []
        flag = 0
        regime_cnt = 0
        for root,folds,files in os.walk('autoplait/_out/dat_tmp/dat1/'):
            for f in files:
                if len(re.findall(r'^segment.[\d]',f))!=0:
                    regime_cnt += 1
                    with open(root+f) as fr:
                        line = fr.readline()
                        while line!='':
                            stl = line.split()
                            candidate_set.append([int(s) for s in stl])
                            line = fr.readline()
                    for c in candidate_set:
                        if(test_data.date_ind.max() in c):
                            flag=1
                            break
                if flag==1:
                    break
                else:
                    candidate_set = []     
        if regime_cnt<=1:
            break
            
        # ensure priod
        def after_period_a(a,b,p):
            return b - int((b-a)/p)*p
        def before_period_a(a,b,p):
            return b - (int((b-a)/p)+1)*p
        if len(candidate_set) >=2:
            clen = len(candidate_set)
            for i in range(clen):
                if i!=0:
                    candidate_set[(clen-1)-i][1] = before_period_a(candidate_set[(clen-1)-i][1],candidate_set[(clen-1)-i+1][0],7)
                    
        boolexp = False
        for r in candidate_set:
            boolexp = boolexp|((train.date_ind <= r[1] )&(train.date_ind > r[0]))
        
        #select data
        selected_train = train[boolexp]
        selected_train['date_ind_new'] = pd.Series()
        for i,c in enumerate(candidate_set):
            csum = 0
            j=0
            while(j<i):
                csum += (candidate_set[j][1]-candidate_set[j][0])
                j += 1
            selected_train.loc[(selected_train.date_ind>c[0])&(selected_train.date_ind<=c[1]),'date_ind_new']=selected_train.date_ind-c[0]-1+csum
        selected_train = selected_train.sample(frac=1)
        selected_train.date_ind_new = selected_train.date_ind_new.astype(np.int)
        
        temp = selected_train[selected_train.date_ind == selected_train.date_ind.max()][['date_ind','date_ind_new']].drop_duplicates()
        dif += int(temp.date_ind - temp.date_ind_new)
        train = selected_train
        
        #change weights
        init_weights_0=[]
        for r in candidate_set:
            init_weights_0.append(weights[0][r[0]+1:r[1]+1])

        init_weights_0 = np.vstack(init_weights_0)
        weights[0] = init_weights_0
    

    
    # forecast
    test_data['scale_p'] = test_data.I_real_max - test_data.I_real_min
    test_data['scale_q'] = test_data.I_imag_max - test_data.I_imag_min
    test_data['date_ind_new'] = test_data.date_ind- dif
    
    d = test_data.date_ind_new.as_matrix()
    h = test_data.hour.as_matrix()
    t = test_data.t.as_matrix()




    pred = pd.DataFrame(model.predict([d,h,t]))
    test_data['p_pred']=pred[0].as_matrix()
    test_data['q_pred']=pred[1].as_matrix()
    test_data['p_pred'] = test_data['p_pred'] * test_data['scale_p'] + test_data.I_real_min
    test_data['q_pred'] = test_data['q_pred'] * test_data['scale_q'] + test_data.I_imag_min
    test_data = test_data.sort_values(['date_ind','hour'])
    test_data[['I_imag','I_real','p_pred','q_pred']].to_csv('result/neucast_'+str(end)+'_'+str(day_pred)+'_'+args.method+'_'+str(times)+'.csv',index = False)
    