import argparse
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True  
session = tf.Session( config=config )

import math
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import numpy as np

from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential,Model,Input
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Multiply,Add,Lambda,Dot,Concatenate,Subtract,Flatten,Dropout
from keras.optimizers import Adamax,Adam,SGD
from keras.regularizers import l2
from keras.constraints import non_neg
from keras.initializers import VarianceScaling,TruncatedNormal
from keras.layers.advanced_activations import LeakyReLU

import readline
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import os,re
import scipy.io as sio  
pandas2ri.activate()
ro.r('library(forecast)')



def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuCast.")
    parser.add_argument('--loc_id', type=int, default=0,
                        help='location_id.')
    parser.add_argument('--method', type=str, default='hw')
    return parser.parse_args()

# In[4]:
if __name__ == '__main__':
    args = parse_args()
    loc_id = args.loc_id
    method = args.method
    train_data = pd.read_csv('data/SC.csv')
    train_data = train_data.dropna()
    train_data = train_data.sample(frac=1)
    train_data.tem_ind = (train_data.tem_ind-train_data.tem_ind.min())/(train_data.tem_ind.max()-train_data.tem_ind.min())
    train_data.rain_ind = (train_data.rain_ind-train_data.rain_ind.min())/(train_data.rain_ind.max()-train_data.rain_ind.min())

    # In[6]:
    train_data = train_data[train_data.loc_ind==loc_id]
    train_data = train_data[(~train_data.p.isnull())&(~train_data.q.isnull())]
    pre_day = 0
    day_pred = 5
    test_data = train_data[(train_data.date_ind >= train_data.date_ind.max()- pre_day - day_pred)&(train_data.date_ind < train_data.date_ind.max() - pre_day)]
    train_data = train_data[train_data.date_ind < train_data.date_ind.max() - pre_day - day_pred]
    n_epochs = 100
    n_days = int(train_data[['date_ind']].max()+1+day_pred)
    n_hours = int(train_data[['hour']].max()+1)
    n_tem = int(train_data[['tem_ind']].max()+1)
    n_rain = int(train_data[['rain_ind']].max()+1)
    k_factors = 16
    d_factors = 4
    lr = 0.0005
    batch_size = 32
    c=0
    alpha = 0.2
    np.random.seed(1337)

    def get_model(n_days,n_hours,d_factors=4,k_factors=16,n_layers=1, c=0, ext=1):
        Dinp = Input((1,))
        D = Embedding(n_days, d_factors, input_length=1,name='DayEmbedding',embeddings_regularizer=l2(c))(Dinp)
        D = Flatten()(D)

        Hinp = Input((1,))
        H = Embedding(n_hours, d_factors, input_length=1,name='HourEmbedding',embeddings_regularizer=l2(c))(Hinp)
        H = Flatten()(H)

        Tinp = Input((1,))
        Rinp = Input((1,))


        latent_layer = Concatenate()([D,H])
        for i in range(n_layers):
            latent_layer = Dense(k_factors)(latent_layer)
            latent_layer = LeakyReLU(alpha=0.1)(latent_layer)

        if ext!=0:
            external_layer = Concatenate()([Tinp,Rinp])
            for i in range(n_layers - 1):
                external_layer = Dense(k_factors)(external_layer)
                external_layer = LeakyReLU(alpha=0.1)(external_layer)
            external_layer = Dense(2)(external_layer)
            external_layer = LeakyReLU(alpha=0.1)(external_layer)
            out_layer = Concatenate()([latent_layer,external_layer])
        else:
            out_layer = latent_layer
            
        for i in range(n_layers):
            out_layer = Dense(k_factors)(out_layer)
            out_layer = LeakyReLU(alpha=0.1)(out_layer)
        out_layer = Dense(2,name='out')(out_layer)
        out_layer = LeakyReLU(alpha=0.1)(out_layer)
        
        model = Model([Dinp,Hinp,Tinp,Rinp],[out_layer])
        return model
    def execCmd(cmd):  
        r = os.popen(cmd)  
        text = r.read()  
        r.close()  
        return text 

    def mse(y_true, y_pred):
        return K.mean(K.square(y_true-y_pred))
    def rmse(y_true, y_pred):
        return mse(y_true, y_pred) ** 0.5


    class TrainStop(Callback):
        def __init__(self, monitor='loss',thresh=0.0001,patience = 2,least = 10):
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
            self.epoch+=1
            if self.last_loss - logs.get('loss')>=self.thresh:
                self.last_loss = logs.get('loss')
            else:
                self.count+=1
                if self.count >= self.patience:
                    if self.epoch>self.least:
                        self.model.stop_training = True


    # first epoch factor modeling                    
    model = get_model(n_days,n_hours,d_factors=d_factors,k_factors=k_factors)
    adam = Adam(lr=lr)
    model.compile(loss='mse',optimizer=adam,metrics=[rmse])
    day = train_data[['date_ind']].as_matrix()
    hour = train_data[['hour']].as_matrix()
    tem = train_data[['tem_ind']].as_matrix()
    rain = train_data[['rain_ind']].as_matrix()
    P = train_data[['p','q']].as_matrix()
    callbacks = [TrainStop()]
    model.fit([day, hour,tem,rain], [P], batch_size=batch_size, epochs=n_epochs,  verbose=0, callbacks=callbacks)

    
    # first epoch seasonal smoothing
    dayEmb_forecast=pd.DataFrame()
    for i in range(d_factors):
        pdf = pd.DataFrame(model.get_weights()[0])[i][:-day_pred]
        rdf = pandas2ri.py2ri(pdf)
        ro.globalenv['r_timeseries'] = rdf
        if method =='hw':
            pred = ro.r('as.data.frame(hw(ts(r_timeseries,frequency = 7),h = ' + str(day_pred) + '))')
        elif method =='sar':
            pred = ro.r('as.data.frame(forecast(auto.arima(ts(r_timeseries,frequency = 7 )),h = ' + str(day_pred) + '))')
        else:
            pred = ro.r('as.data.frame(forecast(auto.arima(r_timeseries),h = ' + str(day_pred) + '))')
        dayEmb_forecast[i]=pred['Point Forecast']
    dayEmb_forecast = dayEmb_forecast.reset_index().drop(labels = 'index',axis = 1)

    
    weights = model.get_weights()
    weights[0] = np.vstack([model.get_weights()[0][:-day_pred],dayEmb_forecast.as_matrix()])
    model.set_weights(weights)

    # first epoch forecast without high-level purification
    test_data['scale_p'] = test_data.loc_p_max-test_data.loc_p_min
    test_data['scale_q'] = test_data.loc_q_max-test_data.loc_q_min
    d = test_data.date_ind.as_matrix()
    h = test_data.hour.as_matrix()
    t = test_data.tem_ind.as_matrix()
    r = test_data.rain_ind.as_matrix()
    pred = pd.DataFrame(model.predict([d,h,t,r]))
    test_data['p_pred'] = pred[0].as_matrix()
    test_data['q_pred'] = pred[1].as_matrix()

    # high-level purification
    ws = pd.DataFrame(model.get_weights()[0])[list(range(d_factors))]
    ws.to_csv('autoplait/_elecdat/dayEmb',header=None,index=None,sep=' ')
    os.chdir('autoplait/')
    cmd = 'sh elec.sh '+str(d_factors)+' '+str(alpha)
    execCmd(cmd)
    os.chdir('../')


    candidate_set = []
    flag = 0
    seg_cnt = 0
    for root,folds,files in os.walk('autoplait/_out/dat_tmp/dat1/'):
        for f in files:
            if len(re.findall(r'^segment.[\d]',f))!=0:
                with open(root+f) as fr:
                    seg_cnt += 1
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
        boolexp = boolexp|((train_data.date_ind <= r[1] )&(train_data.date_ind > r[0]))

    selected_train = train_data[boolexp]
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



    # second epoch factor modeling 
    n_epochs = 10 
    n_days = int(selected_train[['date_ind_new']].max()+1+day_pred)
    lr /=10
    model = get_model(n_days,n_hours,d_factors=d_factors,k_factors=k_factors)
    model.compile(loss='mse',optimizer=adam,metrics=[rmse])
    init_weights_0=[]
    for r in candidate_set:
        init_weights_0.append(weights[0][r[0]+1:r[1]+1])
    init_weights_0 = np.vstack(init_weights_0)
    weights[0] = init_weights_0
    model.set_weights(weights)


    day = selected_train[['date_ind_new']].as_matrix()
    hour = selected_train[['hour']].as_matrix()
    tem = selected_train[['tem_ind']].as_matrix()
    rain = selected_train[['rain_ind']].as_matrix()
    P = selected_train[['p','q']].as_matrix()
    callbacks = [TrainStop()]
    model.fit([day, hour,tem,rain], [P], batch_size=batch_size ,epochs=n_epochs,  verbose=0, callbacks=callbacks)


	# second epoch seasonal-smoothing 
    dayEmb_forecast=pd.DataFrame()
    for i in range(d_factors):
        pdf = pd.DataFrame(model.get_weights()[0])[i][:-day_pred]
        rdf = pandas2ri.py2ri(pdf)
        ro.globalenv['r_timeseries'] = rdf
        if method =='hw':
            pred = ro.r('as.data.frame(hw(ts(r_timeseries,frequency = 7),h = ' + str(day_pred) + '))')
        elif method =='sar':
            pred = ro.r('as.data.frame(forecast(auto.arima(ts(r_timeseries,frequency = 7 )),h = ' + str(day_pred) + '))')
        else:
            pred = ro.r('as.data.frame(forecast(auto.arima(r_timeseries),h = ' + str(day_pred) + '))')
        dayEmb_forecast[i]=pred['Point Forecast']
    dayEmb_forecast = dayEmb_forecast.reset_index().drop(labels = 'index',axis = 1)
    weights = model.get_weights()
    weights[0] = np.vstack([model.get_weights()[0][:-day_pred],dayEmb_forecast.as_matrix()])
    model.set_weights(weights)



    temp = selected_train[selected_train.date_ind == selected_train.date_ind.max()][['date_ind','date_ind_new']].drop_duplicates()
    dif = int(temp.date_ind - temp.date_ind_new)


    test_data['scale_p'] = test_data.loc_p_max-test_data.loc_p_min
    test_data['scale_q'] = test_data.loc_q_max-test_data.loc_q_min
    test_data['date_ind_new'] = test_data.date_ind- dif
    d = test_data.date_ind_new.as_matrix()
    h = test_data.hour.as_matrix()
    t = test_data.tem_ind.as_matrix()
    r = test_data.rain_ind.as_matrix()

    # forecasting
    pred = pd.DataFrame(model.predict([d, h, t, r]))
    test_data['p_pred_ap'] = pred[0].as_matrix()
    test_data['q_pred_ap'] = pred[1].as_matrix()
    rmseII_p = np.sqrt(np.mean(((test_data.p_pred_ap-test_data.p)*test_data.scale_p)**2))
    rmseII_q = np.sqrt(np.mean(((test_data.q_pred_ap-test_data.q)*test_data.scale_q)**2))

    test_data['p_pred'] = test_data['p_pred'] * test_data['scale_p'] + test_data['loc_p_min']
    test_data['q_pred'] = test_data['q_pred'] * test_data['scale_q'] + test_data['loc_q_min']
    test_data['p_pred_ap'] = test_data['p_pred_ap'] * test_data['scale_p'] + test_data['loc_p_min']
    test_data['q_pred_ap'] = test_data['q_pred_ap'] * test_data['scale_q'] + test_data['loc_q_min']
    test_data[['date_ind','hour','p_pred','q_pred','p_pred_ap','q_pred_ap']].to_csv('result/SC_neucast_%s_loc%d.csv'%(method,loc_id),index = False)




