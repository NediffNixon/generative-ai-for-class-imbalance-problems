#imports
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Union

import optuna as opt
from gan_sampler.utils import StopOnNaNCallback
from gan_sampler.smote_wgangp import SMOTE_WGANGP
from gan_sampler.wgangp_resample import WGANGP_RESAMPLE
from gan_sampler.dcgan_resample import DCGAN_RESAMPLE
from gan_sampler.smote_dcgan import SMOTE_DCGAN

from gan_sampler.utils import build_generator
from gan_sampler.utils import build_discriminator
from gan_sampler.utils import build_critic

#DCGAN optuna objective function
def dcgan_objective(trial,
                     X_train:Union[pd.DataFrame,np.ndarray],
                     y_train:Union[pd.DataFrame,np.ndarray],
                     trad_oversampling:Union[None,str]=None,
                     enable_pruning:bool=True,
                     batch_size:int=32):
    """Optuna objective function for DCGAN hyper-parameter tuning.

    Args:
        trial (optuna.Trial): Optuna trial.
        X_train (Union[pd.DataFrame,np.ndarray]) of shape (n_samples, n_features): Training input features.
        y_train (Union[pd.DataFrame,np.ndarray]) of shape (n_samples,): The target values (class labels) of training set.
        trad_oversampling (Union[None,str], optional): Accepts 'smote', 'bsmote' or 'adasyn'. Defaults to None.
        enable_pruning (bool, optional): Optuna pruning is enabled if set to True. Defaults to True.
        batch_size (int, optional): Training batch size. Defaults to 32.

    Returns:
        total_loss of each trial.
    """
    #clear tensorflow session
    tf.keras.backend.clear_session()
    
    #input dimension
    inp_dim=X_train.shape[1]
    #hyper-parameters
    gc_lr = trial.suggest_float("gc_learning_rate", 1e-5, 1e-2, log=True)
    layers= trial.suggest_int("generator_layers",3,15)
    callbacks=[]
    #callback
    callback1=tf.keras.callbacks.EarlyStopping(monitor = 'total_loss', 
                                          min_delta=0.1, 
                                          patience=50, 
                                          mode='min',
                                          restore_best_weights=True,
                                          start_from_epoch=100)
    #NaN callback
    callback2=StopOnNaNCallback()
    
    callbacks.append([callback1,callback2])
    #keras pruning callback
    if enable_pruning:
        callback3=opt.integration.TFKerasPruningCallback(trial,'total_loss')
        callbacks.append(callback3)
    
    #DCGAN object
    generator=build_generator(inp_dim,layers=layers)
    discriminator=build_discriminator(inp_dim)
    
    if isinstance(trad_oversampling,str):
        sm_gan=SMOTE_DCGAN(generator,discriminator,oversampler=trad_oversampling)
    else:
        sm_gan=DCGAN_RESAMPLE(generator,discriminator)

    #compile dcgan
    sm_gan.compile(
        g_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=gc_lr),
        d_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=gc_lr)
    )
    
    #train the dcgan and get the training history.
    *_,his=sm_gan.fit_resample(
        X_train,y_train,
        epochs=1000,
        batch_size=batch_size,
        hyper_param_tuning=True,
        callbacks=callbacks,
        verbose=0
    )
    #store dcgan generator for each trial
    trial.set_user_attr('dcgan_generator',generator)
    
    #check for early stopping callback trigger
    if callback1.stopped_epoch !=0:
        ep=callback1.stopped_epoch - 50
        return np.round(abs(his['total_loss'].iloc[ep]),3)
    return np.round(abs(his['total_loss'].iloc[-1]),3)


#WGANGP optuna objective function
def wgangp_objective(trial,
                     X_train:Union[pd.DataFrame,np.ndarray],
                     y_train:Union[pd.DataFrame,np.ndarray],
                     trad_oversampling:Union[None,str]=None,
                     enable_pruning:bool=True,
                     batch_size:int=32):
    """Optuna objective function for WGANGP hyper-parameter tuning.

    Args:
        trial (optuna.Trial): Optuna trial.
        X_train (Union[pd.DataFrame,np.ndarray]) of shape (n_samples, n_features): Training input features.
        y_train (Union[pd.DataFrame,np.ndarray]) of shape (n_samples,): The target values (class labels) of training set.
        trad_oversampling (Union[None,str], optional): Accepts 'smote', 'bsmote' or 'adasyn'. Defaults to None.
        enable_pruning (bool, optional): Optuna pruning is enabled if set to True. Defaults to True.
        batch_size (int, optional): Training batch size. Defaults to 32.

    Returns:
        total_loss of each trial.
    """
    
    #clear tensorflow session
    tf.keras.backend.clear_session()
    
    #input dimension
    inp_dim=X_train.shape[1]
    #hyper-parameters
    critic_steps=trial.suggest_int("critic_steps", 3, 7)
    gp_weight=trial.suggest_float("gp_weight", 2.0, 20.0, step=0.5)
    gc_lr = trial.suggest_float("gc_learning_rate", 1e-5, 1e-2, log=True)
    layers= trial.suggest_int("generator_layers",3,15)
    callbacks=[]
    #callback
    callback1=tf.keras.callbacks.EarlyStopping(monitor = 'total_loss', 
                                          min_delta=0.1, 
                                          patience=50, 
                                          mode='min',
                                          restore_best_weights=True,
                                          start_from_epoch=100)
    #NaN callback
    callback2=StopOnNaNCallback()
    
    callbacks.append([callback1,callback2])
    #keras pruning callback
    if enable_pruning:
        callback3=opt.integration.TFKerasPruningCallback(trial,'total_loss')
        callbacks.append(callback3)
    
    #WGANGP object
    generator=build_generator(inp_dim,layers=layers)
    critic=build_critic(inp_dim)
    
    if isinstance(trad_oversampling,str):
        sm_gan=SMOTE_WGANGP(generator,critic,oversampler=trad_oversampling,critic_steps=critic_steps,gp_weight=gp_weight)
    else:
        sm_gan=WGANGP_RESAMPLE(generator,critic,critic_steps=critic_steps,gp_weight=gp_weight)

    #compile wgangp
    sm_gan.compile(
        g_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=gc_lr),
        c_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=gc_lr)
    )
    
    #train wgangp and get the training history
    *_,his=sm_gan.fit_resample(
        X_train,y_train,
        epochs=1000,
        batch_size=batch_size,
        hyper_param_tuning=True,
        callbacks=callbacks,
        verbose=0
    )
    
    #store WGANGP generator for each trial
    trial.set_user_attr('wgan_generator',generator)
    
    #check for early stopping callback trigger
    if callback1.stopped_epoch !=0:
        ep=callback1.stopped_epoch - 50
        return np.round(abs(his['total_loss'].iloc[ep]),3)
    return np.round(abs(his['total_loss'].iloc[-1]),3)
