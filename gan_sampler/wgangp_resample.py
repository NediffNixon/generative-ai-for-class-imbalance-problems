#imports
import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Union

from sklearn.preprocessing import PowerTransformer
from gan_sampler.utils import data_builder
from gan_sampler.utils import wasserstein_loss
from gan_sampler.utils import sliced_wasserstein_loss
from gan_sampler.wgangp import WGANGP

class WGANGP_RESAMPLE():
    """ WGANGP sampler class.
    Initalises the WGANGP architecture.

    Args:
    generator (tensorflow.keras.Model): Generator of WGANGP.
    critic (tensorflow.keras.Model): Critic of WGANGP.
    critic_steps (int, optional): number of training updates for critic per training of the generator. Defaults to 3.
    gp_weight (float, optional): weight for the gradient penalty. Defaults to 10.0.
        
    Methods:
        compile: compiles the optimizers for the generator and the critic of WGANGP.
        fit_resample: resamples the input to a balanced dataset.
    """
    #initialise class variables
    def __init__(self,
                 generator:tf.keras.Model,
                 critic:tf.keras.Model,
                 critic_steps:int=3,
                 gp_weight:float=10.0):
        
        self.generator=generator
        self.critic=critic
        self.critic_steps=critic_steps
        self.gp_weight=gp_weight
    
    #optimizers compiler
    def compile(self,
                g_optimizer:tf.keras.optimizers.Optimizer,
                c_optimizer:tf.keras.optimizers.Optimizer):
        
        """Compiles optimizers for WGANGP generator and critic.

        Args:
            g_optimizer (tensorflow.keras.optimizers.Optimizer): Generator optimizer.
            c_optimizer (tensorflow.keras.optimizers.Optimizer): Critic optimizer.
        """
        self.g_optimizer=g_optimizer
        self.c_optimizer=c_optimizer
        
        #function execution flag
        self.compile_executed=True
        
    #resampler function    
    def fit_resample(self,
                     X:Union[pd.DataFrame,np.ndarray],
                     y:Union[pd.DataFrame,np.ndarray],
                     epochs:int=100,
                     batch_size:int=32,
                     callbacks:Union[None,list]=None,
                     history:bool=True,
                     verbose:Union[int,str]='auto',
                     scale:bool=True,
                     hyper_param_tuning:bool=False,
                     wgan_eval_metric:Union[None,str]=None):
        
        """Resampler for WGANGP.

        Args:
            X (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples, n_features): Input feature samples.
            y (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples,): The target values (class labels).
            epochs (int, optional): Number of training epochs for WGANGP. Defaults to 100.
            batch_size (int, optional): Batch-size for WGANGP. Defaults to 32.
            callbacks (Union[None, list], optional): WGANGP callbacks. Supports Tensorflow callbacks. Defaults to None.
            history (bool, optional): Returns the WGANGP training history if set to True. Defaults to True.
            verbose (Union[int, str], optional): Tensorflow verbosity level for WGANGP training. Defaults to 'auto'.
            scale (bool, optional): Performs PowerTransform (Sklearn) on X if set to True and returns the fitted scaler object. Defaults to True.
            hyper_param_tuning (bool, optional): Runs in hyper-parameter mode and does not perform resampling if set to True. Defaults to False.
            wgan_eval_metric (Union[None, str], optional): Returns wasserstein loss if set to 'wass' or returns sliced wasserstein loss if set to 'sliced_wass'. Defaults to None.

        Raises:
            RuntimeError: If resampled before compiling WGANGP optimizers.

        Returns:
            Resampled Input features with balanced target class labels.
        """
        #check if compiled
        if not self.compile_executed:
            raise RuntimeError("Must be compiled before fitting!")
        
        #True parameters
        true_var=[]
        
        #scaling input features
        if scale:
            ss=PowerTransformer()
            ss.fit(X)
            X=ss.transform(X)
            true_var.append(ss)
        
        #real data and number of samples to generate   
        real_data=X[np.where(y==1)[0]]
        real_data_hyp=tf.convert_to_tensor(real_data,dtype=tf.float32)
        samples=tf.shape(X)[0] - (2*(tf.shape(real_data_hyp)[0]))
        real_data=data_builder(real_data_hyp,batch_size)
        
        #wgangp object
        wgan=WGANGP(self.critic,self.generator,syn_data=None,critic_steps=self.critic_steps,gp_weight=self.gp_weight,random_noise=True)
        
        #wgangp compiler
        wgan.compile(
            c_optimizer=self.c_optimizer,
            g_optimizer=self.g_optimizer
        )
        
        print("Training WGANGP model...")
        #fit the wgangp model
        wgan.fit(
            real_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        print("WGANGP model finished training...")
        
        # wgangp training history
        if history:
            his = pd.DataFrame(wgan.history.history)
            true_var.append(his)
            
        
        #If on hyper-parameter tuning mode, no need to resample.
        #calculate wasserstein loss or sliced wasserstein loss.
        if hyper_param_tuning:
            if wgan_eval_metric is not None:
                fake=tf.random.normal(
                    shape=tf.shape(real_data_hyp)
                )
                pred=self.generator.predict(fake)
                if wgan_eval_metric=='wass':
                    score=wasserstein_loss(real_data_hyp,pred,num_iter_max=int(1e+8))
                    true_var.append(score)
                    
                elif wgan_eval_metric=='sliced_wass':
                    score=sliced_wasserstein_loss(real_data_hyp,pred)
                    true_var.append(score)
                    
            return true_var
        
        print("Resampling using WGANGP...")
        
        #wgangp resampling from generator
        samples_noise_vector=tf.random.normal(
            shape=(samples,tf.shape(X)[1])
        )
        pred=self.generator.predict(samples_noise_vector)
        
        #WGANGP wasserstein loss
        if wgan_eval_metric=='wass':
            score=wasserstein_loss(real_data_hyp,pred,num_iter_max=int(1e+8))
            true_var.append(score)
        elif wgan_eval_metric=='sliced_wass':
            score=sliced_wasserstein_loss(real_data_hyp,pred)
            true_var.append(score)
        
        #X resampled
        X_resampled=np.concatenate((X,pred),axis=0)
        
        #y resampled
        y_pred=np.ones(pred.shape[0],dtype=np.int32)
        y_resampled=np.concatenate((y,y_pred),axis=0)
        
        print("Resampling completed...")
        
        #return sampled data
        return X_resampled,y_resampled,*true_var