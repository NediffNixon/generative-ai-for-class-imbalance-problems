#imports
import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Union

from imblearn.over_sampling import SMOTE , ADASYN, BorderlineSMOTE
from sklearn.preprocessing import PowerTransformer
from gan_sampler.utils import data_builder
from gan_sampler.utils import wasserstein_loss
from gan_sampler.utils import sliced_wasserstein_loss
from gan_sampler.wgangp import WGANGP

#SMOTE + WGAN-GP
class SMOTE_WGANGP():
    """ (SMOTE, Borderline-SMOTE and ADASYN) + WGANGP sampler class.
    Initalises the WGANGP architecture.

    Args:
    generator (tensorflow.keras.Model): Generator of WGANGP.
    critic (tensorflow.keras.Model): Critic of WGANGP.
    oversampler (str, optional): Type of traditional over-sampling technique to use. Accepts 'smote', 'bsmote' or 'adasyn' that represent SMOTE, Borderline-SMOTE or ADASYN respectively. Defaults to 'smote'.
    critic_steps (int, optional): number of training updates for critic per training of the generator. Defaults to 3.
    gp_weight (float, optional): weight for the gradient penalty. Defaults to 10.0.
        
    Methods:
        compile: compiles the optimizers for the generator and the critic of WGANGP.
        fit_resample: resamples the input to a class balanced dataset.
    """
    #initialise class variables
    def __init__(self,
                 generator:tf.keras.Model,
                 critic:tf.keras.Model,
                 oversampler:str='smote',
                 critic_steps:int=3,
                 gp_weight:float=10.0):
        
        #initializing class variables
        self.generator=generator
        self.critic=critic
        self.critic_steps=critic_steps
        self.gp_weight=gp_weight
        self.oversampler=oversampler
        
        #compile flag
        self.compile_executed=False
    
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
        
        #over-sampling object
        if self.oversampler=='smote':
            sm=SMOTE(random_state=42)
        elif self.oversampler=='bsmote':
            sm=BorderlineSMOTE(random_state=42)
        elif self.oversampler=='adasyn':
            sm=ADASYN(random_state=42)
            
        #resample
        X_res,y_res=sm.fit_resample(X,y)
        
        #scale the input features
        if scale:
            ss=PowerTransformer()
            ss.fit(X_res)
            X_res=ss.transform(X_res)
            true_var.append(ss)
        
        #real data    
        real_data=X_res[np.where(y_res[:len(X)]==1)[0]]
        real_data_hyp=tf.convert_to_tensor(real_data,dtype=tf.float32)
        real_data=data_builder(real_data_hyp,batch_size)
        
        #synthetic data generated
        syn_data=X_res[len(X):]
        syn_data=tf.convert_to_tensor(syn_data,dtype=tf.float32)
        
        #wgangp object
        wgan=WGANGP(self.critic,self.generator,syn_data,critic_steps=self.critic_steps,gp_weight=self.gp_weight,random_noise=False)
        
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
        if hyper_param_tuning:
            #calculate wasserstein or sliced wasserstein loss.
            if wgan_eval_metric is not None:
                fake=tf.random.shuffle(syn_data)[:tf.shape(real_data_hyp)[0]]
                pred=self.generator.predict(fake)
                if wgan_eval_metric=='wass':
                    score=wasserstein_loss(real_data_hyp,pred,num_iter_max=int(1e+8))
                    true_var.append(score)
                    
                elif wgan_eval_metric=='sliced_wass':
                    score=sliced_wasserstein_loss(real_data_hyp,pred)
                    true_var.append(score)
                
            return true_var
        
        print("Sampling using WGANGP...")
        
        #wgangp resampling from generator
        pred=self.generator.predict(syn_data)
        
        #wgangp loss
        if wgan_eval_metric=='wass':
            #wasserstein loss
            score=wasserstein_loss(real_data_hyp,pred,num_iter_max=int(1e+8))
            true_var.append(score)
        elif wgan_eval_metric=='sliced_wass':
            score=sliced_wasserstein_loss(real_data_hyp,pred)
            true_var.append(score)
        
        #X resampled
        X_resampled=np.concatenate((X_res[:len(X)],pred),axis=0)
        
        #y resampled
        y_pred=np.ones(pred.shape[0],dtype=np.int32)
        y_resampled=np.concatenate((y,y_pred),axis=0)
        
        print("Sampling completed...")
        #return sampled data
        return X_resampled,y_resampled,*true_var