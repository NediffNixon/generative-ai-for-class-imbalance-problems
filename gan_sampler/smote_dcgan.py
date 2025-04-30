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
from gan_sampler.dcgan import DCGAN

#SMOTE + WGAN-GP
class SMOTE_DCGAN():
    """ (SMOTE, Borderline-SMOTE and ADASYN) + DCGAN sampler class.
    Initalises the WGANGP architecture.

    Args:
    generator (tensorflow.keras.Model): Generator of DCGAN.
    discriminator (tensorflow.keras.Model): Discriminator of DCGAN.
    oversampler (str, optional): Type of traditional over-sampling technique to use. Accepts 'smote', 'bsmote' or 'adasyn' that represent SMOTE, Borderline-SMOTE or ADASYN respectively. Defaults to 'smote'.
        
    Methods:
        compile: compiles the optimizers for the generator and the critic of DCGAN.
        fit_resample: resamples the input to a class balanced dataset.
    """
    #initialise class variables
    def __init__(self,
                 generator:tf.keras.Model,
                 discriminator:tf.keras.Model,
                 oversampler:str='smote'):
        
        self.generator=generator
        self.discriminator=discriminator
        self.oversampler=oversampler
        
        #compile flag
        self.compile_executed=False
    
    #optimizers compiler
    def compile(self,
                g_optimizer:tf.keras.optimizers.Optimizer,
                d_optimizer:tf.keras.optimizers.Optimizer):
        
        """Compiles optimizers for DCGAN generator and dicriminator.

        Args:
            g_optimizer (tensorflow.keras.optimizers.Optimizer): Generator optimizer.
            d_optimizer (tensorflow.keras.optimizers.Optimizer): Discriminator optimizer.
        """
        self.g_optimizer=g_optimizer
        self.d_optimizer=d_optimizer
        
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
                     gan_eval_metric:Union[None,str]=None):
        
        """Resampler for DCGAN.

        Args:
            X (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples, n_features): Input feature samples.
            y (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples,): The target values (class labels).
            epochs (int, optional): Number of training epochs for DCGAN. Defaults to 100.
            batch_size (int, optional): Batch-size for DCGAN. Defaults to 32.
            callbacks (Union[None, list], optional): DCGAN callbacks. Supports Tensorflow callbacks. Defaults to None.
            history (bool, optional): Returns the DCGAN training history if set to True. Defaults to True.
            verbose (Union[int, str], optional): Tensorflow verbosity level for DCGAN training. Defaults to 'auto'.
            scale (bool, optional): Performs PowerTransform (Sklearn) on X if set to True and returns the fitted scaler object. Defaults to True.
            hyper_param_tuning (bool, optional): Runs in hyper-parameter mode and does not perform resampling if set to True. Defaults to False.
            gan_eval_metric (Union[None, str], optional): Returns wasserstein loss if set to 'wass' or returns sliced wasserstein loss if set to 'sliced_wass'. Defaults to None.

        Raises:
            RuntimeError: If resampled before compiling DCGAN optimizers.

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
        
        #scale input features
        if scale:
            ss=PowerTransformer()
            ss.fit(X_res)
            X_res=ss.transform(X_res)
            true_var.append(ss)
        
        #real data    
        real_data=X_res[np.where(y_res[:len(X)]==1)[0]]
        real_data_hyp=tf.convert_to_tensor(real_data,dtype=tf.float32)
        real_data=data_builder(real_data_hyp,batch_size)
        
        #synthetic data
        syn_data=X_res[len(X):]
        syn_data=tf.convert_to_tensor(syn_data,dtype=tf.float32)
        
        #wgangp object
        gan=DCGAN(self.discriminator,self.generator,syn_data,random_noise=False)
        
        #wgangp compiler
        gan.compile(
            d_optimizer=self.d_optimizer,
            g_optimizer=self.g_optimizer
        )
        
        print("Training DCGAN model...")
        
        #fit the wgangp model
        gan.fit(
            real_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        print("DCGAN model finished training...")
        
        # wgangp training history
        if history:
            his = pd.DataFrame(gan.history.history)
            true_var.append(his)
        
        #If on hyper-parameter tuning mode, no need to resample.
        if hyper_param_tuning:
            #calculate wasserstein or sliced wasserstein loss.
            if gan_eval_metric is not None:
                fake=tf.random.shuffle(syn_data)[:tf.shape(real_data_hyp)[0]]
                pred=self.generator.predict(fake)
                if gan_eval_metric=='wass':
                    score=wasserstein_loss(real_data_hyp,pred,num_iter_max=int(1e+8))
                    true_var.append(score)
                    
                elif gan_eval_metric=='sliced_wass':
                    score=sliced_wasserstein_loss(real_data_hyp,pred)
                    true_var.append(score)
                
            return true_var
        
        print("Resampling using DCGAN...")
        
        #dcgan resampling from generator
        pred=self.generator.predict(syn_data)
        
        #dcgan loss
        if gan_eval_metric=='wass':
            #wasserstein loss
            score=wasserstein_loss(real_data_hyp,pred,num_iter_max=int(1e+8))
            true_var.append(score)
        elif gan_eval_metric=='sliced_wass':
            score=sliced_wasserstein_loss(real_data_hyp,pred)
            true_var.append(score)
        
        
        #X resampled
        X_resampled=np.concatenate((X_res[:len(X)],pred),axis=0)
        
        #y resampled
        y_pred=np.ones(pred.shape[0],dtype=np.int32)
        y_resampled=np.concatenate((y,y_pred),axis=0)
        
        print("Resampling completed...")
        
        #return sampled data
        return X_resampled,y_resampled,*true_var