"""A GAN based over-sampling package.
    Includes implementation of both GAN and WGANGP based over-sampling.
    Also imcludes a combined strategy implementation using
    traditional over-sampling techniques (SMOTE, Borderline-SMOTE, ADASYN)
    and then fine tuning them using GAN and WGANGP.
    
    utils: contains utility functions such as data builder for GAN training,
    GAN loss functions such as wasserstein loss and sliced wasserstein loss and
    generator, discriminator and critic neural network builders.
    
    gan_optuna: Optuna objective functions for GAN and WGANGP hyper-parameter tuning.
    
    dcgan: Implementation of GAN over-sampling technique.
    
    dcgan_resample: A pipeline to implement the dcgan over-sampling technique using
    random noise as generator input.
    
    smote_dcgan: A pipeline to implement the two step strategy using traditional
    over-sampling techniques (SMOTE, Borderline-SMOTE and ADASYN) as dcgan generator
    input.
    
    wgangp: Implementation of WGANGP over-sampling technique.
    
    wgangp_resample: A pipeline to implement the wgangp over-sampling technique using
    random noise as generator input.
    
    smote_wgangp: A pipeline to implement the two step strategy using traditional
    over-sampling techniques (SMOTE, Borderline-SMOTE and ADASYN) as wgangp generator
    input.
    
    simulators: a simulation logic for all the over-sampling technqiues that performs
    hyper-parameter tuning for both DCGAN and WGANGP and then resample. Once resampled,
    an xgboost model is fit using a random sampling technique 3 times and average scores
    (Precision, Recall, F1-score, G-mean and ROC-AUC score) are calculated.
"""
from gan_sampler import dcgan_resample
from gan_sampler import dcgan
from gan_sampler import gan_optuna
from gan_sampler import simulators
from gan_sampler import smote_dcgan
from gan_sampler import smote_wgangp
from gan_sampler import utils
from gan_sampler import wgangp_resample
from gan_sampler import wgangp
