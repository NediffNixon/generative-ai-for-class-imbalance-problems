#imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
from typing import Union, Optional

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.metrics import geometric_mean_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from xgboost import XGBClassifier

import tensorflow as tf
import optuna as opt

from gan_sampler.gan_optuna import wgangp_objective
from gan_sampler.gan_optuna import dcgan_objective

#Simulation random-states = [0,22,42]

#Simulation without oversampling.
def no_oversampling_sim(X:Union[pd.DataFrame,np.ndarray],
                        y:Union[pd.DataFrame,np.ndarray]
                        ) -> pd.DataFrame:
    """The performance of XGBoost Classifier without any over-sampling.

    Args:
        X (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples, n_features): Input feature samples.
        y (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples,): The target values (class labels).
        X and y must be of same type.
    Returns:
        Average performance of xgboost as a pandas dataframe.
    """
    #random-states for simulation
    random_states=[0,22,42]
    
    #scoring dataframe
    rows=['Precision','Recall','F1-score','G-mean','ROC-AUC score']
    columns=["No-oversampling"]
    scores_df=pd.DataFrame(index=rows,columns=columns)
    
    #sampling simulation
    scores=defaultdict(list)
    
    #random-state simulation for each sampler.
    for i in tqdm(random_states,desc="XGBoost Scoring",colour='blue'):
        #train-test-split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i)
        
        #train-valid-split
        X_sub_train,X_valid,y_sub_train,y_valid=train_test_split(X_train,y_train,test_size=0.2,random_state=i)
        
        #gradient boosting classifier
        gbc=XGBClassifier(objective='binary:logistic',random_state=42,early_stopping_rounds=50,n_jobs=2)
        
        #fit the model and predict the labels of validation set
        gbc.fit(X_sub_train,y_sub_train,eval_set=[(X_valid,y_valid)], verbose=False)
        
        #predictions on test set
        gbc_pred=gbc.predict(X_test)
        
        #calculate scores
        scores["Precision"].append(np.round(precision_score(y_test,gbc_pred),4))
        scores["Recall"].append(np.round(recall_score(y_test,gbc_pred),4))
        scores["F1-score"].append(np.round(f1_score(y_test,gbc_pred),4))
        scores["G-mean"].append(np.round(geometric_mean_score(y_test,gbc_pred,average='binary'),4))
        scores["ROC-AUC score"].append(np.round(roc_auc_score(y_test,gbc_pred),4))
    
    #Average scores
    scores_df['No-oversampling']=np.mean(list(scores.values()),axis=1)
    
    #return sim results
    return scores_df

#Simulation for traditional over-sampling techniques.
def trad_oversampling_sim(X:Union[pd.DataFrame,np.ndarray],
                          y:Union[pd.DataFrame,np.ndarray]
                          ) -> pd.DataFrame:
    """SMOTE, Borderline-SMOTE and ADASYN simulator using 5 predefined random states.
    The performance of these techniques is measured using a XGBoost classifier.

    Args:
        X (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples, n_features): Input feature samples.
        y (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples,): The target values (class labels).

    Returns:
        Average score of each sampling technique as a pandas dataframe.
    """
    #random-states for simulation
    random_states=[0,22,42]
    
    #traditional over-samplers
    samplers=[SMOTE(random_state=42),BorderlineSMOTE(random_state=42),ADASYN(random_state=42)]
    
    #scoring dataframe
    rows=['Precision','Recall','F1-score','G-mean','ROC-AUC score']
    columns=["SMOTE",'Borderline-SMOTE','ADASYN']
    scores_df=pd.DataFrame(index=rows,columns=columns)
    
    #sampling simulation
    for sampler in tqdm(samplers,desc="Samplers",colour='blue'):
        scores=defaultdict(list)
        
        #random-state simulation for each sampler.
        for i in tqdm(random_states,desc="XGBoost Scoring",colour='blue'):
            #train-test-split
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=i)
            
            #train-valid-split
            X_sub_train,X_valid,y_sub_train,y_valid=train_test_split(X_train,y_train,test_size=0.2,random_state=i)
            
            #sampler resampling
            X_train_res,y_train_res=sampler.fit_resample(X_sub_train,y_sub_train)
            
            #gradient boosting classifier
            gbc=XGBClassifier(objective='binary:logistic',random_state=42,early_stopping_rounds=50,n_jobs=2)
            
            #fit the model and predict the labels of validation set
            gbc.fit(X_train_res,y_train_res,eval_set=[(X_valid,y_valid)], verbose=False)
            
            #predictions on test set
            gbc_pred=gbc.predict(X_test)
            
            #calculate scores
            scores["Precision"].append(np.round(precision_score(y_test,gbc_pred),4))
            scores["Recall"].append(np.round(recall_score(y_test,gbc_pred),4))
            scores["F1-score"].append(np.round(f1_score(y_test,gbc_pred),4))
            scores["G-mean"].append(np.round(geometric_mean_score(y_test,gbc_pred,average='binary'),4))
            scores["ROC-AUC score"].append(np.round(roc_auc_score(y_test,gbc_pred),4))
        
        # Average the scores of each sampler.
        if isinstance(sampler,SMOTE):
            scores_df['SMOTE']=np.mean(list(scores.values()),axis=1)
        elif isinstance(sampler,BorderlineSMOTE):
            scores_df['Borderline-SMOTE']=np.mean(list(scores.values()),axis=1)
        else:
            scores_df['ADASYN']=np.mean(list(scores.values()),axis=1)
    
    #return sim results
    return scores_df

#DCGAN training simulation
def dcgan_training_sim(X:Union[pd.DataFrame,np.ndarray],
                       y:Union[pd.DataFrame,np.ndarray],
                       traditional_oversampling:Optional[str]=None,
                       batch_size:int=32,
                       hyp_tuning_pruning:bool=True,
                       hyp_tuning_trials:int=20
                       ) -> list:
    """DCGAN training simulator that is either trained using traditional over-sampling techniques for generator input
    or samples from random normal distribution.
    The simulator runs for 3 predefined random states and returns a list of DCGAN generators for each simulation run.

    Args:
        X (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples, n_features): Input feature samples.
        y (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples,): The target values (class labels).
        traditional_oversampling (Union[None, str], optional): Set to None to sample from random normal distribution
        or choose between 'smote', 'bsmote' and 'adasyn' to sample from SMOTE, Borderline-SMOTE or ADASYN oversampling techniques. Defaults to None.
        batch_size (int, optional): Batch size for DCGAN training. Defaults to 32.
        hyp_tuning_pruning (bool, optional): If set to True then optuna pruning is enabled. Defaults to True.
        hyp_tuning_trials (int, optional): Number of trial for each optuna optimization. Defaults to 20.
        
    Returns:
        List of DCGAN generators from each simulation.
    """
    #random-states for simulation
    random_states=[0,22,42]
    
    #DCGAN generator list.
    generators=[]
    
    #random-state simulation for DCGAN
    for i in tqdm(random_states,desc="DCGAN training sim",colour='blue'):
        
        #train-test-split
        X_train,_,y_train,_=train_test_split(X,y,test_size=0.20,random_state=i)
        
        #train-valid-split
        X_sub_train,_,y_sub_train,_=train_test_split(X_train,y_train,test_size=0.2,random_state=i)
        
        #optuna optimization   
        study=opt.create_study(direction="minimize",pruner=opt.pruners.HyperbandPruner(),study_name="WGANGP-training")
        study.optimize(lambda trial: dcgan_objective(trial,X_sub_train,y_sub_train,trad_oversampling=traditional_oversampling,enable_pruning=hyp_tuning_pruning,batch_size=batch_size),n_trials=hyp_tuning_trials)
        
        #store the best DCGAN model's generator
        generators.append(study.best_trial.user_attrs['dcgan_generator'])
    
    #return simulated DCGAN generators    
    return generators

#DCGAN sampling simulation
def dcgan_sampling_sim(X:Union[pd.DataFrame,np.ndarray],
                       y:Union[pd.DataFrame,np.ndarray],
                       traditional_oversampling:Optional[str]=None,
                       batch_size:int=32
                       ) -> pd.DataFrame:
    """DCGAN simulation using either random normal distribution samping or from SMOTE, Borderline-SMOTE or ADASYN.
    Simulation runs for 3 redefined random states.

    Args:
        X (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples, n_features): Input feature samples.
        y (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples,): The target values (class labels).
        traditional_oversampling (Union[None, str], optional): Set to None to sample from random normal distribution
        or choose between 'smote', 'bsmote' and 'adasyn' to sample from SMOTE, Borderline-SMOTE or ADASYN oversampling techniques. Defaults to None.
        batch_size (int, optional): Batch size for DCGAN training. Defaults to 32.
        
    Returns:
        Average scores for the simulation.
    """
    #random-states for simulation
    random_states=[0,22,42]
    
    #scores dictionary
    scores=defaultdict(list)
    
    #get the DCGAN generators for simulation
    generators=dcgan_training_sim(X,y,traditional_oversampling=traditional_oversampling,batch_size=batch_size)
    
    #scoring dataframe
    rows=['Precision','Recall','F1-score','G-mean','ROC-AUC score']
    columns=['DCGAN' if traditional_oversampling is None else f"{traditional_oversampling.upper()}_DCGAN"]
    scores_df=pd.DataFrame(index=rows,columns=columns)
    
    #DCGAN resampling simulation
    for gen,i in zip(generators,tqdm(random_states,desc="XGBoost Scoring",colour='blue')):
        
        #train-test-split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=i)
        
        #train-valid-split
        X_sub_train,X_valid,y_sub_train,y_valid=train_test_split(X_train,y_train,test_size=0.20,random_state=i)
        
        #DCGAN generator
        dcgan_gen=gen
        
        #check if traditional oversampling is used and make predictions accordingly
        if traditional_oversampling=="smote":
            sm=SMOTE(random_state=42)
            X_train_res,y_train_res=sm.fit_resample(X_sub_train,y_sub_train)
            pred=dcgan_gen.predict(X_train_res[len(X_sub_train):])
            y_pred=y_train_res[len(X_sub_train):]
        
        elif traditional_oversampling=="bsmote":
            sm=BorderlineSMOTE(random_state=42)
            X_train_res,y_train_res=sm.fit_resample(X_sub_train,y_sub_train)
            pred=dcgan_gen.predict(X_train_res[len(X_sub_train):])
            y_pred=y_train_res[len(X_sub_train):]
        
        elif traditional_oversampling=="adasyn":
            sm=ADASYN(random_state=42)
            X_train_res,y_train_res=sm.fit_resample(X_sub_train,y_sub_train)
            pred=dcgan_gen.predict(X_train_res[len(X_sub_train):])
            y_pred=y_train_res[len(X_sub_train):]
            
        if traditional_oversampling is None:
            num_samples=dict(Counter(y_sub_train))
            num_samples=num_samples[0]-num_samples[1]
            samples_noise_vector=tf.random.normal(
                shape=(num_samples,tf.shape(X_sub_train)[1])
            )
            pred=dcgan_gen.predict(samples_noise_vector)
            y_pred=np.ones(pred.shape[0],dtype=np.int32)
        
        #resampled train set
        X_train_res=np.concatenate((X_sub_train,pred),axis=0)
        y_train_res=np.concatenate((y_sub_train,y_pred),axis=0)
        
        X_valid=X_valid.to_numpy()
        X_test=X_test.to_numpy()
        
        #gradient boosting classifier
        gbc=XGBClassifier(objective='binary:logistic',random_state=42,early_stopping_rounds=50,n_jobs=2)
        
        #fit the model and predict the labels of validation set
        gbc.fit(X_train_res,y_train_res,eval_set=[(X_valid,y_valid)],verbose=False)
        
        #make predictions on test set
        gbc_pred=gbc.predict(X_test)
        
        #calculate scores
        scores["Precision"].append(np.round(precision_score(y_test,gbc_pred),4))
        scores["Recall"].append(np.round(recall_score(y_test,gbc_pred),4))
        scores["F1-score"].append(np.round(f1_score(y_test,gbc_pred),4))
        scores["G-mean"].append(np.round(geometric_mean_score(y_test,gbc_pred,average='binary'),4))
        scores["ROC-AUC score"].append(np.round(roc_auc_score(y_test,gbc_pred),4))
        
        #Average the scores
        scores_df['DCGAN' if traditional_oversampling is None else f"{traditional_oversampling.upper()}_DCGAN"]=np.mean(list(scores.values()),axis=1)
    
    #return simulation results
    return scores_df


def wgangp_training_sim(X:Union[pd.DataFrame,np.ndarray],
                        y:Union[pd.DataFrame,np.ndarray],
                        traditional_oversampling:Optional[str]=None,
                        batch_size:int=32,
                        hyp_tuning_pruning:bool=True,
                        hyp_tuning_trials:int=20
                        ) -> list:
    """WGANGP training simulator that is either trained using traditional over-sampling techniques for generator input
    or samples from random normal distribution.
    The simulator runs for 3 predefined random states and returns a list of WGANGP generators for each simulation run.

    Args:
        X (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples, n_features): Input feature samples.
        y (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples,): The target values (class labels).
        traditional_oversampling (Union[None, str], optional): Set to None to sample from random normal distribution
        or choose between 'smote', 'bsmote' and 'adasyn' to sample from SMOTE, Borderline-SMOTE or ADASYN oversampling techniques. Defaults to None.
        batch_size (int, optional): Batch size for WGANGP training. Defaults to 32.
        hyp_tuning_pruning (bool, optional): If set to True then optuna pruning is enabled. Defaults to True.
        hyp_tuning_trials (int, optional): Number of trial for each optuna optimization. Defaults to 20.
        
    Returns:
        List of WGANGP generators from each simulation.
    """
    #random-states for simulation
    random_states=[0,22,42]
    
    #WGANGP generator list.
    generators=[]
    
    #random-state simulation for DCGAN
    for i in tqdm(random_states,desc="WGANGP optimization sim",colour='blue'):  
          
        #train-test-split
        X_train,_,y_train,_=train_test_split(X,y,test_size=0.20,random_state=i)
        
        #train-valid-split
        X_sub_train,_,y_sub_train,_=train_test_split(X_train,y_train,test_size=0.2,random_state=i)
            
        #optuna optimization   
        study=opt.create_study(direction="minimize",pruner=opt.pruners.HyperbandPruner(),study_name="WGANGP-training")
        study.optimize(lambda trial: wgangp_objective(trial,X_sub_train,y_sub_train,trad_oversampling=traditional_oversampling,enable_pruning=hyp_tuning_pruning,batch_size=batch_size),n_trials=hyp_tuning_trials)
        
        #store the best WGANGP model's generator
        generators.append(study.best_trial.user_attrs['wgan_generator'])
    
    #return the simulation generators   
    return generators

#WGANGP sampling simulation
def wgangp_sampling_sim(X:Union[pd.DataFrame,np.ndarray],
                        y:Union[pd.DataFrame,np.ndarray],
                        traditional_oversampling:Optional[str]=None,
                        batch_size:int=32,
                        hyp_tuning_pruning:bool=True,
                        hyp_tuning_trials:int=20
                        ) -> pd.DataFrame:
    """DCGAN simulation using either random normal distribution samping or from SMOTE, Borderline-SMOTE or ADASYN.
    Simulation runs for 3 redefined random states.

    Args:
        X (Union[pandas.DataFrame, numpy.ndarray]) of shape (n_samples, n_features): Input feature samples.
        y (Union[pandas.DataFrame or numpy.ndarray]) of shape (n_samples,): The target values (class labels).
        traditional_oversampling (Union[None, str], optional): Set to None to sample from random normal distribution
        or choose between 'smote', 'bsmote' and 'adasyn' to sample from SMOTE, Borderline-SMOTE or ADASYN oversampling techniques. Defaults to None.
        batch_size (int, optional): Batch size for WGANGP training. Defaults to 32.
        hyp_tuning_pruning (bool, optional): If set to True then optuna pruning is enabled. Defaults to True.
        hyp_tuning_trials (int, optional): Number of trial for each optuna optimization. Defaults to 20.
        
    Returns:
        Average scores for the simulation.
    """
    #random-states for simulation
    random_states=[0,22,42]
    
    #scores dictionary
    scores=defaultdict(list)
    
    #get the WGANGP generators for simulation
    generators=wgangp_training_sim(X,y,traditional_oversampling=traditional_oversampling,batch_size=batch_size,hyp_tuning_pruning=hyp_tuning_pruning, hyp_tuning_trials=hyp_tuning_trials)
    
    #scoring dataframe
    rows=['Precision','Recall','F1-score','G-mean','ROC-AUC score']
    columns=['WGANGP' if traditional_oversampling is None else f"{traditional_oversampling.upper()}_WGANGP"]
    scores_df=pd.DataFrame(index=rows,columns=columns)
    
    #WGANGP resampling simulation
    for gen,i in zip(generators,tqdm(random_states,desc="XGBoost Scoring",colour='blue')):
        
        #train-test-split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=i)
        
        #train-valid-split
        X_sub_train,X_valid,y_sub_train,y_valid=train_test_split(X_train,y_train,test_size=0.20,random_state=i)
        
        #WGANGP generator
        wgan_gen=gen
        
        #check if traditional oversampling is used and make predictions accordingly
        if traditional_oversampling=="smote":
            sm=SMOTE(random_state=42)
            X_train_res,y_train_res=sm.fit_resample(X_sub_train,y_sub_train)
            pred=wgan_gen.predict(X_train_res[len(X_sub_train):])
            y_pred=y_train_res[len(X_sub_train):]
        
        elif traditional_oversampling=="bsmote":
            sm=BorderlineSMOTE(random_state=42)
            X_train_res,y_train_res=sm.fit_resample(X_sub_train,y_sub_train)
            pred=wgan_gen.predict(X_train_res[len(X_sub_train):])
            y_pred=y_train_res[len(X_sub_train):]
        
        elif traditional_oversampling=="adasyn":
            sm=ADASYN(random_state=42)
            X_train_res,y_train_res=sm.fit_resample(X_sub_train,y_sub_train)
            pred=wgan_gen.predict(X_train_res[len(X_sub_train):])
            y_pred=y_train_res[len(X_sub_train):]
            
        if traditional_oversampling is None:
            num_samples=dict(Counter(y_sub_train))
            num_samples=num_samples[0]-num_samples[1]
            samples_noise_vector=tf.random.normal(
                shape=(num_samples,tf.shape(X_sub_train)[1])
            )
            pred=wgan_gen.predict(samples_noise_vector)
            y_pred=np.ones(pred.shape[0],dtype=np.int32)
        
        #resampled train set
        X_train_res=np.concatenate((X_sub_train,pred),axis=0)
        y_train_res=np.concatenate((y_sub_train,y_pred),axis=0)
        
        X_valid=X_valid.to_numpy()
        X_test=X_test.to_numpy()
        
        #gradient boosting classifier
        gbc=XGBClassifier(objective='binary:logistic',random_state=42,early_stopping_rounds=50,n_jobs=2)
        
        #fit the model and predict the labels of validation set
        gbc.fit(X_train_res,y_train_res,eval_set=[(X_valid,y_valid)],verbose=False)
        
        #make predictions on test set
        gbc_pred=gbc.predict(X_test)
        
        #calculate scores
        scores["Precision"].append(np.round(precision_score(y_test,gbc_pred),4))
        scores["Recall"].append(np.round(recall_score(y_test,gbc_pred),4))
        scores["F1-score"].append(np.round(f1_score(y_test,gbc_pred),4))
        scores["G-mean"].append(np.round(geometric_mean_score(y_test,gbc_pred,average='binary'),4))
        scores["ROC-AUC score"].append(np.round(roc_auc_score(y_test,gbc_pred),4))
        
        #Average the scores
        scores_df['WGANGP' if traditional_oversampling is None else f"{traditional_oversampling.upper()}_WGANGP"]=np.mean(list(scores.values()),axis=1)
    
    #return simulation results
    return scores_df                