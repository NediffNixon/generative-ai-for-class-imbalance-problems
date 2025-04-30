#imports
import tensorflow as tf
import numpy as np
from typing import Union
from ot import dist, emd2, sliced_wasserstein_distance

# Train and Test Split
def train_test_builder(X:Union[tf.Tensor,list],
                       y:Union[None,tf.Tensor,list]=None,
                       train_size:float=0.66,
                       batch_size:int=32,
                       dtype=tf.float32,
                       shuffle:bool=True
                       ):
    """ A train and test data builder which can accept a single tensor or a list of tensors for X and y.
        Works like the sklearn train_test_split function.

    Args:
        X (Union[tensorflow.Tensor, list]): Input features tensor. Can be a single tensor or a list of tensors.
        y (Union[tensorflow.Tensor, list]): The target values (class labels) tensor. Can be a single tensor or a list of tensors.
        train_size (float, optional): fraction of the training data, must be between 0.1 and 0.9. Defaults to 0.66.
        batch_size (int, optional): batch size. Defaults to 32.
        dtype (tensorflow.dtypes.DType, optional): data type of the train and test data. Defaults to tf.float32.
        shuffle (bool, optional): set to determine if the data must be shuffled or not. Defaults to True.
    Returns:
        Tuple: a tuple of training and testing data.
    """
    if not 0.1<=train_size<=0.9:
        raise ValueError(f"train_size must be between 0.1 and 0.9 but given {train_size}")
    
    #concatenate the data if a list of tensors is provided for X or y.        
    if isinstance(X,list):
        X=tf.concat(X,axis=1)
    if isinstance(y,list):
        y=tf.concat(y,axis=1)
    
    #get the indices of the train and test data.
    indices=tf.range(0,X.shape[0],dtype=tf.int32)
    if shuffle:
        indices=tf.random.shuffle(indices)
    i=int(np.ceil(X.shape[0]*train_size))
    train_indices=indices[:i]
    test_indices=indices[i:]
    
    #get the X train and test data.
    X_train=tf.cast(tf.gather(X,indices=train_indices),dtype=dtype)
    X_test=tf.cast(tf.gather(X,indices=test_indices),dtype=dtype)
    
    #get the y train and test data.
    if y !=None:
        y_train=tf.cast(tf.gather(y,indices=train_indices),dtype=dtype)
        y_test=tf.cast(tf.gather(y,indices=test_indices),dtype=dtype)

    #set the batch size and autotune.
    batch_size=batch_size
    AUTOTUNE=tf.data.AUTOTUNE
    
    #build the train data
    if y !=None:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset= train_dataset.prefetch(buffer_size=AUTOTUNE)

    #build the test data
    if y !=None:
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    else:
        val_dataset = tf.data.Dataset.from_tensor_slices(X_test)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    
    #return train and test data as a tuple
    return train_dataset,val_dataset

#tensor data builder
def data_builder(X:Union[np.ndarray,tf.Tensor,list],
                 batch_size:int=32,
                 shuffle:bool=True,
                 dtype:tf.dtypes.DType=tf.float32
                 ):
    """Builds a tensorflow Dataset using a numpy array or a tensorflow tensor.

    Args:
        X (Union[numpy.ndarray, tensorflow.Tensor]): A numpy array or a Tensor to build a Tensorflow dataset.
        batch_size (int, optional): batch size for the dataset. Defaults to 32.
        shuffle (bool, optional): shuffles X before generating a tensorflow dataset if set to True. Defaults to True.
        dtype (tensorflow.dtypes.DType, optional): Data type of the resulting dataset. Defaults to tf.float32.

    Returns:
        A tensorflow Dataset.
    """
    #cast the input to desired data type.
    if isinstance(X,list):
        data=tuple(map(lambda x : tf.cast(x,dtype=dtype),X))
    else:
        data=tf.cast(X,dtype=dtype)
    
    #set the batch size and autotune.
    batch_size=batch_size
    AUTOTUNE=tf.data.AUTOTUNE
    
    #shuffle if required
    if shuffle and not isinstance(X,list):
        data=tf.random.shuffle(data)
    
    #Generate tensorflow dataset
    data=tf.data.Dataset.from_tensor_slices(data)
    data=data.batch(batch_size)
    data=data.prefetch(buffer_size=AUTOTUNE)
    
    #return generated dataset
    return data

#sliced wasserstein loss
def sliced_wasserstein_loss(real:Union[tf.Tensor,np.ndarray],
                            fake:Union[tf.Tensor,np.ndarray],
                            n_projections:int=20):
    """Computes the sliced wasserstein loss between the real and fake data.

    Args:
        real (Union[tensorflow.Tensor, numpy.ndarray]): Samples from real data distributed.
        fake (Union[tensorflow.Tensor, numpy.ndarray]): Samples from generated data distribution.
        n_projections (int, optional): number of projections to project real and fake samples. Defaults to 20.

    Returns:
        sliced Wasserstein loss
    """
    xs=fake
    xt=real
    #convert to numpy array if real and fake samples are tensors.
    if isinstance(xs,tf.Tensor):
        xs=xs.numpy()
    if isinstance(xt,tf.Tensor):
        xt=xt.numpy()
    
    #sliced wasserstein loss
    return sliced_wasserstein_distance(xs,xt,n_projections=n_projections)

#wasserstein loss
def wasserstein_loss(real:Union[tf.Tensor,np.ndarray],
                     fake:Union[tf.Tensor,np.ndarray],
                     num_iter_max:int=100000):
    """Computes the Wasserstein loss between the real and fake data.

    Args:
        real (Union[tensorflow.Tensor, numpy.ndarray]): Samples from real data distributed.
        fake (Union[tensorflow.Tensor, numpy.ndarray]): Samples from generated data distribution.
        num_iter_max (int, optional): The maximum number of iterations before stopping the optimization algorithm if it has not converged. Defaults to 100000.

    Returns:
        Wasserstein loss
    """
    xs=fake
    xt=real
    #convert to numpy array if real and fake samples are tensors.
    if isinstance(xs,tf.Tensor):
        xs=xs.numpy()
    if isinstance(xt,tf.Tensor):
        xt=xt.numpy()
    #pairwise-distance (cost) matrix
    m=dist(xs,xt)
    
    #uniform distribution on samples
    a,b=np.ones((xs.shape[0],))/xs.shape[0] , np.ones((xt.shape[0],))/xt.shape[0]
    
    #wasserstein loss
    return emd2(a,b,m,numItermax=num_iter_max)

#stop on NAN loss callback
class StopOnNaNCallback(tf.keras.callbacks.Callback):
    """ Tensorflow callback to stop training if loss becomes NAN"""
    def on_train_batch_end(self, batch, logs=None):
        loss = logs.get('total_loss')
        if loss is not None and tf.math.is_nan(loss):
            print("Stopping training due to NaN loss.")
            #clear tensorflow graph
            tf.keras.backend.clear_session()
            self.model.stop_training = True

#Generator neural net builder
def build_generator(input_dim:int=30,
                    layers:int=5, 
                    normalize:bool=True, 
                    dropout:bool=True, 
                    verbose:bool=False
                    ):
    """Builds generator tensorflow functional model.
    The generator first expands the dimensions and then reduces it back to input_dim.
    Args:
        input_dim (int, optional): Input dimensions of the generator(# input features). Defaults to 30.
        layers (int, optional): Number of dense layers. Defaults to 5.
        normalize (bool, optional): Uses BatchNormalization between layers if set to True. Defaults to True.
        dropout (bool, optional): Uses AlphaDropout with a dropout rate of 0.2 if set to True. Defaults to True.
        verbose (bool, optional): Prints number of dense layers, trainable, non-trainable and total parameters if set to True. Defaults to False.

    Returns:
        Tensorflow functional model.
    """
    #Input layer
    inp=tf.keras.Input(shape=(input_dim,))
    x=tf.keras.layers.Dense(input_dim,activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.he_normal(),use_bias=not normalize)(inp)
    
    # check for batch normalization and alpha dropout.
    if normalize:
            x=tf.keras.layers.BatchNormalization()(x)
    if dropout:
            x=tf.keras.layers.AlphaDropout(rate=0.2)(x)
    
    #calculate the midpoint    
    midpoint=int(np.ceil(layers/2))
    
    #adding dense layers
    for i in range(1,midpoint+1):
        x=tf.keras.layers.Dense(input_dim*(2**i),activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.he_normal(),use_bias=not normalize)(x)
        if normalize:
            x=tf.keras.layers.BatchNormalization()(x)
        if dropout:
            x=tf.keras.layers.AlphaDropout(rate=0.2)(x)
        
    for i in range(1,layers-midpoint+1):
        x=tf.keras.layers.Dense((input_dim*(2**midpoint))//(2**i),activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.he_normal(),use_bias=not normalize)(x)
        if normalize:
            x=tf.keras.layers.BatchNormalization()(x)
        if dropout:
            x=tf.keras.layers.AlphaDropout(rate=0.2)(x)
    
    #output layer
    out=tf.keras.layers.Dense(input_dim,kernel_initializer=tf.keras.initializers.he_normal(),use_bias=not normalize)(x)
    
    #generator model
    model=tf.keras.Model(inputs=inp,outputs=out)
    
    #verbosity    
    if verbose:
        print("=======Generator========")
        trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        totalParams = trainableParams + nonTrainableParams
        print("# Dense layers:",layers)
        print("Total params:",totalParams)
        print("Trainable params:",trainableParams)
        print("Non-trainable params:",nonTrainableParams)
        print("=========================")
    
    #returns the model.
    return model

def build_discriminator(input_dim:int=30,
                        normalize:bool=True,
                        dropout:bool=True,
                        verbose:bool=False):
    """Builds discriminator tensorflow functional model.

    Args:
        input_dim (int, optional): Input dimensions of the discriminator(# input features). Defaults to 30.
        normalize (bool, optional): Uses BatchNormalization between layers if set to True. Defaults to True.
        dropout (bool, optional): Uses AlphaDropout with a dropout rate of 0.2 if set to True. Defaults to True.
        verbose (bool, optional): Prints number of dense layers, trainable, non-trainable and total parameters if set to True. Defaults to False.

    Returns:
        Tensorflow functional model.
    """
    #Input layer
    inp=tf.keras.Input(shape=(input_dim,))
    x=tf.keras.layers.Dense(input_dim,activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.he_normal(),use_bias= not normalize)(inp)
    
    # check for batch normalization and alpha dropout.
    if normalize:
            x=tf.keras.layers.BatchNormalization()(x)
    if dropout:
            x=tf.keras.layers.AlphaDropout(rate=0.2)(x)
    
    #calculate number of dense layers
    layers=int(np.log2(int(input_dim/2)))
    
    #adding dense layers
    for i in range(1,layers+1):
        x=tf.keras.layers.Dense(input_dim//(2**i),activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.he_normal(),use_bias=not normalize)(x)
        if normalize:
            x=tf.keras.layers.BatchNormalization()(x)
        if dropout:
            x=tf.keras.layers.AlphaDropout(rate=0.2)(x)
    
    #output layer
    out=tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer=tf.keras.initializers.he_normal(),use_bias= not normalize)(x)
    
    #discriminator model
    model=tf.keras.Model(inputs=inp,outputs=out)
    
    #verbosity
    if verbose:
        print("=======Discriminator========")
        trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        totalParams = trainableParams + nonTrainableParams
        print("# Dense layers:",layers-1)
        print("Total params:",int(totalParams))
        print("Trainable params:",int(trainableParams))
        print("Non-trainable params:",int(nonTrainableParams))
        print("============================")
    
    #return the model
    return model

def build_critic(input_dim:int=30,
                 dropout:bool=True,
                 verbose:bool=False):
    """Builds critic tensorflow functional model.

    Args:
        input_dim (int, optional): Input dimensions of the critic(# input features). Defaults to 30.
        dropout (bool, optional): Uses AlphaDropout with a dropout rate of 0.2 if set to True. Defaults to True.
        verbose (bool, optional): Prints number of dense layers, trainable, non-trainable and total parameters if set to True. Defaults to False.

    Returns:
        Tensorflow functional model.
    """
    #Input layer
    inp=tf.keras.Input(shape=(input_dim,))
    x=tf.keras.layers.Dense(input_dim,activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.he_normal())(inp)
    
    #check for alpha dropouts.    
    if dropout:
            x=tf.keras.layers.AlphaDropout(rate=0.2)(x)
    
    #calculate number of dense layers.
    layers=int(np.log2(int(input_dim/2)))
    
    #adding dense layers
    for i in range(1,layers+1):
        tf.keras.layers.Dense(input_dim//(2**i),activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.he_normal())(x)
        if dropout:
            x=tf.keras.layers.AlphaDropout(rate=0.2)(x)
    
    #output layer
    out=tf.keras.layers.Dense(1,kernel_initializer=tf.keras.initializers.he_normal())(x)
    
    #critic model
    model=tf.keras.Model(inputs=inp,outputs=out)

    #verbosity
    if verbose:
        print("=======Critic=========")
        trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        totalParams = trainableParams + nonTrainableParams
        print("# Dense layers:",layers-1)
        print("Total params:",int(totalParams))
        print("Trainable params:",int(trainableParams))
        print("Non-trainable params:",int(nonTrainableParams))
        print("======================")
    
    #return the model
    return model