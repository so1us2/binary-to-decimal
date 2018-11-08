import numpy as np
import tensorflow as tf
import keras as K
from keras.layers import Input, Dense, Activation, Bidirectional, LSTM, Reshape, BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt



'''
###########################
Data generation functions
###########################
'''

def dec_to_bin(dec):
    '''
    Converts a decimal integer to its binary representation

    arguments:
    dec -- int
    returns:
    bits -- binary representation as a numpy array of 0's and 1's.

    '''
    bits = []
    while dec > 0:
        bit = int(dec % 2)
        dec = np.floor_divide(dec, 2)
        bits.append(bit)

    return np.asarray(bits[::-1])

def bin_to_dec(bin):
    '''
    Converts a binary representation to a decimal integer.

    arguments:
    bin -- binary representation as a numpy array of 0's and 1's
    returns:
    dec -- integer represented by the input.
    '''
    L = np.size(bin)
    pow2 = 1
    dec = 0
    for i in reversed(range(L)):
        dec += pow2 * bin[i]
        pow2 *= 2
    return int(dec)

def zero_pad_left(arr, size):
    len = np.size(arr)
    if size > len:
        zero_pad = np.zeros((size-len,), dtype=int)
        arr = np.concatenate((zero_pad, arr))
    return arr


def decimal_digit_to_4bits(d):
    '''
    Converts 0 - 9 to binary representation

    arguments:
    d -- int in the range 0 <= d <= 9
    returns:
    bits -- padded four-bit representation of d
    '''
    return zero_pad_left(dec_to_bin(d), 4)


def dec_to_decbits(dec, num_digits=3, pad=True):
    '''
    Converts a decimal int to a bitwise representation.

    arguments:
    dec -- decimal int
    num_digits -- how many decimal digits to which to pad to the left
    pad -- boolean flag, whether to pad left

    returns:
    decbits -- bit representation of decimal.  For example, 93 will be represented as 0 9 3 = 0000 1001 0011
    '''

    digits = np.asarray([int(s) for s in str(dec)], dtype=int)
    length = np.size(digits)
    
    if pad:
        digits = zero_pad_left(digits, num_digits)
        
    decbits = list(map(decimal_digit_to_4bits, digits))
    return np.reshape(np.array(decbits, dtype=int), (num_digits * 4,))

def decbits_to_dec(decbits):
    '''
    Converts a bit representation of a decimal, to the int.
    
    arguments:
    decbits -- bit representation of decimal.  For example, 93 will be represented as 0 9 3 = 0000 1001 0011.
    convenience, we allow decbits to be floats, in which case we first apply rounding.  
    
    returns:
    The integer which was represented by decbits.  
    '''
    digits = np.array([], dtype=int)
    while i < np.size(decbits):
        digits[int(i/4)] = bin_to_dec(decbits[i:i+4])
        

def generate_datapoint( num_bits=10, num_digits=3):
    '''
    Picks a number n in the range 0 - 999 uniformly at random.  Outputs a pair
    [n_bin, n_dec] for the learning problem of converting binary to decimal.

    arguments:
    num_bits -- number of bits in n_bin
    num_digits -- number of decimal digits in n_dec
    
    returns:
    n_bin -- binary representation
    n_dec -- decimal representation
    '''
    n = np.random.randint(0,pow(10,num_digits))
    return zero_pad_left(dec_to_bin(n),num_bits), dec_to_decbits(n, num_digits=num_digits)

def generate_datapoints(num_examples=100, num_bits=10, num_digits=3):
    '''
    Generates a dataset.
    
    arguments:
    num_examples -- number of training examples in dataset.
    num_bits -- number of bits in inputs
    num_digits -- number of decimal digits in outputs.
    
    returns:
    X -- input dataset of size (num_examples, num_bits)
    Y -- output dataset of size (num_examples, num_digits * 4)
    '''
    X = []
    Y = []
    for i in range(num_examples):
        x_pt, y_pt = generate_datapoint(num_bits, num_digits)
        X.append(x_pt)
        Y.append(y_pt)
    return np.array(X), np.array(Y)

def decbits_to_dec(decbits):
    '''
    Converts a bit representation of a decimal, to the int.
    
    arguments:
    decbits -- bit representation of decimal.  For example, 93 will be represented as 0 9 3 = 0000 1001 0011.
    convenience, we allow decbits to be floats, in which case we first apply rounding.  
    
    returns:
    The integer which was represented by decbits.  
    '''
    decbits = np.round(decbits)
    digits = []
    i = 0
    while i < np.size(decbits):
        digits.append(str(bin_to_dec(decbits[i:i+4])))
        i += 4
    return int(''.join(digits))


'''
########################
####  Models  ##########
########################
'''


def model_1layer(num_bits=10, num_digits=3, a_h=20):
    '''
    Create a Keras bidirectional LSTM model for computing the
    decimal representation of a number in binary.  
    
    arguments:
    num_bits -- number of bits in the input
    num_digits -- number of digits in the output
    a_h -- number of hidden units in bidirectional LSTM
    
    returns:
    model -- Keras model instance
    '''
    
    
    model_input = Input(shape=(num_bits,1))
    lstm1_state = LSTM(a_h, return_sequences=False)(model_input)
    lstm1_state = Reshape((a_h,))(lstm1_state)
    model_output = Dense(4 * num_digits, activation='sigmoid')(lstm1_state)
    
    return Model(inputs=model_input, outputs=model_output)

def model_2layer(num_bits=10, num_digits=3, a_h1=20, a_h2=20):
    '''
    Create a Keras bidirectional LSTM model for computing the
    decimal representation of a number in binary.  
    
    arguments:
    num_bits -- number of bits in the input
    num_digits -- number of digits in the output
    a_h1 -- number of hidden units in first bidirectional LSTM
    a_h2 -- number of hidden units in second uni-directional LSTM
    
    returns:
    model -- Keras model instance
    '''
    
    
    model_input = Input(shape=(num_bits,1))
    lstm1_state = Bidirectional(LSTM(a_h1, return_sequences=True))(model_input)
    lstm2_state = LSTM(a_h2, return_sequences=False)(lstm1_state)
    dense_in = Reshape((a_h2,))(lstm2_state)
    model_output = Dense(4 * num_digits, activation='sigmoid')(dense_in)
    
    return Model(inputs=model_input, outputs=model_output)

def model_2layer_with_dense(num_bits=10, num_digits=3, a_h1=20, a_h2=40, a_h3=24):
    '''
    Keras LSTM with bi-directional, uni-directional, and dense hidden layers.  
    
    arguments:
    num_bits -- number of bits in the input
    num_digits -- number of digits in the output
    a_h1 -- number of hidden units in first bidirectional LSTM (sequence-returning)
    a_h2 -- number of hidden units in second, uni-directional LSTM (non-sequence-returning)
    a_h3 -- number of hidden units in third dense layer
    '''
    
    model_input = Input(shape=(num_bits,1))
    model_input_bn = BatchNormalization()(model_input)
    lstm1_state = Bidirectional(LSTM(a_h1, return_sequences=True))(model_input_bn)
    lstm2_state = LSTM(a_h2, return_sequences=False)(lstm1_state)
    dense_in = Reshape((a_h2,))(lstm2_state)
    dense_state = Dense(a_h3)(dense_in)
    model_output = Dense(4 * num_digits, activation='sigmoid')(dense_state)
    
    return Model(inputs=model_input, outputs=model_output)

def model_3layer(num_bits=10, num_digits=3, a_h1=20, a_h2=20, a_h3=20):
    '''
    Create a Keras bidirectional LSTM model for computing the
    decimal representation of a number in binary.  
    
    arguments:
    num_bits -- number of bits in the input
    num_digits -- number of digits in the output
    a_h1 -- number of hidden units in first bidirectional LSTM
    a_h2 -- number of hidden units in second uni-directional LSTM
    
    returns:
    model -- Keras model instance
    '''
    
    
    model_input = Input(shape=(num_bits,1))
    lstm1_state = Bidirectional(LSTM(a_h1, return_sequences=True))(model_input)
    lstm2_state = Bidirectional(LSTM(a_h2, return_sequences=True))(lstm1_state)
    lstm3_state = LSTM(a_h3, return_sequences=False)(lstm2_state)
    dense_in = Reshape((a_h3,))(lstm3_state)
    model_output = Dense(4 * num_digits, activation='sigmoid')(dense_in)
    
    return Model(inputs=model_input, outputs=model_output)

def dense_model(num_bits=10, num_digits=3, a_h = 30):
    '''
    Creates a Keras "plain" neural network with three hidden layers.
    
    arguments:
    num_bits -- number of bits in input
    num_diigts -- number of digits in output
    a_h -- number of hidden units in each of the three hidden layers
    
    returns:
    model -- Keras model instance
    '''
    model_input = Input(shape=(num_bits,1))
    dense_in = Reshape((num_bits,))(model_input)
    dense1_state = Dense(a_h,activation='relu')(dense_in)
    dense2_state = Dense(a_h, activation='relu')(dense1_state)
    dense3_state = Dense(a_h, activation='relu')(dense2_state)
    model_output = Dense(num_digits * 4, activation='sigmoid')(dense3_state)
    
    return Model(inputs=model_input, outputs=model_output)