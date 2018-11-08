import numpy as np

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
        bit = dec % 2
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
    return dec

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

    decbits = np.asarray([int(s) for s in str(dec)])
    length = np.size(decbits)
    if pad:
        decbits = zero_pad_left(decbits, num_digits)
    return np.reshape(np.asarray(list(map(decimal_digit_to_4bits, decbits))), (num_digits * 4,))

def generate_datapoint():
    '''
    Picks a number n in the range 0 - 999 uniformly at random.  Outputs a pair
    [n_bin, n_dec] for the learning problem of converting binary to decimal.

    returns:
    A pair [n_bin, n_dec] consisting of a binary and decimal representation of
    a number in the range 0 - 999.
    '''
    n = np.random.randint(0,1000)
    return [zero_pad_left(dec_to_bin(n),10), dec_to_decbits(n)]
