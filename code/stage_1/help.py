import math

def isPowerOfTwo(n):
    return (math.ceil(math.log2(n)) == math.floor(math.log2(n)))