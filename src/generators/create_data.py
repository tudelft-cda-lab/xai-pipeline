import sys, os, re, glob, math
import seaborn as sns
from datetime import datetime
from data_generation import get_curves, get_digits, get_blobs, get_4blobs, get_circles, get_longblobsY, get_longblobsX


if len(sys.argv) < 4:
    print('USAGE: create_data.py {sine-curve|blobs|4blobs|xblobs|yblobs|circles} {#samples} {test-split [0.0-0.9]} {#explain-samples [< test-split*#samples]}')
    sys.exit()

DATASET = sys.argv[1]
nsamples = int(sys.argv[2])
test_split = float(sys.argv[3])
if test_split < 0 or test_split > 0.9:
    print("Error: Incorrect test-split supplied")
    sys.exit()
train_split = 1.0 - test_split

ntest_samples = int(test_split*nsamples)
ntrain_samples = int(train_split*nsamples)
nexplain_samples = int(sys.argv[4])

if nexplain_samples < 1 or nexplain_samples > ntest_samples:
    print("Error: Incorrect number of explain samples supplied")
    sys.exit()

assert (nsamples <= (ntrain_samples + ntest_samples + nexplain_samples))

print('#Samples: %d | #Train: %d | #Test: %d | #Explain: %d'%(nsamples, ntrain_samples, ntest_samples, nexplain_samples))

now = datetime.now()
now_str = str(nsamples)+'samples-'+now.strftime("%d%m%y-%H%M%S")


print('Generating the data...')
if DATASET == 'sine-curve':
     get_curves(nsamples, ntrain_samples, ntest_samples, nexplain_samples, now_str)
elif DATASET == 'blobs':
        get_blobs(nsamples, ntrain_samples, ntest_samples, nexplain_samples, now_str)
elif DATASET == '4blobs':
        get_4blobs(nsamples, ntrain_samples, ntest_samples, nexplain_samples, now_str)
elif DATASET == 'yblobs':
        get_longblobsY(nsamples, ntrain_samples, ntest_samples, nexplain_samples, now_str)
elif DATASET == 'xblobs':
        get_longblobsX(nsamples, ntrain_samples, ntest_samples, nexplain_samples, now_str)
elif DATASET == 'circles':
        get_circles(nsamples, ntrain_samples, ntest_samples, nexplain_samples, now_str)
else:
    print('Dataset not recognized')
    sys.exit(-1) 

