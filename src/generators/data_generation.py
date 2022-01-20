import numpy as np
import random, math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean
#! pip install fastdtw
from fastdtw import fastdtw
import sys, os, re, glob, csv


# Variant 1: sine curve dataset
def generateCurve(n, _freq, samplingrate, err, phase):
    trajectory = []
    n = int(n)
    for i in range(n):
        freq = random.uniform(_freq[0], _freq[1])
        line = np.arange(1, 101, samplingrate)
        error = [random.random()*err for x in range(len(line))] 
        l = np.sin((freq*line)+phase )+error
        trajectory.append(l)
    #print(len(data))
    return trajectory #random sine curve + random error

def get_curves(n_samples, ntrain_samples, ntest_samples, nexplain_samples, now_str=None):
    nclasses = 2
    pal = sns.color_palette("hls", nclasses)
    samplingrate = 1
    segments = [None]*n_samples
    labs = [None]*n_samples

    
    # create nsamples 
    if now_str is not None:
        for i in range(nclasses):                        
            freq = (round(random.uniform(0, 1),2), round(random.uniform(0,1), 2))                
            err = round(random.uniform(0, 1) ,2)              
            phase = int(random.uniform(-15, 15))
            print('Params, ', freq, err, phase)
            label = ','.join([str(x) for x in freq]) + '|' + str(err) + '|' + str(phase)
            class_samples = int(n_samples/nclasses)
            c1 = generateCurve(class_samples, freq, samplingrate, err, phase)
            segments[i*class_samples: ] = c1
            labs[i*class_samples: ] = [label]*(len(c1))
        data = list(zip(segments, labs))
        random.shuffle(data)
        classes = {k: v for v, k in enumerate(set(labs))}
        # divide into train, test and explain
        
        train = data[0:ntrain_samples]
        test = data[ntrain_samples:]
        explain = random.choices(test, k=nexplain_samples)
        x_train, y_train = zip(*train)
        x_test, y_test = zip(*test)
        x_explain, y_explain = zip(*explain)


        # write this dataset in a file
        path = 'datasets/sine-curve/'+now_str
        if not os.path.exists(path):
            os.makedirs(path)
        for n, x, y in [('train', x_train, y_train), ('test', x_test, y_test), ('explain', x_explain, y_explain)]:
            with open(path+'/sine-curve-X_'+n+'.csv', 'w', newline="") as myfile:
                wr = csv.writer(myfile)
                wr.writerows(x)
            with open(path+'/sine-curve-Y_'+n+'.csv', 'w', newline="") as myfile:
                wr = csv.writer(myfile)
                wr.writerow(y)

            # Plot the dataset
            X_embedded = TSNE(random_state=42, n_components =2).fit_transform(x)

            fig = plt.figure(figsize=(10,5))
            plt.title('sine-curve ' + n)
            for (_x, _y) in zip(X_embedded,y):
                plt.plot(*_x.T, marker='o', color=pal[classes[_y]])
                plt.annotate(str(classes[_y]), (_x[0], _x[1]), color=pal[classes[_y]])
            plt.savefig(path+'/raw-data-sine-curve-'+n+'.png')
            plt.close(fig)
    else:
        print('now_str not supplied???')
        sys.exit()
    return

def get_longblobsX(n_samples, ntrain_samples, ntest_samples, nexplain_samples, now_str=None):
    nclasses = 2
    pal = sns.color_palette("hls", nclasses)

    if now_str is not None:
        std = [0.9, 0.1]#[round(random.uniform(0, 1),2) for x in range(nclasses)]
        centers1 = [(5, 2), (5, 20)]
        print('Params, ', std)
           
        X, labs = make_blobs(n_samples=n_samples, cluster_std=std, centers=centers1, random_state=42)
        
        data = list(zip(X, labs))

        random.shuffle(data)

        # divide into train, test and explain
        
        train = data[0:ntrain_samples]
        test = data[ntrain_samples:]
        explain = random.choices(test, k=nexplain_samples)
        x_train, y_train = zip(*train)
        x_test, y_test = zip(*test)
        x_explain, y_explain = zip(*explain)


        # write this dataset in a file
        path = 'datasets/blobs/'+now_str
        if not os.path.exists(path):
            os.makedirs(path)
        for n, x, y in [('train', x_train, y_train), ('test', x_test, y_test), ('explain', x_explain, y_explain)]:
            with open(path+'/longXblobs-X_'+n+'.csv', 'w', newline="") as myfile:
                wr = csv.writer(myfile)
                wr.writerows(x)
            with open(path+'/longXblobs-Y_'+n+'.csv', 'w', newline="") as myfile:
                wr = csv.writer(myfile)
                wr.writerow(y)

            # Plot the dataset
            fig = plt.figure(figsize=(10, 5))
            plt.title('blobs-longX ' + n)
            for (_x, _y) in zip(x,y):
                plt.plot(_x[0], _x[1], marker='o', color=pal[_y])
                plt.annotate(_y, (_x[0], _x[1]), color=pal[_y])
            plt.savefig(path+'/raw-data-longXblobs-'+n+'.png')
            plt.close(fig)
    else:
        print('now_str not supplied???')
        sys.exit()

    return
    
def get_longblobsY(n_samples, ntrain_samples, ntest_samples, nexplain_samples, now_str=None):
    nclasses = 2
    pal = sns.color_palette("hls", nclasses)

    if now_str is not None:
        std = [0.9, 0.1]#[round(random.uniform(0, 1),2) for x in range(nclasses)]
        centers1 = [(2, 5), (10, 5)]
        print('Params, ', std)
           
        X, labs = make_blobs(n_samples=n_samples, cluster_std=std, centers=centers1, random_state=42)
        
        data = list(zip(X, labs))

        random.shuffle(data)

        # divide into train, test and explain
        
        train = data[0:ntrain_samples]
        test = data[ntrain_samples:]
        explain = random.choices(test, k=nexplain_samples)
        x_train, y_train = zip(*train)
        x_test, y_test = zip(*test)
        x_explain, y_explain = zip(*explain)


        # write this dataset in a file
        path = 'datasets/blobs/'+now_str
        if not os.path.exists(path):
            os.makedirs(path)
        for n, x, y in [('train', x_train, y_train), ('test', x_test, y_test), ('explain', x_explain, y_explain)]:
            with open(path+'/longYblobs-X_'+n+'.csv', 'w', newline="") as myfile:
                wr = csv.writer(myfile)
                wr.writerows(x)
            with open(path+'/longYblobs-Y_'+n+'.csv', 'w', newline="") as myfile:
                wr = csv.writer(myfile)
                wr.writerow(y)

            # Plot the dataset
            fig = plt.figure(figsize=(5, 10))
            plt.title('blobs-longY ' + n)
            for (_x, _y) in zip(x,y):
                plt.plot(_x[0], _x[1], marker='o', color=pal[_y])
                plt.annotate(_y, (_x[0], _x[1]), color=pal[_y])
            plt.savefig(path+'/raw-data-longYblobs-'+n+'.png')
            plt.close(fig)
    else:
        print('now_str not supplied???')
        sys.exit()

    return
    
def get_4blobs(n_samples, ntrain_samples, ntest_samples, nexplain_samples, now_str=None):
    nclasses = 2
    pal = sns.color_palette("hls", nclasses)

    if now_str is not None:
        std = [0.4, 0.5] #[round(random.uniform(0, 1),2) for x in range(nclasses)]
        centers1 = [(2, 5), (2, 15)]
        centers2 = [(15, 15), (15, 5)]
        print('Params, ', std)
           
        X, labs = make_blobs(n_samples=int(n_samples/2), cluster_std=std, centers=centers1, random_state=42)
        X2, labs2 = make_blobs(n_samples=int(n_samples/2), cluster_std=std, centers=centers2, random_state=42)

        
        data = list(zip(X, labs))
        data.extend(zip(X2, labs2))
        random.shuffle(data)

        # divide into train, test and explain
        
        train = data[0:ntrain_samples]
        test = data[ntrain_samples:]
        explain = random.choices(test, k=nexplain_samples)
        x_train, y_train = zip(*train)
        x_test, y_test = zip(*test)
        x_explain, y_explain = zip(*explain)


        # write this dataset in a file
        path = 'datasets/blobs/'+now_str
        if not os.path.exists(path):
            os.makedirs(path)
        for n, x, y in [('train', x_train, y_train), ('test', x_test, y_test), ('explain', x_explain, y_explain)]:
            with open(path+'/4blobs-X_'+n+'.csv', 'w', newline="") as myfile:
                wr = csv.writer(myfile)
                wr.writerows(x)
            with open(path+'/4blobs-Y_'+n+'.csv', 'w', newline="") as myfile:

                wr = csv.writer(myfile)
                wr.writerow(y)

            # Plot the dataset
            fig = plt.figure(figsize=(10,5))
            plt.title('4blobs ' + n)
            for (_x, _y) in zip(x,y):
                plt.plot(_x[0], _x[1], marker='o', color=pal[_y])
                plt.annotate(_y, (_x[0], _x[1]), color=pal[_y])
            plt.savefig(path+'/raw-data-4blobs-'+n+'.png')
            plt.close(fig)
    else:
        print('now_str not supplied???')
        sys.exit()

    return


def get_blobs(n_samples, ntrain_samples, ntest_samples, nexplain_samples, now_str=None):
    nclasses = 2
    pal = sns.color_palette("hls", nclasses)

    if now_str is not None:
        std = [round(random.uniform(0, 1),2) for x in range(nclasses)]
        print('Params, ', std)
        X, labs = make_blobs(n_samples=n_samples, centers=nclasses, cluster_std=std, random_state=42)
        
        data = list(zip(X, labs))
        random.shuffle(data)

        # divide into train, test and explain
        
        train = data[0:ntrain_samples]
        test = data[ntrain_samples:]
        explain = random.choices(test, k=nexplain_samples)
        x_train, y_train = zip(*train)
        x_test, y_test = zip(*test)
        x_explain, y_explain = zip(*explain)


        # write this dataset in a file
        path = 'datasets/blobs/'+now_str
        if not os.path.exists(path):
            os.makedirs(path)
        for n, x, y in [('train', x_train, y_train), ('test', x_test, y_test), ('explain', x_explain, y_explain)]:
            with open(path+'/blobs-X_'+n+'.csv', 'w', newline="") as myfile:
                wr = csv.writer(myfile)
                wr.writerows(x)
            with open(path+'/blobs-Y_'+n+'.csv', 'w', newline="") as myfile:
                wr = csv.writer(myfile)
                wr.writerow(y)

            # Plot the dataset
            fig = plt.figure(figsize=(10,5))
            plt.title('blobs ' + n)
            for (_x, _y) in zip(x,y):
                plt.plot(_x[0], _x[1], marker='o', color=pal[_y])
                plt.annotate(_y, (_x[0], _x[1]), color=pal[_y])
            plt.savefig(path+'/raw-data-blobs-'+n+'.png')
            plt.close(fig)
    else:
        print('now_str not supplied???')
        sys.exit()

    return 
    
def get_circles(n_samples, ntrain_samples, ntest_samples, nexplain_samples, now_str=None):
    nclasses = 2
    pal = sns.color_palette("hls", nclasses)

    if now_str is not None:
        
        
        noise = random.uniform(0,0.1)
        factors = np.arange (0.1, 0.9, 0.1)
        factor = random.choice(factors)

        print('Params, ', noise, factor)    
        X, labs = make_circles(n_samples=n_samples, random_state=3, noise=noise, factor=factor)

        data = list(zip(X, labs))
        random.shuffle(data)

        # divide into train, test and explain
        
        train = data[0:ntrain_samples]
        test = data[ntrain_samples:]
        explain = random.choices(test, k=nexplain_samples)
        x_train, y_train = zip(*train)
        x_test, y_test = zip(*test)
        x_explain, y_explain = zip(*explain)


        # write this dataset in a file
        path = 'datasets/circles/'+now_str
        if not os.path.exists(path):
            os.makedirs(path)
        for n, x, y in [('train', x_train, y_train), ('test', x_test, y_test), ('explain', x_explain, y_explain)]:
            with open(path+'/circles-X_'+n+'.csv', 'w', newline="") as myfile:
                wr = csv.writer(myfile)
                wr.writerows(x)
            with open(path+'/circles-Y_'+n+'.csv', 'w', newline="") as myfile:
                wr = csv.writer(myfile)
                wr.writerow(y)
            # Plot the dataset
            fig = plt.figure(figsize=(10,5))
            plt.title('cirlces ' + n)
            for (_x, _y) in zip(x,y):
                plt.plot(_x[0], _x[1], marker='o', color=pal[_y])
                plt.annotate(_y, (_x[0], _x[1]), color=pal[_y])
            plt.savefig(path+'/raw-data-circles-'+n+'.png')
            plt.close(fig)
    else:
        print('now_str not supplied???')
        sys.exit()
       
    return     


# Variant: handwritten dataset
def parseFile(lines):
    points = dict()
    newchar = False
    cont = False
    point = []
    cclass = None
    
    #classes_ = [x.strip('"') for x in lines[0].split(' ')][1:-1]
    for line in lines[1:]:
        if '.COMMENT' in line and 'Class' in line and '[' in line and '#' not in line:
            b = re.findall('.*?\.COMMENT\s+Class\s+\[(.*?)\]', line)
            cclass = b[0]
            #print(cclass)
            newchar = True
            point = []
            continue
        if '.PEN_UP' in line:
            cont = False
            if cclass not in points.keys():
                points[cclass] = []
            points[cclass].append(point)
        if '.PEN_DOWN' in line:
            cont = True
            continue
        if newchar and cont:
            b = re.findall('.*?(\d+)\s+([-\d]+).*', line)
            #print(line)
            xy = b[0]
            point.append((int(xy[0]), int(xy[1])))

    return points
    
    
def get_digits(classes, nclasses, nprototypes, path):
    
    LIM = 10 # limit how many samples per class
    classdict = {k: v for v, k in enumerate(classes)}
    print(classdict)
    nclasses = len(classes)
    segments = dict()
    files = glob.glob(path+'/*') # Path to dataset

    for f in files:
        f_ = open(f, 'r')
        lines = f_.readlines()
        content = parseFile(lines)
        for cclass, segment in content.items():
            if cclass not in classes:
                continue
            if cclass not in segments.keys():
                segments[cclass] = []
            #if len(segments[cclass]) > LIM:
            #    continue
            segments[cclass].extend(segment)
        f_.close()


    all_segments = [item for sublist in segments.values() for item in sublist]
    print(len(all_segments))
    data = all_segments

    return 




    

