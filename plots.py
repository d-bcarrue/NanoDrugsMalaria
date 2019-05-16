
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

path = "./results/resultados_EXTRATREES.bin"
path_df = "./datasets/ds.raw.csv"

df = pd.read_csv(path_df)
def plot_no_ordenado(feature_imp,standar_dev,ruta,add = None):
    """This function uses as input the importance of the features,
    their standard deviation and finally the path to the file to be created.
    the generated graph arranges the features from major to minor to facilitate the visualization."""
    plt.figure(figsize=(14,5))
    plt.xlim((0,107))
    plt.ylim((0,np.max(feature_imp + standar_dev) + 0.005))
    if not add == None:
        x = [ 0 if i in add else j for i,j in enumerate(feature_imp)]
        x2 = [ j for i,j in enumerate(feature_imp) if i in add]
        err = [ 0 if i in add else j for i,j in enumerate(standar_dev)]
        err2 = [ j for i,j in enumerate(standar_dev) if i in add]
        plt.bar([i for i in range(len(feature_imp))],x,yerr=err,color='#369dbc',align='edge')
        plt.bar(add,x2,yerr=err2,color='#bc369d',align='edge')
    else:
        plt.bar([i for i in range(len(feature_imp))],
                feature_imp,
                yerr=standar_dev,
                color='#369dbc',
                align='edge')
    plt.plot((-1,110),(np.mean(feature_imp),np.mean(feature_imp)),'--r')
    plt.ylabel('Mean impurity decrease')
    plt.xlabel('Index of features')
    plt.savefig('{}.png'.format(ruta),format='png')
    plt.show()

def plot_ordenado(feature_imp,standar_dev,ruta,add = None):
    """This function uses as input the importance of the features,
    their standard deviation and finally the path to the file to be created.
    The order of the bars corresponds to the original order """
    st_orde = standar_dev[np.argsort(feature_imp)[::-1]]
    features = np.sort(feature_imp)[::-1]
    if add != None:
        c = enumerate(sorted(enumerate(list(feature_imp)),key=lambda x: x[1], reverse=True))
        add = [i for i,(j,_) in c if j in add]
    plt.figure(figsize=(14,5))
    plt.xlim((0,107))
    plt.ylim((0,np.max(feature_imp + standar_dev) + 0.005))
    if not add == None:
        x = [ 0 if i in add else j for i,j in enumerate(features)]
        x2 = [ j for i,j in enumerate(features) if i in add]
        err = [ 0 if i in add else j for i,j in enumerate(st_orde)]
        err2 = [ j for i,j in enumerate(st_orde) if i in add]
        plt.bar([i for i in range(len(features))],x,yerr=err,color='#369dbc',align='edge')
        plt.bar(add,x2,yerr=err2,color='#bc369d',align='edge')
    else:
        plt.bar([i for i in range(len(feature_imp))],
                np.sort(feature_imp)[::-1],
                yerr=st_orde,color='#369dbc',
                align='edge')

    plt.plot((-1,110),(np.mean(feature_imp),np.mean(feature_imp)),'--r')
    plt.ylabel('Mean impurity decrease')
    plt.xlabel('Index of sorted features')
    plt.savefig('{}.png'.format(ruta),format='png')
    plt.show()


df = pd.read_csv('/home/diego/datasets/ds.raw.csv')
with open(path, 'rb') as file:
    a = pickle.load(file)
feature_imp = np.mean(a,axis=0)
standar_dev = np.std(a, axis=0)

plot_ordenado(feature_imp,
              standar_dev,
              './results/plot_ordenado_extratrees',
              [i for i,j in enumerate(df.columns[1:]) if j in ['np_DPnpu(c2)','np_DPnpu(c4)']])

plot_no_ordenado(feature_imp,
                 standar_dev,
                 './results/plot_no_ordenado_extratrees',
                 [i for i,j in enumerate(df.columns[1:]) if j in ['np_DPnpu(c2)','np_DPnpu(c4)']])
