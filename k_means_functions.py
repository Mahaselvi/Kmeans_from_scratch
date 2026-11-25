import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#Intialising the centroids
def init_centroids(x,k,rng):
    #rng is a numpy object
    idx=rng.choice(len(x),size=k,replace=False)   # k different numbers are chosen from range len(x)-1
    #replace=False ensures there is no repetitions in the numbers chosen
    # returns the datapoints in the indices in idx
    return x[idx].copy()


def init_centroids_pp(x,k,rng):
    #integers() function returns a random value from range 0 to len(x)-1
    centroids=[x[rng.integers(len(x))]]
    for i in range(k-1):
        d2=np.min(
            ((x[:,None,:]-np.array(centroids)[None,:,:])**2).sum(axis=2),
            axis=1,
            )

        probs=d2/d2.sum()
        centroids.append(x[rng.choice(len(x),p=probs)])
    return np.array(centroids)


def assign_clusters(x,centroids):
    distances=np.linalg.norm(x[:,None,:]-centroids[None,:,:],axis=2)
    return np.argmin(distances,axis=1)

def update_centroids(x,labels,k,old_centroids,rng):
    new_centroids=np.zeros_like(old_centroids)
    for j in range(k):
        points=x[labels==j]
        if len(points)==0:
            new_centroids[j]=x[rng.integers(len(x))]
        else:
            new_centroids[j]=points.mean(axis=0)
    return new_centroids


def compute_inertia(x,labels,centroids):
    return ((x-centroids[labels])**2).sum()


def kmeans(x,k,max_iters=100,tol=1e-4,seed=0,pp=False):
    rng=np.random.default_rng(seed)
    init=init_centroids_pp if pp else init_centroids
    centroids=init(x,k,rng)
    inertia_history=[] 


    for i in range(max_iters):
        labels=assign_clusters(x,centroids)
        new_centroids=update_centroids(x,labels,k,centroids,rng)
        inertia_history.append(compute_inertia(x,labels,new_centroids))

        shift=np.abs(new_centroids-centroids).max()
        centroids=new_centroids
        if shift<tol:
            break

    labels=assign_clusters(x,centroids)
    return labels,centroids,inertia_history

