import numpy as np
import ipywidgets as ipw

def classify_image(tst,trn,k):
    """  Input: 
    tst: a n x T vector corresponding to T images that are to be classified
    trn: n x m x 10  matrix containing m training samples for each digit
    k: integer between 1 and n
    """
    n,T = tst.shape
    m = trn.shape[1]
    # compute projection matrices:
    P = np.zeros((n,n,10))
    for i in range(10):
        U,S,V = np.linalg.svd(trn[:,:,i])
        U1 = U[:,0:k]
        P[:,:,i] = np.dot(U1,U1.T)

    # find errors:
    sqerr = np.zeros((10,T))
    for i in range(10):
        sqerr[i,:] = np.sum((tst - np.dot(np.squeeze(P[:,:,i]),tst))**2,0)

    return np.argmin(sqerr,0)

def linear_combo(a1,a2,a3,digit,trn):
    U,S,V = np.linalg.svd(trn[:,:,digit])
    U1 = U[:,0]
    U2 = U[:,1]
    U3 = U[:,2]
    v = a1*U1 + a2*U2 + a3*U3
    return v

def vec2mat(v):
    # shift to zero:
    v = (v - min(v))
    # scale to 1:
    v = v/max(v)
    # shift to [-0.5,0.5]:
    v = v - 0.5
    return np.reshape(v,(16,16))

def return_widgets():
    c1=ipw.FloatSlider(
        value=1,
        min=0.1,
        max=1,
        step=0.1,
        width=50,
        height=10)

    c2=ipw.FloatSlider(
        value=0.0,
        min=-0.5,
        max=0.5,
        step=0.1,
        width=50,
        height=10)

    c3=ipw.FloatSlider(
        value=0.0,
        min=-0.5,
        max=0.5,
        step=0.1,
        width=50,
        height=10,
        padding=0)

    d = ipw.Dropdown(
        options=[str(i) for i in range(10)],
        value='0',
        description='Digit:')

    return c1,c2,c3,d

