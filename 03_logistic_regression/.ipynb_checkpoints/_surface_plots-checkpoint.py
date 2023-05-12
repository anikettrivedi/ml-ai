import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from _logistic_regression_util import sigmoid

######################################################################
############# simplified demonstations of linear regression cost in 3D
######################################################################

def plot_linear_regression_cost_3d():
    
    fig = plt.figure(figsize=(4,4))
    
    ax = fig.add_subplot(111, projection='3d')
    
    w = np.linspace(-20,20,100)
    b = np.linspace(-20,20,100)
    
    z = np.zeros((len(w), len(b)))
    
    j = 0
    for x in w:
        i = 0
        for y in b:
            z[i,j] = x**2 + y**2
            i+=1
        j+=1
        
    W, B = np.meshgrid(w,b)
    
    ax.plot_surface(W, B, z, cmap="Spectral_r", alpha=0.7, antialiased=False)
    ax.plot_wireframe(W, B, z, color='k', alpha=0.1)
    ax.set_xlabel('$w$')
    ax.set_ylabel('$b$')
    ax.set_zlabel('Cost', rotation=90)
    ax.set_title('Squared error cost used in Linear Regression')
    plt.show()
    

######################################################################
##### plotting squared error cost (loss) for logistic regression in 3D
######################################################################

def compute_cost_logistic_square_error(X, y, w, b):
    
    m = X.shape[0]
    cost = 0.0
    
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost = cost + (f_wb_i - y[i])**2
    
    cost = cost/(2*m)
    return np.squeeze(cost)

def plot_logistic_regression_squared_error_cost(X, y):
    
    # selected values for w, b for demonstration purpose
    wx, by = np.meshgrid (np.linspace(-6,12,50), np.linspace(10,-20, 40))
    
    points = np.c_[wx.ravel(), by.ravel()]
    cost = np.zeros(points.shape[0])
    
    for i in range(points.shape[0]):
        w,b = points[i]
        cost[i] = compute_cost_logistic_square_error(X.reshape(-1,1), y, w, b)
    
    cost = cost.reshape(wx.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(wx, by, cost, alpha=0.6, cmap=cm.jet)
    
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('Square Error Cost', rotation=90)
    
    ax.set_title('Logistic Squared Error Cost vs (w,b)')

############################################################
######## plotting two curves of logistic loss function in 2D
############################################################

def plot_two_logistic_loss_curves():
    
    fig, ax = plt.subplots(1, 2, sharey=True)
    
    x = np.linspace(0.01,0.99,20)
    
    ax[0].plot(x, -np.log(x))
    ax[0].text(0.5, 4.0,"y=1")
    ax[0].set_ylabel("loss")
    ax[0].set_xlabel(r"$f_{w,b}(x)$")
    
    ax[1].plot(x, -np.log(1-x))
    ax[1].text(0.5, 4.0, "y=0")
    ax[1].set_xlabel(r"$f_{w,b}(x)$")
    
    plt.suptitle("Loss Curves for Two Categorical Target Values", fontsize=12)
    plt.tight_layout()
    plt.show()

############################################################
######## plotting two curves of logistic loss function in 3D
############################################################

def log_1pexp(x, maximum=20):
    ''' approximate log(1+exp^x)
        https://stats.stackexchange.com/questions/475589/numerical-computation-of-cross-entropy-in-practice
    Args:
    x   : (ndarray Shape (n,1) or (n,)  input
    out : (ndarray Shape matches x      output ~= np.log(1+exp(x))
    '''

    out  = np.zeros_like(x,dtype=float)
    i    = x <= maximum
    ni   = np.logical_not(i)

    out[i]  = np.log(1 + np.exp(x[i]))
    out[ni] = x[ni]
    return out

def compute_cost_matrix(X, y, w, b):
    """
    Computes the cost using  using matrices
    Args:
      X : (ndarray, Shape (m,n))          matrix of examples
      y : (ndarray  Shape (m,) or (m,1))  target value of each example
      w : (ndarray  Shape (n,) or (n,1))  Values of parameter(s) of the model
      b : (scalar )                       Values of parameter of the model
      verbose : (Boolean) If true, print out intermediate value f_wb
    Returns:
      total_cost: (scalar)                cost
    """
    m = X.shape[0]
    y = y.reshape(-1,1)             # ensure 2D
    w = w.reshape(-1,1)             # ensure 2D
  
    z = X @ w + b                                                           #(m,n)(n,1)=(m,1)
    cost = -(y * z) + log_1pexp(z)
    cost = np.sum(cost)/m                                                   # (scalar)

    return cost     
    
# plotting logistic cost function
def plot_logistic_regression_logistic_cost(X,y):
    
    wx, by = np.meshgrid(np.linspace(-6,12,50), np.linspace(0, -20, 40))
    points = np.c_[wx.ravel(), by.ravel()]
    cost = np.zeros(points.shape[0],dtype=np.longdouble)
    
    for i in range(points.shape[0]):
        w,b = points[i]
        cost[i] = compute_cost_matrix(X.reshape(-1,1), y, w, b)
    
    cost = cost.reshape(wx.shape)
    
    fig = plt.figure(figsize=(9,5))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(wx, by, cost, alpha=0.6,cmap=cm.jet,)

    ax.set_xlabel('w', fontsize=16)
    ax.set_ylabel('b', fontsize=16)
    ax.set_zlabel("Cost", rotation=90, fontsize=16)
    ax.set_title('Logistic Cost vs (w, b)')

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(wx, by, np.log(cost), alpha=0.6,cmap=cm.jet,)

    ax.set_xlabel('w', fontsize=16)
    ax.set_ylabel('b', fontsize=16)
    ax.set_zlabel('\nlog(Cost)', fontsize=16)
    ax.set_title('log(Logistic Cost) vs (w, b)')
    
    plt.show()
    return cost