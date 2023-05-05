import copy, math
import numpy as np

# linear regression with multiple variables related functions

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(w, X[i]) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

def compute_gradient (X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    
    for i in range(m):
        err = (np.dot(w, X[i]) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
   
    # An array to store cost J at each iteration primarily for graphing later
    J_history = []
    w0_hist = []
    # w1_hist = w1_hist[]
    b_hist = []
    
    w = copy.deepcopy(w_in)
    b = copy.deepcopy(b_in)
    
    # print(f"Iteration Cost          w0       w1       w2       w3       b       djdw0    djdw1    djdw2    djdw3    djdb  ")
    # print(f"---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|")
    
    for i in range (num_iters):
        
        # calculate gradients
        dj_db, dj_dw = gradient_function (X, y, w, b)
        
        # update params using w, b, alpha & gradients
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # save cost J at each iteration
        if i < 100000:
            J_history.append (cost_function(X, y, w, b))
            w0_hist.append (w[0])
            # w1_hist.append (w[1])
            b_hist.append (b) 
            
        if i% math.ceil(num_iters/10) == 0:
            cst = J_history[i]
            print (f"iteration: {i} :: w0 = {w[0]}, b = {b}")
            # print(f"{i:9d} {cst:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_dw[3]: 0.1e} {dj_db: 0.1e}")     
        
    
    print (f"w0 final = {w[0]}, b final = {b}")
    
    return w, b, J_history, w0_hist, b_hist
