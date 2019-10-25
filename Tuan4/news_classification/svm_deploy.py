import numpy as np
from sklearn import metrics

def svm_loss_naive(W, X, y, reg):
    d, C = W.shape
    _, N = X.shape

    # naive loss and grad
    loss = 0
    dW = np.zeros_like(W)
    for n in range(N):
        xn = X[:, n]
        score = W.T.dot(xn)
        for j in range(C):
            if j == y[n]:
                continue
            margin = 1 - score[y[n]] + score[j]
            if margin > 0:
                loss += margin
                dW[:, j] += xn
                dW[:, y[n]] -= xn

    loss /= N
    loss += 0.5*reg*np.sum(W * W) # regularization

    dW /= N
    dW += reg*W # gradient off regularization
    return loss, dW

# Mini-batch gradient descent
def multiclass_svm_GD_naive(X, y, Winit, reg, lr=.1, \
        batch_size = 100, num_iters = 1000, print_every = 100):
    W = Winit
    loss_history = np.zeros((num_iters))
    for it in range(num_iters):
        # randomly pick a batch of X
        idx = np.random.choice(X.shape[1], batch_size)
        X_batch = X[:, idx]
        y_batch = y[idx]

        loss_history[it], dW = \
            svm_loss_naive(W, X_batch, y_batch, reg)

        W -= lr*dW
        # if it % print_every == 1:
        #     print('it %d/%d, loss = %f' \
        #         %(it, num_iters, loss_history[it]))

    return W, loss_history

def model_with_svm_naive(X_data, y_data, X_test, y_test,Winit):
    reg = 0.1
    W , loss = multiclass_svm_GD_naive(X_data, y_data,Winit,reg)
    score_vector = np.dot(W.T, X_test)
    score_vector = score_vector.T
    pred_labels = np.zeros_like(y_test)
    for i in range(score_vector.shape[0]):
        pred_labels[i] = np.argmax(score_vector[i])
    print("Test accuracy with loss function 1: ", metrics.accuracy_score(pred_labels, y_test))
    return pred_labels




# more efficient way to compute loss and grad
def svm_loss_vectorized(W, X, y, reg):
    d, C = W.shape
    _, N = X.shape
    loss = 0
    dW = np.zeros_like(W)
    Z = W.T.dot(X)

    correct_class_score = np.choose(y, Z).reshape(N,1).T
    margins = np.maximum(0, Z - correct_class_score + 1)
    margins[y, np.arange(margins.shape[1])] = 0
    loss = np.sum(margins, axis = (0, 1))
    loss /= N
    loss += 0.5 * reg * np.sum(W * W)

    F = (margins > 0).astype(int)
    F[y, np.arange(F.shape[1])] = np.sum(-F, axis = 0)
    dW = X.dot(F.T)/N + reg*W
    return loss, dW

def multiclass_svm_GD_vectorized(X, y, Winit, reg, lr=.1, \
        batch_size = 100, num_iters = 1000, print_every = 100):
    W = Winit
    loss_history = np.zeros((num_iters))
    for it in range(num_iters):
        # randomly pick a batch of X
        idx = np.random.choice(X.shape[1], batch_size)
        X_batch = X[:, idx]
        y_batch = y[idx]

        loss_history[it], dW = \
            svm_loss_vectorized(W, X_batch, y_batch, reg)

        W -= lr*dW
        # if it % print_every == 1:
        #     print('it %d/%d, loss = %f' \
        #         %(it, num_iters, loss_history[it]))

    return W, loss_history

def model_svm_GD_vectorized(X_data, y_data, X_test, y_test,Winit):
    reg = 0.1
    W , loss = multiclass_svm_GD_vectorized(X_data, y_data,Winit,reg)
    score_vector = np.dot(W.T, X_test)
    score_vector = score_vector.T
    pred_labels = np.zeros_like(y_test)
    for i in range(score_vector.shape[0]):
        pred_labels[i] = np.argmax(score_vector[i])
    print("Test accuracy with loss function 2: ", metrics.accuracy_score(pred_labels, y_test))
    return pred_labels
