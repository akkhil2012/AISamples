import numpy as np

# simple linear regression with gradient descent
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# initialise
w = 0.0          # weight
b = 0.0          # bias
lr = 0.01        # learning rate
epochs = 1000

for epoch in range(epochs):
    # forward pass — predictions
    y_pred = w * X + b

    # loss — mean squared error
    loss = np.mean((y_pred - y) ** 2)

    # gradients
    dw = np.mean(2 * (y_pred - y) * X)   # ∂L/∂w
    db = np.mean(2 * (y_pred - y))        # ∂L/∂b

    # update weights — gradient descent step
    w = w - lr * dw
    b = b - lr * db

    if epoch % 100 == 0:
        print(f"epoch {epoch}: loss={loss:.4f} w={w:.4f} b={b:.4f}")

# output converges to w≈2.0, b≈0.0 (correct)