import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Training Data
# -----------------------------
x_train = np.array([1.0, 2.0])   # feature: size (1000 sqft)
y_train = np.array([300.0, 500.0])   # target: price (1000s of dollars)


# -----------------------------
# 2. Cost Function
# -----------------------------
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    return (1 / (2 * m)) * cost


# -----------------------------
# 3. Gradient Function
# -----------------------------
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


# -----------------------------
# 4. Gradient Descent Algorithm
# -----------------------------
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_history = []
    p_history = []
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        J_history.append(cost_function(x, y, w, b))
        p_history.append([w, b])
        
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw:0.3e}, dj_db: {dj_db:0.3e} ",
                  f"w: {w:0.3e}, b: {b:0.5e}")
            
    return w, b, J_history, p_history


# -----------------------------
# 5. Run Gradient Descent
# -----------------------------
w_init = 0
b_init = 0
iterations = 10000
alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient
)

print(f"\n(w, b) found by gradient descent: ({w_final:.4f}, {b_final:.4f})")


# -----------------------------
# 6. Plot Cost vs Iterations
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(J_hist[:100])
ax1.set_title("Cost vs. Iteration (start)")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Cost")

ax2.plot(range(1000, iterations), J_hist[1000:])
ax2.set_title("Cost vs. Iteration (end)")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Cost")
plt.show()


# -----------------------------
# 7. Plot Fitted Line vs Training Data
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(x_train, y_train, color="red", label="Training Data")
x_line = np.linspace(0, 3, 100)
y_line = w_final * x_line + b_final
plt.plot(x_line, y_line, label=f"Fitted Line: y={w_final:.2f}x+{b_final:.2f}")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")
plt.title("Best Fit Line after Gradient Descent")
plt.legend()
plt.show()


# -----------------------------
# 8. Contour Plot of Cost Function
# -----------------------------
w_vals = np.linspace(0, 300, 100)
b_vals = np.linspace(0, 200, 100)
J_vals = np.zeros((len(w_vals), len(b_vals)))

for i in range(len(w_vals)):
    for j in range(len(b_vals)):
        J_vals[i, j] = compute_cost(x_train, y_train, w_vals[i], b_vals[j])

W, B = np.meshgrid(w_vals, b_vals)

plt.figure(figsize=(8,6))
CS = plt.contour(W, B, J_vals.T, levels=np.logspace(0, 5, 35), cmap="viridis")
plt.clabel(CS, inline=True, fontsize=8)
p_hist = np.array(p_hist)
plt.plot(p_hist[:,0], p_hist[:,1], 'ro-', markersize=3, linewidth=1, label="Gradient Descent Path")
plt.xlabel("w")
plt.ylabel("b")
plt.title("Contour Plot of Cost Function with Gradient Descent Path")
plt.legend()
plt.show()


# -----------------------------
# 9. Divergence Example with Large Alpha
# -----------------------------
w_init = 0
b_init = 0
iterations = 10
alpha = 0.8  # too large

w_final_bad, b_final_bad, J_hist_bad, p_hist_bad = gradient_descent(
    x_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient
)

plt.figure(figsize=(6,4))
plt.plot(J_hist_bad, 'r-o')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Divergence with Too Large Learning Rate")
plt.show()


# -----------------------------
# 10. Predictions
# -----------------------------
print(f"Prediction for 1000 sqft house: {w_final*1.0 + b_final:.1f} thousand dollars")
print(f"Prediction for 1200 sqft house: {w_final*1.2 + b_final:.1f} thousand dollars")
print(f"Prediction for 2000 sqft house: {w_final*2.0 + b_final:.1f} thousand dollars")
