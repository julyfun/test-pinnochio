import numpy as np

# 假设 J 是雅可比矩阵，err 是误差向量，damp 是阻尼因子
J = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
err = np.array([0.1, 0.2, 0, 0, 0, 0])
damp = 0.01

# 计算阻尼伪逆
A = J.dot(J.T) + damp * np.eye(6)
x = np.linalg.solve(A, err)

# 计算速度向量 v
v = - J.T.dot(x)

print("Velocity vector v:")
print(v)
