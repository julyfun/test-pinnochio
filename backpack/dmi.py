import pinocchio
import numpy as np

# 假设 dMi 是一个变换矩阵
dMi = pinocchio.SE3(np.array([
    [0.866, -0.5, 0, 0.1],
    [0.5, 0.866, 0, 0.2],
    [0, 0, 1, 0.3],
    [0, 0, 0, 1]
])
                    )

# 计算对数映射
log_dMi = pinocchio.log(dMi)

# 获取误差向量
err = log_dMi.vector

print("Error vector:")
print(err)
