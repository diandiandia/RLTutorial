import numpy as np
import matplotlib.pyplot as plt

# Beale 函数（二维测试函数）
def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

# 梯度
def beale_grad(x, y):
    # 避免数值溢出的保护措施
    x = np.clip(x, -10, 10)
    y = np.clip(y, -10, 10)
    
    # 计算梯度
    term1 = 2 * (1.5 - x + x*y)
    term2 = 2 * (2.25 - x + x*y**2)
    term3 = 2 * (2.625 - x + x*y**3)
    
    dx = term1 * (y - 1) + term2 * (y**2 - 1) + term3 * (y**3 - 1)
    dy = term1 * x + 4.5 * (2.25 - x + x*y**2) * (x*y) + 7.875 * (2.625 - x + x*y**3) * (x*y**2)
    
    # 再次限制梯度值，防止爆炸
    dx = np.clip(dx, -1e6, 1e6)
    dy = np.clip(dy, -1e6, 1e6)
    
    return np.array([dx, dy])

# SGD 模拟
def sgd_path(steps=1000, lr=0.0001, noise_scale=0.1):
    theta = np.array([3.0, 3.0])  # 初始点
    path = [theta.copy()]
    for _ in range(steps):
        grad = beale_grad(theta[0], theta[1])
        noise = noise_scale * np.random.randn(2)  # SGD的随机噪声
        theta -= lr * (grad + noise)
        
        # 限制theta的范围，防止溢出
        theta = np.clip(theta, -10, 10)
        
        path.append(theta.copy())
    return np.array(path)

# GD 路径（无噪声）
def gd_path(steps=200, lr=0.001):
    theta = np.array([3.0, 3.0])
    path = [theta.copy()]
    for _ in range(steps):
        grad = beale_grad(theta[0], theta[1])
        theta -= lr * grad
        
        # 限制theta的范围，防止溢出
        theta = np.clip(theta, -10, 10)
        
        path.append(theta.copy())
    return np.array(path)

# 绘图
x = np.linspace(-4.5, 4.5, 400)
y = np.linspace(-4.5, 4.5, 400)
X, Y = np.meshgrid(x, y)
Z = beale(X, Y)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.contour(X, Y, Z, levels=20, alpha=0.6, colors='white', linewidths=0.5)

path_sgd = sgd_path()
path_gd = gd_path()

plt.plot(path_gd[:,0], path_gd[:,1], 'r-o', markersize=3, label='GD (Batch)')
plt.plot(path_sgd[:,0], path_sgd[:,1], 'b-o', markersize=3, alpha=0.7, label='SGD')

plt.xlabel('θ1')
plt.ylabel('θ2')
plt.title('SGD vs GD on Beale Function')
plt.legend()
plt.colorbar(label='Loss')

# 在非交互环境中保存图像而不是显示
plt.savefig('sgd_vs_gd_beale.png', dpi=300, bbox_inches='tight')
# plt.show()  # 注释掉交互式显示
print("图像已保存为 sgd_vs_gd_beale.png")