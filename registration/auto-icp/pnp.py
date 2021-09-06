import numpy as np
from math import sin, cos, pi

def solve_pnp(source, target):
    source_center = np.mean(source, axis=1, keepdims=True)
    target_center = np.mean(target, axis=1, keepdims=True)
    W = (target - target_center).dot((source - source_center).T)
    u, _, vt = np.linalg.svd(W)
    R = u.dot(vt)
    t = target_center - R.dot(source_center)
    print(R, t)

if __name__ == '__main__':
    source = np.array([[1.0,2.0],
                 [2,3],
                 [3,7]]).T
    
    theta = pi/4
    R = np.array([[cos(theta), -sin(theta)],
                  [sin(theta), cos(theta)]])
    t = np.array([[1, 1]]).T
    n = np.random.randn(source.shape[0], source.shape[1])/100
    target = R.dot(source) + t + n
    print(R)
    solve_pnp(source, target)
        