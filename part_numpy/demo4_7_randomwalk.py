import numpy as np
import random
import matplotlib.pyplot as plt

def get_step():
    """获取步长"""
    step = random.choice([-1, 1])
    return step
def randomwalk():
    """随机漫步"""
    position = 0
    steps = 10000
    walk = [position]
    for i in range(steps):
        step = get_step()
        position += step
        walk.append(position)
    return walk

def plot_walk(walk):
    """绘制漫步图"""
    plt.plot(walk)
    plt.title("Random Walk")
    plt.xlabel("Steps")
    plt.ylabel("Position")
    plt.show()


if __name__ == "__main__":
    walk = randomwalk()
    plot_walk(walk)