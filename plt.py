import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

if __name__ == '__main__':
    import numpy as np
    plt.plot(np.arange(1, 100, 1))
    plt.savefig('a.png')
