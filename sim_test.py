import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 

N = 10000

def main():
    batA_position = np.random.randint(0, 300, (1, 2))[0]
    print(batA_position)
    p_list = []
    for i in range(N):
        p = 100
        p_list.append(p)

        v =
    
    with open("data.txt", tyoe="w") as f:
        f.writelines(p_list)
    
    plt.plot(p_list)
    plt.savefig("tt.png")
    plt.show()




if __name__ == "__main__":
    main()