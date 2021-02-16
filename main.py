import BPNN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

if __name__ == "__main__":

    #Create data
    # x_array_train = np.random.uniform(1, 10, 400)
    # y_array_train = np.random.uniform(1, 10, 400)
    # np.savetxt("x_array_train.txt", x_array_train)
    # np.savetxt("y_array_train.txt", y_array_train)

    # x_array_validation = np.random.uniform(1, 10, 200)
    # y_array_validation = np.random.uniform(1, 10, 200)
    # np.savetxt("x_array_validation" + startTime + ".txt", x_array_validation)
    # np.savetxt("y_array_validation" + startTime + ".txt", y_array_validation)

    # x_array_testing = np.random.uniform(1, 10, 100)
    # y_array_testing = np.random.uniform(1, 10, 100)
    # np.savetxt("x_array_testing.txt", x_array_testing)
    # np.savetxt("y_array_testing.txt", y_array_testing)
    
    #Load data
    dataFolder = ".\\data\\"

    x_array_train = np.loadtxt(dataFolder + "x_array_train.txt")
    y_array_train = np.loadtxt(dataFolder + "y_array_train.txt")

    x_array_validation = np.loadtxt(dataFolder + "x_array_validation.txt")
    y_array_validation = np.loadtxt(dataFolder + "y_array_validation.txt")

    x_array_testing = np.loadtxt(dataFolder + "x_array_testing.txt")
    y_array_testing = np.loadtxt(dataFolder + "y_array_testing.txt")

    func = lambda x, y :  (x / 2) ** 2 + (y ** 3) / (x ** 2)

    ans_train = func(x_array_train, y_array_train)
    ans_validation = func(x_array_validation, y_array_validation)
    ans_testing = func(x_array_testing, y_array_testing)

    activity_func = lambda x, alpha :  1.0 / (1 + np.exp(- alpha * x))
    network = BPNN.network_graph([2, 45 ,1], [None, activity_func, None])   #神經元3層(節點2-45-ㄔㄛ1個)

    # plot data
    # ax3D_train = Axes3D(plt.figure())
    # ax3D_train.scatter(x_array_train, y_array_train, ans_train, marker = "o", color = "blue", depthshade=False)
    # ax3D_train.set_xlabel("x")
    # ax3D_train.set_ylabel("y")
    # ax3D_train.set_zlabel("z")
    # ax3D_train.set_title("Training data")
    # ax3D_train.legend(loc="lower right")

    # ax3D_train = Axes3D(plt.figure())
    # ax3D_train.scatter(x_array_testing, y_array_testing, ans_testing, marker = "o", color = "blue", depthshade=False)
    # ax3D_train.set_xlabel("x")
    # ax3D_train.set_ylabel("y")
    # ax3D_train.set_zlabel("z")
    # ax3D_train.set_title("Testing data")
    # ax3D_train.legend(loc="lower right")

    # plt.show()


    #Training
    epoch = 5000
    error_ave = np.zeros(epoch)
    for i in range(epoch):
        error, outList = network.train([x_array_train, y_array_train], [ans_train])
        error_ave[i] =  np.sqrt(np.mean(np.square(error)))

        print("{0} : {1}".format(i, error_ave[i]))

    plt.figure()
    plt.plot(np.arange(0, len(error_ave), 1), error_ave)
    plt.xlabel("x")
    plt.ylabel("E")

    ax3D_train = Axes3D(plt.figure())
    ax3D_train.scatter(x_array_train, y_array_train, outList, label = "Output", depthshade = False, marker = "x", color = "red")
    ax3D_train.scatter(x_array_train, y_array_train, ans_train, label = "Target", depthshade = False, marker = "o", color = "blue")
    ax3D_train.set_xlabel("x")
    ax3D_train.set_ylabel("y")
    ax3D_train.set_zlabel("z")
    ax3D_train.set_title("Training")

    ax3D_train.legend(loc="lower right")

    #Validation
    error, outList = network.validation([x_array_validation, y_array_validation], [ans_validation])

    # plt.figure()
    # plt.plot(np.arange(0, len(error_ave), 1), error_ave)
    # plt.xlabel("x")
    # plt.ylabel("E")

    ax3D_Validation = Axes3D(plt.figure())
    ax3D_Validation.scatter(x_array_validation, y_array_validation, outList, label = "Output", depthshade = False, marker = "x", color = "red")
    ax3D_Validation.scatter(x_array_validation, y_array_validation, ans_validation, label = "Target", depthshade = False, marker = "o", color = "blue")
    ax3D_Validation.set_xlabel("x")
    ax3D_Validation.set_ylabel("y")
    ax3D_Validation.set_zlabel("z")
    ax3D_Validation.set_title("Validation")

    ax3D_Validation.legend(loc="lower right")

    #Validation
    error, outList = network.validation([x_array_testing, y_array_testing], [ans_testing])

    # plt.figure()
    # plt.plot(np.arange(0, len(error_ave), 1), error_ave)
    # plt.xlabel("x")
    # plt.ylabel("E")

    ax3D_testing = Axes3D(plt.figure())
    ax3D_testing.scatter(x_array_testing, y_array_testing, outList, label = "Output", depthshade = False, marker = "x", color = "red")
    ax3D_testing.scatter(x_array_testing, y_array_testing, ans_testing, label = "Target", depthshade = False, marker = "o", color = "blue")
    ax3D_testing.set_xlabel("x")
    ax3D_testing.set_ylabel("y")
    ax3D_testing.set_zlabel("z")
    ax3D_testing.set_title("Testing")

    ax3D_testing.legend(loc="lower right")

    plt.show()



