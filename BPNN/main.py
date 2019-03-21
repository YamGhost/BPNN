import BPNN
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

datalen = 400

# x_array = numpy.random.uniform(-2, 2, datalen)
# y_array = numpy.random.uniform(-2, 2, datalen)
# x_array = numpy.random.uniform(1, 10, datalen)
# y_array = numpy.random.uniform(1, 10, datalen)
# numpy.savetxt('x_array_train.txt', x_array)
# numpy.savetxt('y_array_train.txt', y_array)
# numpy.savetxt("x_array_train.txt", x_array)
# numpy.savetxt("y_array_train.txt", y_array)
# startTime = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
# numpy.savetxt("x_array_train" + startTime + ".txt", x_array)
# numpy.savetxt("y_array_train" + startTime + ".txt", y_array)

# x_array_validation = numpy.random.uniform(1, 10, 200)
# y_array_validation = numpy.random.uniform(1, 10, 200)
# numpy.savetxt('x_array_validation.txt', x_array_validation)
# numpy.savetxt('y_array_validation.txt', y_array_validation)
x_array_validation = numpy.loadtxt("x_array_validation.txt")
y_array_validation = numpy.loadtxt("y_array_validation.txt")


x_array = numpy.loadtxt('x_array_train.txt')
y_array = numpy.loadtxt('y_array_train.txt')
# x_array = numpy.loadtxt('x_array_train_2.txt')
# y_array = numpy.loadtxt('y_array_train_2.txt')
# x_array = numpy.loadtxt('xlist.txt')
# y_array = numpy.loadtxt('ylist.txt')

func = lambda x, y :  (x / 2) ** 2 + (y ** 3) / (x ** 2)
# func = lambda x, y :  2 * x ** 2 + 0.25 * y **2
ans = func(x_array, y_array)
ans_validation = func(x_array_validation, y_array_validation)

network = BPNN.network_graph([x_array, y_array], [ans], [10])
# network = BPNN.network_graph([x_array, y_array], [ans], [5, 2])

epoch = 500
error_ave = numpy.zeros(epoch)
for i in range(epoch):
    error, outList = network.train()
    error_ave[i] =  numpy.sqrt(numpy.mean(numpy.square(error)))
    # error_ave[i] =  numpy.mean(numpy.square(error))
    # error_ave[i] =  numpy.mean(0.5 * numpy.square(error))
    # error_ave[i] = 0.5 * numpy.mean(numpy.square(error))

    print('{0} : {1}'.format(i, error_ave[i]))


# startTime = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())

plt.figure()
plt.plot(numpy.arange(0, len(error_ave), 1), error_ave)
plt.xlabel('x')
plt.ylabel('E')

ax3D = Axes3D(plt.figure())
ax3D.scatter(x_array, y_array, outList, label = "Output")
ax3D.scatter(x_array, y_array, ans, label = "Target")
ax3D.set_xlabel("x")
ax3D.set_ylabel("y")
ax3D.set_zlabel("z")

ax3D.legend(loc='lower right')


# error, outList = network.validation([x_array_validation, y_array_validation], [ans_validation])

# plt.figure()
# plt.plot(numpy.arange(0, len(error_ave), 1), error_ave)
# plt.xlabel('x')
# plt.ylabel('E')

# ax3D_Validation = Axes3D(plt.figure())
# ax3D_Validation.scatter(x_array_validation, y_array_validation, outList, label = "Output")
# ax3D_Validation.scatter(x_array_validation, x_array_validation, ans_validation, label = "Target")
# ax3D_Validation.set_xlabel("x")
# ax3D_Validation.set_ylabel("y")
# ax3D_Validation.set_zlabel("z")

# ax3D_Validation.legend(loc='lower right')

plt.show()



