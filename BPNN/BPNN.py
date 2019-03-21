import numpy
import sys
import warnings
import matplotlib.pyplot as plt

class network_graph:
    
    def __init__(self, dataInput, ans, hiddenWide, activityFunc = None):

        self.input = numpy.matrix(dataInput)        
        self.ans = numpy.matrix(ans)

        # self.deepNum = len(hiddenWide)
        self.deepNum = len(hiddenWide) + 2

        # self.wideNum = hiddenWide
        self.wideNum = self.input.shape[0]
        self.wideNum = numpy.hstack((self.wideNum, hiddenWide)) 
        self.wideNum = numpy.hstack((self.wideNum, self.ans.shape[0]))

        self.wNum = [0] * self.deepNum

        # try:

        if self.input.shape[0] != self.wideNum[0]:
            raise warnings.warn('The input array len is not matched the first wide number!', UserWarning)

        if self.ans.shape[0] != self.wideNum[-1]:                
            raise warnings.warn('The answer array len is not matched the final wide number!', UserWarning)

        for i in range(self.input.shape[0] - 1):
            if len(self.input[i]) != len(self.input[i + 1]):
                raise warnings.warn('The input array elements don\'t have same len!', UserWarning)
            
        for i in range(self.ans.shape[0] - 1):
            if len(self.ans[i]) != len(self.ans[i + 1]):
                raise warnings.warn('The answer array elements don\'t have same len!', UserWarning)
        
        if self.input.shape[1] != self.ans.shape[1]:
            raise warnings.warn('The input array shape isn\'t same as the answer array shape!', UserWarning)    

        self.net = [[] for i in range(self.deepNum)]
        self.out = [[] for i in range(self.deepNum)]
        self.w = {}
        self.b = {}
        self.delta = {}

        self.activity_func = lambda x, alpha :  1.0 / (1 + numpy.exp(- alpha * x))

        self.derive_func = lambda f, step, alpha, x : (f(x + step, alpha) - f(x - step, alpha)) / (2.0 * step)

        self.LR = 0.001

        """
        w_str: Wn(從輸入層向後第n層權重)
        """
        for i in range(1, self.deepNum):

            self.wNum[i] = self.wideNum[i - 1] * self.wideNum[i]

            w_str = 'W' + str(i)
    
            w_rand_array = numpy.matrix(numpy.random.rand(self.wideNum[i - 1], self.wideNum[i]))
            self.w.update({w_str : w_rand_array})

            b_str = 'B' + str(i)

            b_rand_array = numpy.matrix(numpy.random.rand(self.wideNum[i]))
            self.b.update({b_str : b_rand_array})

       
    def single_input_forward(self, input_pair_num):

        """
        net: Wx + b
        out: f(net)
        """
        self.net[0] = numpy.matrix(self.input, dtype = numpy.float64)[:, input_pair_num].T
        self.out[0] = numpy.matrix(numpy.copy(self.net[0]))

        for i in range(1, self.deepNum):  
            self.net[i] = numpy.dot(self.out[i - 1], self.w['W' + str(i)])
            self.net[i] += self.b['B' + str(i)]

            if (i != (self.deepNum - 1)):
                self.out[i] = self.activity_func(self.net[i], 1)
            else:
                self.out[i] = numpy.matrix(numpy.copy(self.net[i]))

    def single_output_error_update(self, output_pair_num):
        
        update_data_w = []
        update_data_b = []
        for i in range(1, self.deepNum)[::-1]:

            try:
                error_update_level = self.deepNum - i
                if error_update_level == 1:

                    delta_array = (self.out[i].T - self.ans[:, output_pair_num])
                    # delta_array = numpy.multiply(delta_array, self.derive_func(self.activity_func, 1, self.net[i].T, 0.0001)) #(d - y) * f'(net)

                    d_str = 'D' + str(i)                    
                    self.delta.update({d_str : delta_array})

                    delta_w = numpy.dot(delta_array, self.out[i - 1])   #delta * y

                    update_data_w.append(self.w['W' + str(i)] - self.LR * numpy.matrix(delta_w).T)
                    update_data_b.append(self.b['B' + str(i)] - self.LR * numpy.matrix(delta_array))

                    # update_data_w = self.w['W' + str(i)] - self.LR * numpy.matrix(delta_w).T
                    # update_data_b = self.b['B' + str(i)] - self.LR * numpy.matrix(delta_array)

                    # self.w.update({'W' + str(i) : numpy.matrix(update_data_w)})
                    # self.b.update({'B' + str(i) : numpy.matrix(update_data_b)})
                # elif error_update_level == 2:
                else:
                    delta_array = self.delta['D' + str(i + 1)]
                    delta_array = numpy.dot(delta_array, self.w['W' + str(i + 1)].T)    #delta * W
                    delta_array = numpy.multiply(delta_array, self.derive_func(self.activity_func, 1, self.net[i], 0.0001))   #delta * f'(net)

                    d_str = 'D' + str(i)                    
                    self.delta.update({d_str : delta_array})

                    delta_w = numpy.dot(delta_array.T, self.out[i - 1])

                    update_data_w.append(self.w['W' + str(i)] - self.LR * numpy.matrix(delta_w).T)
                    update_data_b.append(self.b['B' + str(i)] - self.LR * numpy.matrix(delta_array))

                    # update_data_w = self.w['W' + str(i)] - self.LR * numpy.matrix(delta_w).T
                    # update_data_b = self.b['B' + str(i)] - self.LR * numpy.matrix(delta_array)

                    # self.w.update({'W' + str(i) : numpy.matrix(update_data_w)})
                    # self.b.update({'B' + str(i) : numpy.matrix(update_data_b)})

                # else:
                #     raise warnings.warn('Error update function no define!')
                    
            except Exception as ex:
                sys.exit(False)

        for i in range(1, self.deepNum)[::-1]:
            reverseIndex = self.deepNum - i - 1
            self.w.update({'W' + str(i) : numpy.matrix(update_data_w[reverseIndex])})
            self.b.update({'B' + str(i) : numpy.matrix(update_data_b[reverseIndex])})

    def train(self):
        error = numpy.zeros(self.ans.shape)
        out = numpy.zeros(self.ans.shape)
        for i in range(self.input.shape[1]):

            self.single_input_forward(i)
            self.single_output_error_update(i)
            # error[:, i] = 0.5 * numpy.square(self.ans[:, i] - self.out[-1])
            error[:, i] = self.out[-1] - self.ans[:, i]
            out[:, i] = numpy.copy(self.out[-1])
            # print('{0} : {1}'.format(i, error[:, i]))

        # plt.plot(error[0])
        # plt.show()
        return error, out
    
    def validation(self, targetInput, targetOutput):
        inputShape = numpy.shape(targetInput)
        error = numpy.zeros(inputShape)
        out = numpy.zeros(inputShape)
        for i in range(inputShape[1]):

            self.single_input_forward(i)
            # self.single_output_error_update(i)
            error[:, i] = targetOutput[:, i] - self.out[-1]
            out[:, i] = numpy.copy(self.out[-1])

            # print('{0} : {1}'.format(i, error[:, i]))

        # plt.plot(error[0])
        # plt.show()
        return error, out