import numpy as np 
import sys
import warnings
import matplotlib.pyplot as plt

class network_graph:
    
    def __init__(self, layoutWideNum, layoutActivityFunc, LR = 0.0001):

        self.layoutWideNum = layoutWideNum
        self.layoutActivityFunc = layoutActivityFunc
        self.deepNum = len(layoutWideNum)
        if len(self.layoutActivityFunc) != self.deepNum:
            raise warnings.warn("The WideNum length must same as layoutActivityFunc length!", UserWarning)

        self.wNum = [0] * self.deepNum    

        self.net = [[] for i in range(self.deepNum)]
        self.out = [[] for i in range(self.deepNum)]
        self.w = {}
        self.b = {}
        self.delta = {}

        self.derive_func = lambda f, x, step, *args : (f(x + step, *args) - f(x - step, *args)) / (2.0 * step)
        self.LR = LR

        """
        w_str: Wn(從輸入層向後第n層權重)
        """
        for i in range(1, self.deepNum):

            self.wNum[i] = self.layoutWideNum[i - 1] * self.layoutWideNum[i]

            w_str = "W" + str(i)
    
            w_rand_array = np.matrix(np.random.rand(self.layoutWideNum[i - 1], self.layoutWideNum[i]))
            self.w.update({w_str : w_rand_array})

            b_str = "B" + str(i)

            b_rand_array = np.matrix(np.random.rand(self.layoutWideNum[i]))
            self.b.update({b_str : b_rand_array})

    def validDataSize(self, targetInput, targetOutput):
        
        targetInputShape = np.shape(targetInput)
        targetOutputShape = np.shape(targetOutput)

        # if not isinstance(targetInput, np.matrix) or not isinstance(targetOutput, np.matrix):
        #     raise warnings.warn("The data type must be np matrix!", UserWarning)

        if np.ndim(targetInput) != 2 or np.ndim(targetOutput) != 2:
            raise warnings.warn("The dim must be 2D!", UserWarning)

        if targetInputShape[0] != self.layoutWideNum[0]:
            raise warnings.warn("The input array len is not matched the first wide number!", UserWarning)

        if targetOutputShape[0] != self.layoutWideNum[-1]:                
            raise warnings.warn("The answer array len is not matched the final wide number!", UserWarning)

        for i in range(targetInputShape[0] - 1):
            if len(targetInput[i]) != len(targetInput[i + 1]):
                raise warnings.warn("The input array elements don\"t have same len!", UserWarning)
            
        for i in range(targetOutputShape[0] - 1):
            if len(targetOutput[i]) != len(targetOutput[i + 1]):
                raise warnings.warn("The answer array elements don\"t have same len!", UserWarning)
        
        if targetInputShape[1] != targetOutputShape[1]:
            raise warnings.warn("The input array shape isn\"t same as the answer array shape!", UserWarning)

    def single_input_forward(self, targetInput, targetOutput, pairNum):

        """
        net: Wx + b
        out: f(net)        
        """

        # self.validDataSize(targetInput, targetOutput)
        # for input_pair_num in range(0, np.shape(targetInput)[1]):        
        self.net[0] = np.matrix(targetInput, dtype = np.float64)[:, pairNum].T

        if self.layoutActivityFunc[0] == None:
            self.out[0] = np.matrix(np.copy(self.net[0]))
        else:
            self.out[0] = self.layoutActivityFunc[0](self.net[0], 1)

        for i in range(1, self.deepNum):  
            self.net[i] = np.dot(self.out[i - 1], self.w["W" + str(i)])
            self.net[i] += self.b["B" + str(i)]

            if self.layoutActivityFunc[i] == None:
                self.out[i] = np.matrix(np.copy(self.net[i]))
            else:
                self.out[i] = self.layoutActivityFunc[i](self.net[i], 1)
            # if (i != (self.deepNum - 1)):
            #     self.out[i] = self.activity_func(self.net[i], 1)
            # else:
            #     self.out[i] = np.matrix(np.copy(self.net[i]))

    def single_output_error_update(self, targetInput, targetOutput, pairNum):

        update_data_w = []
        update_data_b = []
        for i in range(1, self.deepNum)[::-1]:

            error_update_level = self.deepNum - i
            if error_update_level == 1:

                delta_array = (self.out[i].T - targetOutput[:, pairNum])
                if self.layoutActivityFunc[i] != None:
                    delta_array = np.multiply(delta_array, self.derive_func(self.layoutActivityFunc[i], self.net[i].T, 0.0001, 1))

                # delta_array = np.multiply(delta_array, self.derive_func(self.activity_func, 1, self.net[i].T, 0.0001)) #(d - y) * f"(net)

                d_str = "D" + str(i)                    
                self.delta.update({d_str : delta_array})

                delta_w = np.dot(delta_array, self.out[i - 1])   #delta * y

                update_data_w.append(self.w["W" + str(i)] - self.LR * np.matrix(delta_w).T)
                update_data_b.append(self.b["B" + str(i)] - self.LR * np.matrix(delta_array))

            else:
                delta_array = self.delta["D" + str(i + 1)]
                delta_array = np.dot(delta_array, self.w["W" + str(i + 1)].T)    #delta * W
                if self.layoutActivityFunc[i] != None:
                    delta_array = np.multiply(delta_array, self.derive_func(self.layoutActivityFunc[i], self.net[i], 0.0001, 1))
                # delta_array = np.multiply(delta_array, self.derive_func(self.activity_func, self.net[i], 0.0001, 1))   #delta * f"(net)

                d_str = "D" + str(i)                    
                self.delta.update({d_str : delta_array})

                delta_w = np.dot(delta_array.T, self.out[i - 1])

                update_data_w.append(self.w["W" + str(i)] - self.LR * np.matrix(delta_w).T)
                update_data_b.append(self.b["B" + str(i)] - self.LR * np.matrix(delta_array))
                        

        for i in range(1, self.deepNum)[::-1]:
            reverseIndex = self.deepNum - i - 1
            self.w.update({"W" + str(i) : np.matrix(update_data_w[reverseIndex])})
            self.b.update({"B" + str(i) : np.matrix(update_data_b[reverseIndex])})

    def train(self, targetInput, targetOutput):

        self.validDataSize(targetInput, targetOutput)        
        targetInput = np.matrix(targetInput)
        targetOutput = np.matrix(targetOutput)

        targetOutputShape = np.shape(targetOutput)
        error = np.zeros(targetOutputShape)
        out = np.zeros(targetOutputShape)
        for i in range(targetOutputShape[1]):

            self.single_input_forward(targetInput, targetOutput, i)
            self.single_output_error_update(targetInput, targetOutput, i)

            error[:, i] = self.out[-1] - targetOutput[:, i]
            out[:, i] = np.copy(self.out[-1])
            # print("{0} : {1}".format(i, error[:, i]))

        # plt.plot(error[0])
        return error, out
    
    def validation(self, targetInput, targetOutput):
        self.validDataSize(targetInput, targetOutput)
        
        targetInput = np.matrix(targetInput)
        targetOutput = np.matrix(targetOutput)

        targetOutputShape = np.shape(targetOutput)
        error = np.zeros(targetOutputShape)
        out = np.zeros(targetOutputShape)
        for i in range(targetOutputShape[1]):

            self.single_input_forward(targetInput, targetOutput, i)
            # self.single_output_error_update(targetInput, targetOutput, i)

            error[:, i] = self.out[-1] - targetOutput[:, i]
            out[:, i] = np.copy(self.out[-1])
            # print("{0} : {1}".format(i, error[:, i]))

        # plt.plot(error[0])
        return error, out