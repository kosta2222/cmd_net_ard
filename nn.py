import numpy as np
import matplotlib.pyplot as plt
import sys
from work_with_arr import calc_vec_as_one_zero, calc_as_hash
import math

import pickle

# 2-слойная сеть связей
TRESHOLD_FUNC_HALF = 0
TRESHOLD_FUNC_HALF_DERIV = 1
SIGMOID = 2
SIGMOID_DERIV = 3
RELU = 4
RELU_DERIV = 5
TAN = 6
TAN_DERIV = 7
INIT_W_MY = 8
INIT_W_RANDOM = 9
LEAKY_RELU = 10
LEAKY_RELU_DERIV = 11


class Two_lay_fcn:

    def __init__(self, in_1=0, out_1=0, out_2=0, act_func_1=SIGMOID, act_func_2=SIGMOID, load_f_name=''):
        # индексы для сериализации в массиве
        self.W1_k = 0
        self.W2_k = 1
        self.B1_k = 2
        self.B2_k = 3
        self.act_func_1_k = 4
        self.act_func_2_k = 5

        if load_f_name != '':  # файл сериализации нам  задан, загружаем
            with open(load_f_name, 'rb') as f:
                net = pickle.load(f)
            self.W1 = net[self.W1_k]
            self.B1 = net[self.B1_k]
            self.W2 = net[self.W2_k]
            self.B2 = net[self.B2_k]
            self.act_func_1=net[self.act_func_1_k]
            self.act_func_2=net[self.act_func_2_k]
            

        else:  # файл сериализации нам не задан

            np.random.seed(1)
            self.W1 = np.random.normal(0, 1, (out_1, in_1))
            self.W2 = np.random.normal(0, 1, (out_2, out_1))

            self.B1 = np.random.random((out_1, 1))
            self.B2 = np.random.random((out_2, 1))

            self.act_func_1 = act_func_1
            self.act_func_2 = act_func_2

    def set_X_Y(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)

    # def sigmoid(self, z, derv=False):
    #     if derv:
    #         return z * (1 - z)
    #     return 1 / (1 + np.exp(-z))

    def operations(self, op, x):

        y = np.zeros(x.shape[0])
        alpha_leaky_relu = 1.7159
        alpha_sigmoid = 2
        alpha_tan = 1.7159
        beta_tan = 2/3

        height = x.shape[0]
        if op == RELU:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = 0
                else:
                    y[row] = x[row][0]

            return np.array(y.T)

        elif op == RELU_DERIV:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = 0
            else:
                y[row] = 1

            return np.array(y.T)
        elif op == TRESHOLD_FUNC_HALF:
            for row in range(height):
                if (x[row][0] > 0.5):
                    y[row] = 1
            else:
                y[row] = 0
            # print('Tr half y', y.T)
            return np.array([y]).T
        elif op == TRESHOLD_FUNC_HALF_DERIV:
            return 1
        elif op == LEAKY_RELU:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = alpha_leaky_relu * x[row][0]
                else:
                    y[row] = x[row][0]
            return np.array(y.T)

        elif op == LEAKY_RELU_DERIV:
            if (x <= 0):
                return alpha_leaky_relu
            else:
                return 1
        elif op == SIGMOID:
            # print('Sigm x', x)
            # print('X reshp', x.T[0])
            y = 1 / (1 + np.exp(-alpha_sigmoid * x))
            # print('Sigm y', y)
            return y
        elif op == SIGMOID_DERIV:
            # y = 1 / (1 + math.exp(-alpha_sigmoid * x))
            return alpha_sigmoid * x * (1 - x)
        elif op == TAN:
            y = alpha_tan * math.tanh(beta_tan * x)
            return y
        elif op == TAN_DERIV:
            return beta_tan * alpha_tan * 4 / ((math.exp(beta_tan * x) + math.exp(-beta_tan * x))**2)
        else:
            print("Op or function does not support ", op)

    def forward(self, X, predict=False):
        X = np.array(X)
        # Getting the training eself.Xample as a column vector.
        inputs = X.reshape(X.shape[0], 1)

        matr_prod_hid = self.W1.dot(inputs) + self.B1
        hid = self.operations(self.act_func_1, matr_prod_hid)
        # print('act func 1', self.act_func_1)
        # print('hid', hid)

        matr_prod_out = self.W2.dot(hid) + self.B2
        out_cn = self.operations(self.act_func_2, matr_prod_out)

        if predict:
            return out_cn
        # return (hid, out_cn, a3)

    def fit(self, learning_rate=0.1, reg_param=0, max_iter=5000):

        hid_err = 0
        out_cn_err = 0

        hid_bias_err = 0
        out_bias_err = 0

        cost = np.zeros((max_iter, 1))
        for i in range(max_iter):

            hid_err = 0
            out_cn_err = 0

            hid_bias_err = 0
            out_bias_err = 0

            gl_err = 0

            m = self.X.shape[0]
            for j in range(m):
                sys.stdout.write(
                    "\rIteration: {} and {} ".format(i + 1, j + 1))

                # Forward Prop.
                input_vec = self.X[j].reshape(self.X[j].shape[0], 1)

                matr_prod_hid = self.W1.dot(input_vec) + self.B1
                hid = self.operations(self.act_func_1, matr_prod_hid)

                matr_prod_out = self.W2.dot(hid) + self.B2
                out_cn = self.operations(self.act_func_2, matr_prod_out)

                # Back prop.
                error_metric = out_cn - \
                    self.Y[j].reshape(self.Y[j].shape[0], 1)
                out_cn_half_err = np.multiply(out_cn - self.Y[j].reshape(self.Y[j].shape[0], 1), \
                                              self.operations(self.act_func_2 + 1, out_cn))

                # print('out_cn_half_err shape', out_cn_half_err.shape)                              
                out_cn_err +=\
                 out_cn_half_err.dot(hid.T)

                hid_half_err = np.multiply((self.W2.T.dot(out_cn_half_err)),
                                           self.operations(self.act_func_1 + 1, hid))
                hid_err += hid_half_err.dot(input_vec.T)

                hid_bias_err += hid_half_err * 1
                out_bias_err += out_cn_half_err * 1  
                gl_err += np.sum(np.square(error_metric))

            self.W1 = self.W1 - learning_rate * \
                (hid_err / m) + ((reg_param / m) * self.W1)

            self.W2 = self.W2 - learning_rate * \
                (out_cn_err / m) + ((reg_param / m) * self.W2)

            self.B1 = self.B1 - learning_rate * (hid_bias_err / m)
            self.B2 = self.B2 - learning_rate * (out_bias_err / m)

            gl_err = gl_err / 2
            cost[i] = gl_err
            sys.stdout.write("error {0}".format(gl_err))
            sys.stdout.flush()  # Updating the teself.Xt.

        # for x in self.X:
        #     print("\n")
        #     print(x)
        #     print(self.forward(x, predict=True))

        plt.plot(range(max_iter), cost)
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.show()

        for single_array_ind in range(m):
         inputs = self.X[single_array_ind]

         output_2_layer = self.forward(inputs, predict=True)

         equal_flag = 0
         out_nc = self.W2.shape[1]
         for row in range(out_nc):
            elem_net = output_2_layer[row]
            elem_train_out = self.Y[single_array_ind][row]
            if elem_net > 0.5:
                elem_net = 1
            else:
                elem_net = 0
            print("elem:", elem_net)
            print("elem tr out:", elem_train_out)
            if elem_net == elem_train_out:
                equal_flag = 1
            else:
                equal_flag = 0
                break
         if equal_flag == 1:
            print('-vecs are equal-')
         else:
            print('-vecs are not equal-')

        print("========")
       

    def to_file(self, f_name):
        ser_arr = [None] * 6

        ser_arr[self.W1_k] = self.W1
        ser_arr[self.B1_k] = self.B1
        ser_arr[self.W2_k] = self.W2
        ser_arr[self.B2_k] = self.B2
        ser_arr[self.act_func_1_k] = self.act_func_1
        ser_arr[self.act_func_2_k] = self.act_func_2

        with open(f_name, 'wb') as f:
            pickle.dump(ser_arr, f)
        print('Weights saved')


def main():
    X = [(1, 0, 0, 0, 1, 0, 0, 0),
         (0, 1, 0, 0, 1, 0, 0, 0),
         (1, 0, 0, 0, 0, 0, 0, 1),
         (0, 1, 0, 0, 0, 0, 0, 1)
         ]

    Y = [(1, 0, 1, 0),
         (1, 0, 0, 0),
         (1, 0, 1, 1),
         (1, 0, 0, 1)]

    net = Two_lay_fcn(8, 3, 4)
    net.set_X_Y(X, Y)
    net.fit(max_iter=1000)
    # net.to_file('wei.my')


FIND_FUNC = 1
ICONST = 2
STOP = 32

# локальная Vm


def vm(program):
    print('Loc vm started.')
    steck = []
    ip = 0
    arg = 0
    op = program[ip]
    while op != STOP:
        if op == ICONST:
            pass
        elif op == FIND_FUNC:
            ip += 1
            arg = program[ip]
            # print("arg", arg)
            if arg == 1:
                print("Включаю лампу-1")
            elif arg == 0:
                print("Выключаю лампу-1")
            elif arg == 3:
                print("Включаю лампу-2")
            elif arg == 2:
                print("Выключаю лампу-2")
        else:
            print("Vm opcode unrecognized")
            return
        ip += 1
        op = program[ip]

    return 0


# Параметры серийного порта для Arduino
SERIAL_PORT = 'COM3'
SERIAL_SPEED = 9600

# утилитная константа выхода
FINISH_SUCS = 0
FINISH_ERR = 1


def test(arg_exc='loc_vm'):
    """
    Будем тестировать локальную тестировочную [заглушку] Vm или Vm на Arduino, зависимость arg
    """
    net = Two_lay_fcn(load_f_name='wei.my')

    ser = None
    if arg_exc == 'ard_vm':
        ser = serial.Serial(SERIAL_PORT, SERIAL_SPEED)

    sents = {'Включи': [1, 0, 0, 0],
             'Выключи': [0, 1, 0, 0],
             'лампу-1': [1, 0, 0, 0],
             'лампу-2': [0, 0, 0, 1]}

    # loger = Logger()
    # nn_params_new = Nn_params()
    # deserialization(nn_params_new, 'wei1.my', loger)

    
    b_c = []
    vec_gathered=[] 
    cmd = ''
    res = -1
    shell_is_running = True
    while shell_is_running:
        print("Запустить r")
        print("Выйти exit")
        cmd = input('->')
        
        # анализируем команды пользователя
        if cmd == 'r':
            """
            Запуск на на определенной Vm числового кода [байт-кода]
            после формирования его с выходов сети связей
            """
            net_res = calc_vec_as_one_zero(
                net.forward(vec_gathered, predict=True))
            vec_gathered=[]    
            print('net res', net_res)    
            op = calc_as_hash(net_res[0:1])
            b_c.append(op)
            arg = calc_as_hash(net_res[2:4])
            b_c.append(arg) 
            b_c.append(STOP)
            print("b_c", b_c)
            if arg_exc == 'loc_vm':
                # тестировочная заглушка
                res = vm(b_c)
                while True:
                    # если машина закончила выполнение, возвращает (0)
                    if res == 0:
                        cmd = input('->')
                        if cmd == 'exit':
                           sys.exit(FINISH_SUCS)
                        b_c = []
                        vec_gathered=[]
                        """  
                        формирование вектора для сети связей
                        """
                        vec = sents.get(cmd)
                        if vec is None:
                          print("Cmd not faund in dictionary")
                          sys.exit(FINISH_ERR)
                        vec_gathered.extend(vec)
                        print("vec gathered", vec_gathered)
                        break
            elif arg_exc == 'ard_vm':
                # arduino исполнитель
                # передаем в байтах
                ser.write(bytes(b_c))
                # Получение строк от vm от arduino
                while True:
                    dev_answ = ser.readline()
                    print(dev_answ)
                    if dev_answ == b'STOP VM\n':
                        cmd = input('->')
                        b_c = []
                        vec_gathered=[]
                        if vec is None:
                              print("Cmd not faund in dictionary")
                              sys.exit(FINISH_ERR)
                        vec_gathered.extend(vec)
                        print("vec gathered", vec_gathered)
                        break
        elif cmd == 'exit':
            sys.exit(FINISH_SUCS)
        else:
            """  
            формирование вектора для сети связей
            """
            vec = sents.get(cmd)
            if vec is None:
                print("Cmd not faund in dictionary")
                sys.exit(FINISH_ERR)
            vec_gathered.extend(vec)
            print("vec gathered", vec_gathered)

    ser.close()

main()
# test()
