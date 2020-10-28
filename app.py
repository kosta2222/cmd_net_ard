"""
Некоторые пояснения к коду в desc.txt - например обозначения переменных
"""
import serial
import math
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from serial_deserial import to_file, deserialization
from work_with_arr import add_2_vecs_comps, make_needed_vec, merge_2_vecs_to_needed_vec, calc_as_hash, make_hashed_elems_matr
from datetime import datetime
import sys

# 2-слойная сеть связей
TRESHOLD_FUNC = 0
TRESHOLD_FUNC_DERIV = 1
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
INIT_W_CONST = 12
INIT_RANDN = 13
SOFTMAX = 14
SOFTMAX_DERIV = 15
MODIF_MSE = 16

ready = False

# Различные операции по числовому коду


def operations(op, x):
    global ready
    alpha_leaky_relu = 1.7159
    alpha_sigmoid = 2
    alpha_tan = 1.7159
    beta_tan = 2/3
    if op == RELU:
        if (x <= 0):
            return 0
        else:
            return x
    elif op == RELU_DERIV:
        if (x <= 0):
            return 0
        else:
            return 1
    elif op == TRESHOLD_FUNC:
        if (x > 0.5):
            return 1
        else:
            return 0
    elif op == TRESHOLD_FUNC_DERIV:
        return 1
    elif op == LEAKY_RELU:
        if (x <= 0):
            return alpha_leaky_relu
        else:
            return 1
    elif op == LEAKY_RELU_DERIV:
        if (x <= 0):
            return alpha_leaky_relu
        else:
            return 1
    elif op == SIGMOID:
        y = 1 / (1 + math.exp(-alpha_sigmoid * x))
        return y
    elif op == SIGMOID_DERIV:
        y = 1 / (1 + math.exp(-alpha_sigmoid * x))
        return alpha_sigmoid * y * (1 - y)
    elif op == INIT_W_MY:
        if ready:
            ready = False
            return -0.567141530112327
        ready = True
        return 0.567141530112327
    elif op == INIT_W_RANDOM:

        return random.random()
    elif op == TAN:
        y = alpha_tan * math.tanh(beta_tan * x)
        return y
    elif op == TAN_DERIV:
        return beta_tan * alpha_tan * 4 / ((math.exp(beta_tan * x) + math.exp(-beta_tan * x))**2)
    elif op == INIT_W_CONST:
        return 0.567141530112327
    elif op == INIT_RANDN:
        return np.random.randn()
    else:
        print("Op or function does not support ", op)


class Dense:
    def __init__(self):  # конструктор
        self.in_ = None  # количество входов слоя
        self.out = None  # количество выходов слоя
        self.matrix = [0] * 10  # матрица весов
        self.cost_signals = [0] * 10  # вектор взвешенного состояния нейронов
        self.act_func = RELU
        self.hidden = [0] * 10  # вектор после функции активации
        self.errors = [0] * 10  # вектор ошибок слоя
        self.with_bias = False
        for row in range(10):  # создаем матрицу весов
            # подготовка матрицы весов,внутренняя матрица
            self.inner_m = list([0] * 10)
            self.matrix[row] = self.inner_m


class Nn_params:
    net = [None] * 2  # Двойной перпецетрон
    for l_ind in range(2):
        net[l_ind] = Dense()
    sp_d = -1  # алокатор для слоев
    nl_count = 0  # количество слоев

    # разные параметры
    loss_func = MODIF_MSE
    alpha_leaky_relu = 0.01
    alpha_sigmoid = 0.42
    alpha_tan = 1.7159
    beta_tan = 2 / 3

################### Функции обучения ######################


def make_hidden(nn_params, layer_ind, inputs: list):
    layer = nn_params.net[layer_ind]
    for row in range(layer.out):
        tmp_v = 0
        for elem in range(layer.in_):
            if layer.with_bias:
                if elem == 0:
                    tmp_v += layer.matrix[row][elem] * 1
                else:
                    tmp_v += layer.matrix[row][elem] * inputs[elem]

            else:
                tmp_v += layer.matrix[row][elem] * inputs[elem]

        layer.cost_signals[row] = tmp_v
        val = operations(layer.act_func, tmp_v)
        layer.hidden[row] = val


def get_hidden(objLay: Dense):
    return objLay.hidden


def feed_forwarding(nn_params: Nn_params, inputs):
    make_hidden(nn_params, 0, inputs)
    j = nn_params.nl_count
    for i in range(1, j):
        inputs = get_hidden(nn_params.net[i - 1])
        make_hidden(nn_params, i, inputs)

    last_layer = nn_params.net[j-1]

    return get_hidden(last_layer)


def cr_lay(nn_params: Nn_params, in_=0, out=0, act_func=None, with_bias=False, init_w=INIT_W_RANDOM):
    nn_params.sp_d += 1
    layer = nn_params.net[nn_params.sp_d]
    layer.in_ = in_
    layer.out = out
    layer.act_func = act_func

    if with_bias:
        layer.with_bias = True
    else:
        layer.with_bias = False

    if with_bias:
        in_ += 1
    for row in range(out):
        for elem in range(in_):
            layer.matrix[row][elem] = operations(
                init_w, 0)

    nn_params.nl_count += 1
    return nn_params


def calc_out_error(nn_params, targets):
    layer = nn_params.net[nn_params.nl_count-1]
    out = layer.out

    for row in range(out):
        tmp_v = (layer.hidden[row] - targets[row]) * operations(
            layer.act_func + 1, layer.hidden[row])
        layer.errors[row] = tmp_v


def calc_hid_error(nn_params, layer_ind: int):
    layer = nn_params.net[layer_ind]
    layer_next = nn_params.net[layer_ind + 1]
    for elem in range(layer.in_):
        summ = 0
        for row in range(layer.out):
            summ += layer_next.matrix[row][elem] * layer_next.errors[row]
        layer.errors[elem] = summ * operations(
            layer.act_func + 1, layer.hidden[elem])


def upd_matrix(nn_params, layer_ind, errors, inputs, lr):
    layer = nn_params.net[layer_ind]
    for row in range(layer.out):
        error = errors[row]
        for elem in range(layer.in_):
            if layer.with_bias:
                if elem == 0:
                    layer.matrix[row][elem] -= lr * \
                        error * 1
                else:
                    layer.matrix[row][elem] -= lr * \
                        error * inputs[elem]
            else:
                layer.matrix[row][elem] -= lr * \
                    error * inputs[elem]


def calc_diff(out_nn, teacher_answ):
    diff = [0] * len(out_nn)
    for row in range(len(teacher_answ)):
        diff[row] = out_nn[row] - teacher_answ[row]
    return diff


def get_err(diff):
    sum = 0
    for row in range(len(diff)):
        sum += diff[row] * diff[row]
    return sum


#############################################

class Logger:
    def __init__(self):
        pass

    def debug(self, s):
        pass

    def info(self, s):
        pass


def plot_gr(_file: str, errors: list, epochs: list) -> None:
    fig: plt.Figure = None
    ax: plt.Axes = None
    fig, ax = plt.subplots()
    ax.plot(epochs, errors,
            label="learning",
            )
    plt.xlabel('Эпоха обучения')
    plt.ylabel('loss')
    ax.legend()
    plt.savefig(_file)
    print("Graphic saved")
    plt.show()


# train_inp = [[1, 1], [2, 0], [0, 2]]
# train_out = [[1], [0], [0]]
# train_inp = [[1, 1, 1, 0], [1, 1, 0, 1]]
# train_out = [[0, 1], [1, 0]]


STOP = 32
FIND_FUNC_vec = [1, 0]

num_0 = [1, 1]  # 3
num_1 = [0, 1]  # как хеш 2
num_2 = [1, 0]  # как хеш 1
num_3 = [0, 0]  # 0


def main():
    epochs = 1000
    l_r = 0.01

    train_inp_0 = [['Включи', (1, 0, 0, 0)],
                   ['Выключи', (0, 1, 0, 0)],
                   ['лампу-1', (1, 0, 0, 0)],
                   ['лампу-2', (0, 0, 0, 1)]
                   ]
    # train_inp = [(train_inp_0[0][1], train_inp_0[2][1]),
    #              (train_inp_0[0][1], train_inp_0[3][1]),
    #              (train_inp_0[1][1], train_inp_0[2][1]),
    #              (train_inp_0[1][1], train_inp_0[2][1])
    #              ]
    train_inp = ((1, 0, 0, 0, 1, 0, 0, 0),
                 (0, 1, 0, 0, 1, 0, 0, 0),
                 (1, 0, 0, 0, 0, 0, 0, 1),
                 (0, 1, 0, 0, 0, 0, 0, 1)
                 )
    print('train_inp', train_inp)

    train_out = ((1, 0, 1, 0),
                 (1, 0, 0, 0),
                 (1, 0, 1, 1),
                 (1, 0, 0, 1))

    # train_out = [['b_c', merge_2_vecs_to_needed_vec(FIND_FUNC_vec, num_2, 4)],
    #              ['b_c', merge_2_vecs_to_needed_vec(FIND_FUNC_vec, num_3, 4)],
    #              ['b_c', merge_2_vecs_to_needed_vec(FIND_FUNC_vec, num_0, 4)],
    #              ['b_c', merge_2_vecs_to_needed_vec(FIND_FUNC_vec, num_1, 4)]
    #              ]
    # новая ссылка
    # train_inp = make_hashed_elems_matr(train_inp)
    print('train_inp', train_inp)
    # print('train_out', train_inp)

    errors_y = []
    epochs_x = []

    # Создаем обьект параметров сети
    nn_params = Nn_params()
    # Создаем слои
    n = cr_lay(nn_params, 8, 3, TRESHOLD_FUNC, False, INIT_W_RANDOM)
    n = cr_lay(nn_params, 3, 4, TRESHOLD_FUNC, False, INIT_W_RANDOM)

    for ep in range(epochs):  # Кол-во повторений для обучения
        gl_e = 0
        for single_array_ind in range(len(train_inp)):
            inputs = train_inp[single_array_ind]
            print("inputs", inputs)
            output = feed_forwarding(nn_params, inputs)

            print("train out",  train_out[single_array_ind])
            e = calc_diff(output, train_out[single_array_ind])

            gl_e += get_err(e)

            calc_out_error(nn_params, train_out[single_array_ind])

            # Обновление весов
            upd_matrix(nn_params, 1, nn_params.net[1].errors, get_hidden(nn_params.net[0]),
                       l_r)

            calc_hid_error(nn_params, 0)

            upd_matrix(nn_params, 0, nn_params.net[0].errors, inputs, l_r)

        gl_e /= 2
        print("error", gl_e)
        print("ep", ep)
        print()

        errors_y.append(gl_e)
        epochs_x.append(ep)

        if gl_e == 0:
            break

    plot_gr('gr.png', errors_y, epochs_x)

    train_inp_n = len(train_inp)
    for single_array_ind in range(train_inp_n):
        inputs = train_inp[single_array_ind]

        output_2_layer = feed_forwarding(nn_params, inputs)

        equal_flag = 0
        out_net_n = nn_params.net[1].out
        for row in range(out_net_n):
            elem_net = output_2_layer[row]
            elem_train_out = train_out[single_array_ind][1][row]
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

    loger = Logger()
    to_file(nn_params, nn_params.net, loger, 'wei1.my')


ICONST = 1
FIND_FUNC = 3
STOP = 32

# локальная Vm


def vm(program):
    print('Loc vm staeted.')
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


def test(arg_exc='loc_vm'):
    """
    Будем тестировать локальную тестировочную [заглушку] Vm или Vm на Arduino, зависимость arg
    """
    ser = None
    if arg_exc == 'ard_vm':
        ser = serial.Serial(SERIAL_PORT, SERIAL_SPEED)

    sents = {'Включи': [1, 0, 0, 0],
             'Выключи': [0, 1, 0, 0],
             'лампу-1': [1, 0, 0, 0],
             'лампу-2': [0, 0, 0, 1]}

    sents = {'Включи лампу-1': [1, 0, 0, 0, 1, 0, 0, 0],
             'Выключи лампу-1': [0, 1, 0, 0, 0, 0, 0, 1],
             'Включи лампу-2': [1, 0, 0, 0, 0, 0, 0, 1],
             'Выключи лампу-2': [0, 1, 0, 0, 0, 0, 0, 1]}        

    loger = Logger()
    nn_params_new = Nn_params()
    deserialization(nn_params_new, 'wei1.my', loger)
    vecs = []
    b_c = []
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
            b_c.append(STOP)
            # print("b_c", b_c)
            if arg_exc == 'loc_vm':
                # тестировочная заглушка
                res = vm(b_c)
                while True:
                    if res == 0:
                        cmd = input('->')
                        b_c = []
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
                        break
        elif cmd == 'exit':
            sys.exit(FINISH_SUCS)
        else:
            """
            Запуск сети связей с вектором команды и формирований числового кода [байт-кода]
            """
            vec = sents.get(cmd)
            if vec is None:
                print("Cmd unricognized")
                sys.exit(1)
            vecs.extend(vec)
            net_res = feed_forwarding(nn_params_new, vecs)
            # vecs.append(net_res)
            print("vec", vec)
            op = calc_as_hash(net_res[0:2])
            b_c.append(op)
            arg = calc_as_hash(net_res[2:5])
            # Добавляем аргумент только если он не равен нулю
            if arg != 0:
                b_c.append(arg)

    ser.close()


# main()
test()
