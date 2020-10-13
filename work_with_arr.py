def copy_vector(src: list, dest: list, n):
    len_ = 0
    for i in range(n):
        dest[i] = src[i]

        len_ += 1
    return len_


def to_ribbon(src, dest, in_, out):
    len_ = 0
    for row in range(out):
        for elem in range(in_):
            dest[row * in_ + elem] = src[row][elem]
            len_ += 1
    return len_


def add_2_vecs_comps(l1, l2, n):
    res = [0] * n
    for elem in range(n):
        res[elem] = l1[elem] + l2[elem]
        if res[elem] > 1:
            res[elem] = res[elem]

    return res


def calc_as_hash(l1):
    koef = 2
    len_l1 = len(l1)
    summ = 0
    for elem in range(len_l1):
        if elem == 0:
           summ+=l1[elem]
        else:
            summ+=l1[elem] * (koef ** elem)
    res = summ % (2 ** len_l1)         
    return res

# print(calc_as_hash([0, 1]))
# print(calc_as_hash([1, 0]))


def make_needed_vec(l1, len_needed_vec):
    """
    Чтобы с нулями вектор заполнить другим с параметра, который меньшией длинной
    """
    vec_d = [0] * len_needed_vec
    len_l1 = len(l1)
    for elem in range(len_l1):
        vec_d[elem] = l1[elem]
    return vec_d


def merge_2_vecs_to_needed_vec(l1, l2, len_vec_d):
    """
    Сливает 2 вектора в вектор нужной длины изначально заполненный нулями и он нужного размера
    """
    vec_d = [0] * len_vec_d
    len_l1 = len(l1)
    len_l2 = len(l2)
    cn = 0
    for elem in range(len_l1):
        vec_d[elem] = l1[elem]
        cn += 1
    for elem in range(len_l2):
        vec_d[cn] = l2[elem]
        cn += 1
    return vec_d

def make_hashed_elems_matr(matrix_s):
  """
  Здесь мы будем хешировать вектора [эти вектора идут как элемены матрицы] в одно число
  """  
  m = len(matrix_s)

  matrix_d = [0] * m
  n1 = 1 # 1 элемент для выходной матрицы
  for row in range(m):
    matrix_d[row] = list([0] * n1)

  # хешируем
  for row in range(m):
    row_s = matrix_s[row]
    sum_hash = calc_as_hash(row_s)
    sum_hash_as_l = [sum_hash]
    matrix_d[row] = sum_hash_as_l

  return matrix_d  

m=[[0, 1],[1, 0],
   [0, 0], [0, 1]]
print(make_hashed_elems_matr(m))    