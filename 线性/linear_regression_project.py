
# coding: utf-8

# # 欢迎来到线性回归项目
# 
# 若项目中的题目有困难没完成也没关系，我们鼓励你带着问题提交项目，评审人会给予你诸多帮助。
# 
# 所有选做题都可以不做，不影响项目通过。如果你做了，那么项目评审会帮你批改，也会因为选做部分做错而判定为不通过。
# 
# 其中非代码题可以提交手写后扫描的 pdf 文件，或使用 Latex 在文档中直接回答。

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[1]:


# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何NumPy以及相关的科学计算库来完成作业


# 本项目要求矩阵统一使用二维列表表示，如下：
A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO 创建一个 4*4 单位矩阵
I = [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]]


# ## 1.2 返回矩阵的行数和列数

# In[2]:


# TODO 返回矩阵的行数和列数
def shape(M):
    c = len(M[0])
    r = len(M)
    return r,c


# In[3]:


# 运行以下代码测试你的 shape 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_shape')


# ## 1.3 每个元素四舍五入到特定小数数位

# In[80]:


# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
B = [[1.0001,2.0001,3.212,5], 
     [2,3,3,5], 
     [1,2,5,1]]
from decimal import *
def matxRound(M, decPts=4):
    for row in range(len(M)):
        for col in range(len(M[0])):
            M[row][col] = round(M[row][col],decPts)


# In[21]:


# 运行以下代码测试你的 matxRound 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxRound')


# ## 1.4 计算矩阵的转置

# In[188]:


# TODO 计算矩阵的转置
def transpose(M):
    #利用*号操作符，可以列出解压成单独的序列 也就是把数组中相同的序列的元素组成一个元组，然后list转换元组
    return [list(col) for col in zip(*M)]


# In[189]:


# 运行以下代码测试你的 transpose 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_transpose')


# ## 1.5 计算矩阵乘法 AB

# In[56]:


# TODO 计算矩阵乘法 AB，如果无法相乘则raise ValueError
def matxMultiply(A, B):
    A_C = len(A[0])
    B_R = len(B)
        
    if A_C == B_R:
        #A * B
        return [[sum(a * b for a, b in zip(a, b)) for b in zip(*B)] for a in A]
    else:
        raise ValueError()


# In[29]:


# 运行以下代码测试你的 matxMultiply 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxMultiply')


# ---
# 
# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[58]:


# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    B =[[A[j][i] for i in range(len(A[0]))] for j in range(len(A))]
    for item ,x in enumerate(B):
        x.append(b[item][0])
    return B


# In[31]:


# 运行以下代码测试你的 augmentMatrix 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_augmentMatrix')


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[77]:


# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1],M[r2] = M[r2],M[r1]


# In[33]:


# 运行以下代码测试你的 swapRows 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_swapRows')


# In[135]:


# TODO r1 <--- r1 * scale
# scale为0是非法输入，要求 raise ValueError
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if scale != 0:
        M[r] = [scale * x for x in M[r]]
    else:
        raise ValueError("scale is zero")


# In[136]:


# 运行以下代码测试你的 scaleRow 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_scaleRow')


# In[236]:


# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    M[r1] = [x + (y *scale) for x,y in zip(M[r1],M[r2])]


# In[39]:


# 运行以下代码测试你的 addScaledRow 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_addScaledRow')


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 2.3.1 算法
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None (你可以在选做问题2.4中证明为什么这里A一定是奇异矩阵)
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# **注：** 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# ### 2.3.2 算法推演
# 
# 为了充分了解Gaussian Jordan消元法的计算流程，请根据Gaussian Jordan消元法，分别手动推演矩阵A为奇异矩阵，矩阵A为非奇异矩阵两种情况。

# In[114]:


# 不要修改这里！
import numpy as np
from helper import generateMatrix,printInMatrixFormat
    
rank = 4
A = generateMatrix(rank,singular=False)
b = np.ones(shape=(rank,1)) # it doesn't matter
printInMatrixFormat(rank,A,b)


# 请按照算法的步骤3，逐步推演增广矩阵的变换。
# 
# 在下面列出每一次循环体执行之后的增广矩阵
# 
# 增广矩阵
# $ Ab = \begin{bmatrix}
#     1 & 0.556 & 0.222 & 0.0 & 0.111\\
#     0 & -0.448 & -2.776 & 3.0 & 0.112\\
#     0 & -13.892 & -8.554 & -9.0 & 0.223\\
#     0 & 0.888 & 5.556 & -6.0 & 0.778\end{bmatrix}$
# 
# $ --> \begin{bmatrix}
#     1 & 0 & -0.12 & -0.36 & 0.12\\
#     0 & 1 & 0.616 & 0.648 & -0.016\\
#     0 & 0 & -2.5 & 3.29 & 0.105\\
#     0 & 0 & 5.009 & -6.575 & 0.792\end{bmatrix}$
#     
# $ --> \begin{bmatrix}
#     1 & 0 & 0 & -0.518 & 0.139\\
#     0 & 1 & 0 & 1.457 & -0.113\\
#     0 & 0 & 1 & -1.313 & 0.158\\
#     0 & 0 & 0 & 0.008 & 0.5\end{bmatrix}$
# 
# $ --> \begin{bmatrix}
#     1 & 0 & 0 & 0 & 32.514\\
#     0 & 1 & 0 & 0 & -91.175\\
#     0 & 0 & 1 & 0 & 82.221\\
#     0 & 0 & 0 & 1 & 62.5\end{bmatrix}$
#     
# 

# ### 2.3.3 实现 Gaussian Jordan 消元法

# In[190]:


# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""
from copy import deepcopy

def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    
    num_equations = len(A)
    num_variables = len(b)
    
    def get_max_in_row(M,row):
        max_value = [abs(M[x][row]) for x in range(len(M)) if x >= row]
        max_value_index = max_value.index(max(max_value)) + row
        if max_value == 0 :
            return None
        else:
            return max_value_index
        
    
    if num_equations == num_variables:
        augmented_matrix = augmentMatrix(A,b)
        copy_augmented_matrix = deepcopy(augmented_matrix)
        
        for  i in range(num_equations):
            #最大值所在的行
            max_values_in_row = get_max_in_row(copy_augmented_matrix,i)
            
            #如果要交换的行正好和自己相等，就不用交换
            if i != max_values_in_row:
                swapRows(copy_augmented_matrix,i,max_values_in_row)
        
            scale_coefficient = Decimal('1')/Decimal(copy_augmented_matrix[i][i])
            copy_augmented_matrix[i] = [round(Decimal(scale_num) * scale_coefficient,decPts) for scale_num in copy_augmented_matrix[i]]
           
            
            if i != num_equations:#最后一行就不用清除下面的数据
                for clear_row in range(i+1,num_equations):
                    beta = - Decimal(copy_augmented_matrix[clear_row][i])
                    copy_augmented_matrix[clear_row] = [round((Decimal(x)+(Decimal(y) * beta)),decPts) 
                                                        for x,y in zip(copy_augmented_matrix[clear_row],
                                                                       copy_augmented_matrix[i])]
            if i != 0:#不是第一行都清除上面的数据
                for clear_above_row in range(i)[::-1]:
                    beta_above = - Decimal(copy_augmented_matrix[clear_above_row][i])
                    copy_augmented_matrix[clear_above_row] = [ round((Decimal(x) + (Decimal(y)*beta_above)),decPts) for x,y in 
                                             zip(copy_augmented_matrix[clear_above_row],copy_augmented_matrix[i])]
        return  [[x[num_equations]] for x in copy_augmented_matrix]
    else:
        return None


# In[191]:


# 运行以下代码测试你的 gj_Solve 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_gj_Solve')


# ## (选做) 2.4 算法正确判断了奇异矩阵：
# 
# 在算法的步骤3 中，如果发现某一列对角线和对角线以下所有元素都为0，那么则断定这个矩阵为奇异矩阵。
# 
# 我们用正式的语言描述这个命题，并证明为真。
# 
# 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# TODO 证明：
# 

# # 3  线性回归

# ## 3.1 随机生成样本点

# In[238]:


# 不要修改这里！
from helper import generatePoints
from matplotlib import pyplot as plt

X,Y = generatePoints(num=100)
## 可视化
plt.xlim((-5,5))
plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b')
plt.show()


# ## 3.2 拟合一条直线
# 
# ### 3.2.1 猜测一条直线

# In[246]:


#TODO 请选择最适合的直线 y = kx + b
k = 4.7

b = 12



# 不要修改这里！
plt.xlim((-5,5))
x_vals = plt.axes().get_xlim()
y_vals = [k*x+b for x in x_vals]
plt.plot(x_vals, y_vals, 'r-')

plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b')

plt.show()


# ### 3.2.2 计算平均平方误差 (MSE)

# 我们要编程计算所选直线的平均平方误差(MSE), 即数据集中每个点到直线的Y方向距离的平方的平均数，表达式如下：
# $$
# MSE = \frac{1}{n}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$

# In[247]:


# TODO 实现以下函数并输出所选直线的MSE

def calculateMSE(X,Y,m,b):
    #求所有点到直线y方向上的距离的平方的平均值（y1-mx1 - b）^2+.....+(yn-mxn - b)^2/n
    all_sum = sum([(y - (m * x) - b)**2 for x,y in zip(X,Y)])/len(X)
    return  all_sum

print(calculateMSE(X,Y,k,b))


# ### 3.2.3 调整参数 $m, b$ 来获得最小的平方平均误差
# 
# 你可以调整3.2.1中的参数 $m,b$ 让蓝点均匀覆盖在红线周围，然后微调 $m, b$ 让MSE最小。

# ## 3.3 (选做) 找到参数 $m, b$ 使得平方平均误差最小
# 
# **这一部分需要简单的微积分知识(  $ (x^2)' = 2x $ )。因为这是一个线性代数项目，所以设为选做。**
# 
# 刚刚我们手动调节参数，尝试找到最小的平方平均误差。下面我们要精确得求解 $m, b$ 使得平方平均误差最小。
# 
# 定义目标函数 $E$ 为
# $$
# E = \frac{1}{2}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 因为 $E = \frac{n}{2}MSE$, 所以 $E$ 取到最小值时，$MSE$ 也取到最小值。要找到 $E$ 的最小值，即要找到 $m, b$ 使得 $E$ 相对于 $m$, $E$ 相对于 $b$ 的偏导数等于0. 
# 
# 因此我们要解下面的方程组。
# 
# $$
# \begin{cases}
# \displaystyle
# \frac{\partial E}{\partial m} =0 \\
# \\
# \displaystyle
# \frac{\partial E}{\partial b} =0 \\
# \end{cases}
# $$
# 
# ### 3.3.1 计算目标函数相对于参数的导数
# 首先我们计算两个式子左边的值
# 
# 证明/计算：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-(y_i - mx_i - b)}
# $$

# TODO 证明:
# 
# $$
# E = \frac{1}{2}\\((y_1 - mx_1 - b)^2 + (y_1 - mx_1 - b)^2 + ... + (y_n - mx_n - b)^2 )
# $$
# 
# $$
# \frac{\partial E}{\partial m} = \frac{1}{2}\\(-2x_1(y_1 - mx_1 - b) -2x_2(y_2 - mx_2 - b) + .... -2x_n(y_n - mx_n - b))
# $$
# 
# $$
# \frac{\partial E}{\partial m} = -x_1(y_1 - mx_1 - b) -x_2(y_2 - mx_2 - b) + .... -x_n(y_n - mx_n - b) = \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)}
# $$
# 
# 
# $$
# \frac{\partial E}{\partial b} = \frac{1}{2}\\(-2(y_1 - mx_1 - b) -2(y_2 - mx_2 - b) + .... -2(y_n - mx_n - b))
# $$
# 
# $$
# \frac{\partial E}{\partial b} = -(y_1 - mx_1 - b) -(y_2 - mx_2 - b) + .... -(y_n - mx_n - b) = \sum_{i=1}^{n}{-(y_i - mx_i - b)}
# $$

# ### 3.3.2 实例推演
# 
# 现在我们有了一个二元二次方程组
# 
# $$
# \begin{cases}
# \displaystyle
# \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} =0 \\
# \\
# \displaystyle
# \sum_{i=1}^{n}{-(y_i - mx_i - b)} =0 \\
# \end{cases}
# $$
# 
# 为了加强理解，我们用一个实际例子演练。
# 
# 我们要用三个点 $(1,1), (2,2), (3,2)$ 来拟合一条直线 y = m*x + b, 请写出
# 
# - 目标函数 $E$, 
# - 二元二次方程组，
# - 并求解最优参数 $m, b$

# TODO 写出目标函数，方程组和最优参数
# 
# 
# $$
# E = \frac{1}{2}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 根据
# 
# $$
# \begin{cases}
# \displaystyle
# \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} =0 \\
# \\
# \displaystyle
# \sum_{i=1}^{n}{-(y_i - mx_i - b)} =0 \\
# \end{cases}
# $$可以求出m和b
# 
# 已知点$(1,1), (2,2), (3,2)$，第三个点带入到两个方程可以求出$m$和$b$
# 
# $$
# \begin{cases}
# \displaystyle
# \\14m + 6b =11 \\
# \\
# \displaystyle
# \\6m + 3b =5 \\
# \end{cases}
# $$
# 
# 所以最优参数$m = 0.5$,$b = 0.67$

# ### 3.3.3 将方程组写成矩阵形式
# 
# 我们的二元二次方程组可以用更简洁的矩阵形式表达，将方程组写成矩阵形式更有利于我们使用 Gaussian Jordan 消元法求解。
# 
# 请证明 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = X^TXh - X^TY
# $$
# 
# 其中向量 $Y$, 矩阵 $X$ 和 向量 $h$ 分别为 :
# $$
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 证明:
# 
# 因为
# $$
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix}
# ,
# X^T = \begin{bmatrix}
#     x_1 & x_2 & ....& x_n \\
#     1 & 1 & ....& 1\\
# \end{bmatrix}
# $$
# 
# $$
# X^TXh = \begin{bmatrix}
#     \sum_{i=1}^{n}{x_i^2} + \sum_{i=1}^{n}{bx_i} \\
#     \sum_{i=1}^{n}{mx_i} + nb \\
# \end{bmatrix}
# $$
# 
# $$
# X^TY = \begin{bmatrix}
#     \sum_{i=1}^{n}{x_iy_i} \\
#     \sum_{i=1}^{n}{y_i} \\
# \end{bmatrix}
# $$
# 
# $$
# X^TXh - X^TY = \begin{bmatrix}
#     \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} \\
#     \sum_{i=1}^{n}{-(y_i - mx_i - b)} \\
# \end{bmatrix}
# $$
# 
# 由上面的结论知道：
# $$
# \begin{cases}
# \displaystyle
# \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} =0 \\
# \\
# \displaystyle
# \sum_{i=1}^{n}{-(y_i - mx_i - b)} =0 \\
# \end{cases}
# $$
# 
# 所以可以得出
# 
# 
# $$
# X^TXh - X^TY = 
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}
# $$
# 

# 至此我们知道，通过求解方程 $X^TXh = X^TY$ 来找到最优参数。这个方程十分重要，他有一个名字叫做 **Normal Equation**，也有直观的几何意义。你可以在 [子空间投影](http://open.163.com/movie/2010/11/J/U/M6V0BQC4M_M6V2AJLJU.html) 和 [投影矩阵与最小二乘](http://open.163.com/movie/2010/11/P/U/M6V0BQC4M_M6V2AOJPU.html) 看到更多关于这个方程的内容。

# ### 3.4 求解 $X^TXh = X^TY$ 
# 
# 在3.3 中，我们知道线性回归问题等价于求解 $X^TXh = X^TY$ (如果你选择不做3.3，就勇敢的相信吧，哈哈)

# In[272]:


# TODO 实现线性回归
'''
参数：X, Y
返回：m，b
'''
def linearRegression(X,Y):
    #先构建一个X矩阵
    augmented_martix = [[x,1] for x in X]
    new_Y = [[y] for y in Y]
    transpose_augmented = transpose(augmented_martix)
    new_martix = matxMultiply(transpose_augmented,augmented_martix)
    constant_martix =  matxMultiply(transpose_augmented,new_Y)

    a = gj_Solve(new_martix,constant_martix)
    return  a[0],a[1]
m,b = linearRegression(X,Y)
print(m,b)


# 你求得的回归结果是什么？它足够好了吗？请使用运行以下代码将它画出来。
# 
# 请老师指导一下，下面的y不知道是哪一个？需要自己根据m和b来构造么？

# In[274]:


# 请不要修改下面的代码
x1,x2 = -5,5
y1,y2 = x1*m+b, x2*m+b

plt.xlim((-5,5))
plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(x,y,c='b')
plt.plot((x1,x2),(y1,y2),'r')
plt.text(1,2,'y = {m}x + {b}'.format(m=m,b=b))
plt.show()

