
# coding: utf-8

# In[36]:



import pprint


# In[66]:


# 1.1
F = [[1,2,3,4],
     [1,1,1,1],
     [9,8,8,1]]

E = [[1,2,3,4],
     [5,6,7,8],
     [2,3,4,5],
     [1,3,4,5]]

A= [[1,2,3,4],
     [5,6,8,9],
     [9,8,8,4],
     [6,4,2,4]]

c = [1,2,3,4]

b = [1,2,3]


# In[44]:


# 1.2返回矩阵的行和列
def shape(M):
    c = len(M[0])
    r = 0
    for x in M:
        r += 1
    return r,c


# In[45]:


# 1.3元素四舍五入到特定的小数位
def matxRound(M, decPts=4): 
        print [[round(y,decPts) for y in x ] for x in M]


# In[46]:


# 1.4计算矩阵的转置
def transpose(M):
    return [[row[col] for row in M] for col in range(len(M[0]))]


# In[47]:


# 1.5计算矩阵乘法 AB，如果无法相乘则返回None
def matxMultiply(A,B):
    #查看A的列数和B的行数是否相等
    A_C = len(A[0])
    B_R = 0
    for x in B:
        B_R += 1
        
    if A_C == B_R:
        #A * B
        return [[sum(a * b for a, b in zip(a, b)) for b in zip(*B)] for a in A]
    else:
        return None


# In[48]:


# 2.1 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    R_A = 0
    for x in A:
        R_A += 1
    R_B = len(b)
    print (R_A,R_B)
    if R_A == R_B:
        for item ,x in enumerate(A):
            x.append(b[item])
        return A
    else:
        return None
        


# In[62]:


# 2.2 初等行变换

# 交换两行
# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1],M[r2] = M[r2],M[r1]

# TODO r1 <--- r1 * scale， scale!=0
def scaleRow(M, r, scale):
    M[r] = [round(scale*x,4) for x in M[r]]

# TODO r1 <--- r1 + r2*scale
def addScaledRow(M, r1, r2, scale):
    c = 0
    for x in M:
        c += 1
    if r1 > c or r2 > c:
        return "当前选择的行超出了矩阵的最大列"

    M[r2] = [scale*x for x in M[r2]]
    M[r1] = [x + y for x,y in zip(M[r1],M[r2])]


# In[67]:


def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    #1.1检查A，b的行数是否相等
    R_A,C_A = shape(A)
    R_B = len(b)
    if R_A == R_B:
        #1.2构造增广矩阵Ab,A_b就会A和b的增广矩阵
        augmentMatrix(A,b)
        num_equations = R_A
        num_variables = C_A
        j = 1 
        z = j
        for i in range(num_variables - 1):
            max_num = get_martix_value_position(A,i,num_equations)
            print( "行 = " + str(j) +"---" + "列=" + str(i) )
            print("最大值所在的行 = " + str(max_num))
            if max != None:
                swapRows(A,i,max_num)
            scale = 1.0/A[i][i]
            #把对角元素化为1   
            scaleRow(A,i,scale)
            print A[i]
            while j < num_equations:
                clear_coefficient_row(A,i,j)
                if j == num_equations - 1:
                    j = z
                    break
                j += 1
            j += 1 
            z = j
       
        pprint.pprint(A)
        #返回Ab中的最后一列
        LAST_LIST = get_augmentMatrix_last_row(A)
        pprint.pprint(LAST_LIST)
        return LAST_LIST
        
    else:
        print ("A and c not match")
        return None

#这里做的事情就是从对角元素开始往下取元素，然后得到绝对值所在的最大一行的位置，和第一行交换
#如果这里面的绝对值最大是0，就说明是奇异矩阵
def get_martix_value_position(A,row,col):
    MAX_LIST = {}
    for y in range(row,col):
        #键值对的形式存储
        MAX_LIST[y] = A[y][row]
        
    max_num= max(zip(MAX_LIST.values(),MAX_LIST.keys()))[0]
    key_index  = max(zip(MAX_LIST.values(),MAX_LIST.keys()))[1]
    print MAX_LIST
    print("最大值所在的位置是：" + str(key_index) + "最大值是：" + str(max_num) )
    #取出最大的数，之后在取出矩阵的数，比较如果相等，就把矩阵中的数所在的角标返回去
    if max_num == 0 :
        print ("data is all zero")
        return None
    else:
        return key_index
    
def clear_coefficient_row(A,col,row,decPts=4):
    num_scale = -1.0/A[row][col]
    print("缩放值 = " + str(num_scale) + "缩放之的原数 = " + str(A[row][col]) + "缩放值所在的行和列=" + str(row) + " or "+ str(col))
    print("缩放前 A[" + str(row) + "] =" + str(A[row]))
    A[row] = [round(num_scale * x,decPts) for x in A[row]] 
    print("缩放后 A[" + str(row) + "] =" + str(A[row]))
    print("与之相加的数是" + str(A[col]))
    A[row] = [x + y for x,y in zip(A[col],A[row])]
    print("清除之后 A[" + str(row) + "] =" + str(A[row]))
                  
                  
def get_augmentMatrix_last_row(A):
    LAST_LIST = []
    for x in A:
       LAST_LIST.append(x[len(A)]) 
    return LAST_LIST


# #2.4证明
# “”“
# 证明 I是单位矩阵 Z是零矩阵 Y的第一列是零 X是任意的矩阵
# 
#     A是方阵，假设A是nxn的方阵，又因为I是单位矩阵 Z是零矩阵 Y的第一列是零 X是任意的矩阵
#     所以可以认为当A是非奇异方阵的时候的秩是n 
#     
#     此时的r(A) = r(I) + r(Y) 等于上下分块矩阵的秩的和，上面的矩阵是I和X I是单位阵所以上边的秩可以
#     认为就是单位矩阵的秩r(I)   下部分中Z是零矩阵 Y的第一列是零向量所以下边的秩可以认为是r(Y)
#     
#     此时假设r(I) = m ,那么Y中就有n - m那么多列，因为Y中最后一列是零向量，====》r(Y) < n - m
#     所以此时的总的分块矩阵的秩r(I) + r(Y) < n，也就是r(A) < n,所以A是奇异矩阵，此时的A的秩不是满秩
# “”“
# 
# 
# 

# In[ ]:


#3中问题没有学到，请导师指点，什么是损失精度和线性回归

