
import time

def find_all_comb(h,m,n,cur,res):
    if(m == h):
        res.append(cur.copy())
        
        return
    for i in range(n):
        cur.append(i)
        find_all_comb(h,m+1,n,cur,res)
        cur.pop()
    return 
 
def DSFusion(m_1):
    m_0 = [1-p for p in m_1]
    length = len(m_1)
    
    tmp0 = 1
    tmp1 = 1
    for i in range(length):
        tmp0 *= m_0[i]
        tmp1 *= m_1[i]
    k = 1 - (tmp0 + tmp1)
    p_1 = tmp1 / (1- k)
    p_0 = tmp0 / (1- k)

    return k, p_1

def DSFusion2(m_1):
    m_0 = [1-p for p in m_1]
    length = len(m_1)
    
    k = 0
    p0 = 0
    p1 = 0
    res = list()
    find_all_comb(length,0,2,[],res)
    for cur in res:
        if(check0(cur)):
            tmp0 = 1
            for i in range(length):
                tmp0 *= m_0[i]
            p0 += tmp0
        elif(check1(cur)):
            tmp1 = 1
            for i in range(length):
                tmp1 *= m_1[i]
            p1 += tmp1
        else:
            kk = 1
            for i in range(length):
                if(cur[i] == 0):
                    kk *= (1 - m_1[i])
                else:
                    kk *= m_1[i]
            k += kk
    p1 = p1 / (1-k)
    return k, p1

def check0(alist):
    for a in alist:
        if(a != 0):
            return False
    return True

def check1(alist):
    for a in alist:
        if(a != 1):
            return False
    return True

def check2(alist):
    for a in alist:
        if(a == 2):
            return True
    return False

def set_and(alist):
    res = 3
    for a in alist:
        if(a == 0):
            res &= 1
        elif(a == 1):
            res &= 2
        elif(a == 2):
            res &= 3
    return res

# fixed discount factor
def DSFusionDiscount(prediction_1):
    alpha = 0.5
    length = len(m_1)
    m_1 = prediction_1.copy()
    if(length == 1):
        return m_1[0]
    m_0 = [1-p for p in m_1]
    
    #for i in range(length):
    #    m_1[i] = m_1[i] * alpha
    #    m_0[i] = m_0[i] * alpha
    #    alpha += 0.025
    m_1 = [p * 0.8 for p in m_1]
    m_0 = [p * 0.8 for p in m_0]
    m_all = []
    for i in range(length):
        m_all.append(1 - m_0[i] - m_1[i])

    k = 0
    p0 = 0
    p1 = 0
    p2 = 0
    res = list()
    find_all_comb(length,0,3,[],res)
    
    for cur in res:
        intersec = set_and(cur)
        if(intersec == 1):
            tmp0 = 1
            for i in range(length):
                if(cur[i] == 0):
                    tmp0 *= m_0[i]
                elif(cur[i] == 2):
                    tmp0 *= m_all[i]
            p0 += tmp0
        elif(intersec == 2):
            tmp1 = 1
            for i in range(length):
                if(cur[i] == 1):
                    tmp1 *= m_1[i]
                elif(cur[i] == 2):
                    tmp1 *= m_all[i]
            p1 += tmp1
        elif(intersec == 3):
            tmp2 = 1
            for i in range(length):
                tmp2 *= m_all[i]
            p2 += tmp2
        else:
            kk = 1
            for i in range(length):
                if(cur[i] == 0):
                    kk *= m_0[i]
                elif(cur[i] == 1):
                    kk *= m_1[i]
                elif(cur[i] == 2):
                    kk *= m_all[i]
            k += kk
    
    p0 = p0 / (1-k)
    p1 = p1 / (1-k)
    p2 = p2 / (1-k)
    
    return p1

# varying discount factor
def DSFusionDiscount2(prediction_1):
    m_1 = prediction_1.copy()
    alpha = 0.8
    length = len(m_1)
    if(length == 1):
        return m_1[0]
    m_0 = [1-p for p in m_1]
    for i in range(length):
        m_1[i] = m_1[i] * alpha
        m_0[i] = m_0[i] * alpha
        alpha += 0.02
    #m_1 = [p * 0.8 for p in m_1]
    #m_0 = [p * 0.8 for p in m_0]
    m_all = []
    for i in range(length):
        m_all.append(1 - m_0[i] - m_1[i])

    k = 0
    p0 = 0
    p1 = 0
    p2 = 0
    res = list()
    find_all_comb(length,0,3,[],res)
    
    for cur in res:
        intersec = set_and(cur)
        if(intersec == 1):
            tmp0 = 1
            for i in range(length):
                if(cur[i] == 0):
                    tmp0 *= m_0[i]
                elif(cur[i] == 2):
                    tmp0 *= m_all[i]
            p0 += tmp0
        elif(intersec == 2):
            tmp1 = 1
            for i in range(length):
                if(cur[i] == 1):
                    tmp1 *= m_1[i]
                elif(cur[i] == 2):
                    tmp1 *= m_all[i]
            p1 += tmp1
        elif(intersec == 3):
            tmp2 = 1
            for i in range(length):
                tmp2 *= m_all[i]
            p2 += tmp2
        else:
            kk = 1
            for i in range(length):
                if(cur[i] == 0):
                    kk *= m_0[i]
                elif(cur[i] == 1):
                    kk *= m_1[i]
                elif(cur[i] == 2):
                    kk *= m_all[i]
            k += kk
    
    p0 = p0 / (1-k)
    p1 = p1 / (1-k)
    p2 = p2 / (1-k)
    
    return p1


    



m = [0.6, 0.55, 0.2, 0.5, 0.3, 0.9, 0.7]

starttime = time.time()
res = list()
find_all_comb(10,0,3,[],res)
endtime = time.time()
#print(i,j)
print(endtime-starttime)
#print(DSFusionDiscount(m))
#print(set_and([0,2,0]))