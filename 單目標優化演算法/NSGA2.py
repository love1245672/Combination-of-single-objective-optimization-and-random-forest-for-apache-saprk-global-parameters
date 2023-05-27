# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan
# Supervisor: Prof. Manoj Kumar Tiwari
"""
优化目标：
    min（f1(x), f2(x))
        f1(x) = -x^2
        f2(X) = -(x-2)^2
    s.t x~[-55, 55]
pop_size = 20
max_gen  =  921

"""
#Importing required modules
import math
import random
import matplotlib.pyplot as plt
import random
from xgboost import XGBRegressor
import pandas as pd
from sklearn.svm import SVR
import joblib
import warnings
import sys
warnings.filterwarnings("ignore", category=FutureWarning)

rf_model_km = XGBRegressor()
rf_model_nw = XGBRegressor()
rf_model_pg = XGBRegressor()
rf_model_ts = XGBRegressor()
rf_model_wc = XGBRegressor()

rf_model_km = joblib.load('/home/love1245672/桌面/hibench_report/model_store/rf/rf_model_ScalaSparkKmeans.joblib')
rf_model_nw = joblib.load('/home/love1245672/桌面/hibench_report/model_store/rf/rf_model_ScalaSparkNWeight.joblib')
rf_model_pg = joblib.load('/home/love1245672/桌面/hibench_report/model_store/rf/rf_model_ScalaSparkPagerank.joblib')
rf_model_ts = joblib.load('/home/love1245672/桌面/hibench_report/model_store/rf/rf_model_ScalaSparkTerasort.joblib')
rf_model_wc = joblib.load('/home/love1245672/桌面/hibench_report/model_store/rf/rf_model_ScalaSparkWordcount.joblib')

cols = ['spark.driver.memory','spark.driver.cores','spark.executor.instances','spark.reducer.maxSizeInFlight','spark.shuffle.file.buffer','spark.shuffle.sort.bypassMergeThreshold','spark.memory.fraction','spark.memory.storageFraction','spark.shuffle.memoryFraction', 'spark.storage.memoryFraction','spark.storage.unrollFraction', 'spark.default.parallelism','spark.broadcast.blockSize', 'spark.storage.memoryMapThreshold','spark.io.compression.codec', 'spark.io.compression.lz4.blockSize','spark.io.compression.snappy.blockSize', 'spark.kryoserializer.buffer.max','spark.kryoserializer.buffer','spark.serializer']
#First function to optimize
def score_formula(x,max_dur,min_dur):
    score = ((x - min_dur) / (max_dur - min_dur))*100#100比分數 越小越好
    score = max(0.0, min(score, 100.0))#將分數限制在 0~100之間WW
    return score

def function1(x):
    df_chromosome = pd.DataFrame(x,columns=cols)
    km_score = score_formula(rf_model_km.predict(df_chromosome)[0],204.188,67.699) * 1#權重
    nw_score = score_formula(rf_model_nw.predict(df_chromosome)[0],270.744,90.472) * 1
    return 100 - (km_score+nw_score)/2

#Second function to optimize
def function2(x):
    df_chromosome = pd.DataFrame(x,columns=cols)
    ts_score = score_formula(rf_model_ts.predict(df_chromosome)[0],117.427,64.393) * 1
    wc_score = score_formula(rf_model_wc.predict(df_chromosome)[0],101.12,61.673) * 1
    pg_score = score_formula(rf_model_pg.predict(df_chromosome)[0],253.533,81.831) * 1
    return 100 - (ts_score+wc_score+pg_score)/3

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

# Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or\
                    (values1[p] >= values1[q] and values2[p] > values2[q]) or\
                    (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or\
                    (values1[q] >= values1[p] and values2[q] > values2[p]) or\
                    (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

#Function to carry out the crossover
def crossover(parent1,parent2):
    #處理數值類別的 Feature
    child = {}
    for feature in parent1.keys():
        if (feature != "spark.executor.instances") & (feature != "spark.io.compression.codec") & (feature != "spark.serializer") & (feature != "fitness"):
            if type(parent1[feature][0]) == float:
                child[feature] = [random.choice([parent1[feature][0],parent2[feature][0]])]
                #child[feature] = [round(((parent1[feature][0] + parent2[feature][0]) / 2),1)]
            else:
                child[feature] = [random.choice([parent1[feature][0],parent2[feature][0]])]
                #child[feature] = [((parent1[feature][0] + parent2[feature][0]) // 2)]
    # 處理分類類別的 Feature
    child["spark.executor.instances"] = [random.choice([parent1["spark.executor.instances"][0],parent2["spark.executor.instances"][0]])]
    child["spark.io.compression.codec"] = [random.choice([parent1["spark.io.compression.codec"][0],parent2["spark.io.compression.codec"][0]])]
    child["spark.serializer"] = [random.choice([parent1["spark.serializer"][0],parent2["spark.serializer"][0]])]
    return mutation(child)

#Function to carry out the mutation operator
def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob <1:
        solution =  create_inital_people()
    return solution

def create_inital_people():
    instance_choice = (2,4,8)
    codec_choice = (1,2,3)#snappy:1,  lz4:2,  lzf:3
    serializer_choice = (1,2)# JAVA:1 , Kryo:2
    people = {'spark.driver.memory':[random.randint(4,16)],'spark.driver.cores':[random.randint(1,8)],'spark.executor.instances':[random.choice(instance_choice)],'spark.reducer.maxSizeInFlight':[random.randint(2,128)],
            'spark.shuffle.file.buffer':[random.randint(2,128)],'spark.shuffle.sort.bypassMergeThreshold':[random.randint(50,800)],
            'spark.memory.fraction':[round(random.uniform(0.1, 0.9), 2)],'spark.memory.storageFraction':[round(random.uniform(0.1, 0.9), 2)],
            'spark.shuffle.memoryFraction':[round(random.uniform(0.1, 0.9), 2)], 'spark.storage.memoryFraction':[round(random.uniform(0.1, 0.9), 2)],
            'spark.storage.unrollFraction':[round(random.uniform(0.1, 0.9), 2)], 'spark.default.parallelism':[random.randint(2,16)],
            'spark.broadcast.blockSize':[random.randint(2,128)], 'spark.storage.memoryMapThreshold':[random.randint(2,500)],
            'spark.io.compression.codec':[random.choice(codec_choice)], 'spark.io.compression.lz4.blockSize':[random.randint(2,128)],
            'spark.io.compression.snappy.blockSize':[random.randint(2,128)], 'spark.kryoserializer.buffer.max':[random.randint(4,128)],
            'spark.kryoserializer.buffer':[random.randint(2,128)],'spark.serializer':[random.choice(serializer_choice)]}
    return people

#Main program starts here
pop_size = 1500
max_gen = 50

#Initialization
min_x=-55
max_x=55
# 随机生成初始种群
solution=[create_inital_people() for i in range(0,pop_size)]
gen_no=0
while(gen_no<max_gen):

    # 自适应度计算
    function1_values = [function1(solution[i])for i in range(0,pop_size)]
    function2_values = [function2(solution[i])for i in range(0,pop_size)]
    # pareto等级
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    #print(function1_values)
    #print(non_dominated_sorted_solution)
    print("The best front for Generation number ",gen_no, " is")
    #for valuez in non_dominated_sorted_solution[0]:
    #    print(round(solution[valuez],3),end=" ")
    #print("\n")
    # 拥挤度距离计算
    crowding_distance_values = []
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))

    solution2 = solution[:] # P+Q
    #Generating offsprings
    while(len(solution2)!=2*pop_size):
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)
        # 交叉变异
        solution2.append(crossover(solution[a1],solution[b1]))
    # 计算 P+Q种群的适应度
    function1_values2 = [function1(solution2[i])for i in range(0,2*pop_size)]
    function2_values2 = [function2(solution2[i])for i in range(0,2*pop_size)]
    # 非支配排序
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
    # 拥挤度距离计算
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    # 得到下一代种群P1
    new_solution = []   # index
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i]) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]
    if gen_no == max_gen -1:
        print(non_dominated_sorted_solution)
        f = open("best_config.txt", 'w')
        f.write(str(solution))
        f.close()
    gen_no = gen_no + 1

#Lets plot the final front now
function1 = [i  for i in function1_values]
function2 = [j  for j in function2_values]
plt.xlabel('km_nw', fontsize=15)
plt.ylabel('pg_ts_wc', fontsize=15)
plt.scatter(function1, function2)
plt.show()
