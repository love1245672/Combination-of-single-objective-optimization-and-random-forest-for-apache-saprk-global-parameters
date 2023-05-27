import random
from xgboost import XGBRegressor
import pandas as pd
from sklearn.svm import SVR
import joblib
import warnings
import sys
warnings.filterwarnings("ignore", category=FutureWarning)

target = sys.argv[1]
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
early_stop_count = 0
def score_formula(x,max_dur,min_dur):
    score = ((x - min_dur) / (max_dur - min_dur))*100#100比分數 越小越好
    score = max(0.0, min(score, 100.0))#將分數限制在 0~100之間WW
    return score
    
def fitness(json_chromosome):
    df_chromosome = pd.DataFrame(json_chromosome,columns=cols)
    sum_score = 0
    #---------------------單benchmark優化--------------------------#
    if target == "km":
        sum_score += rf_model_km.predict(df_chromosome)[0]#權重
        return [sum_score]
    if target == "nw":
        sum_score += rf_model_nw.predict(df_chromosome)[0]#權重
        return [sum_score]
    if target == "pg":
        sum_score += rf_model_pg.predict(df_chromosome)[0]#權重
        return [sum_score]
    if target == "ts":
        sum_score += rf_model_ts.predict(df_chromosome)[0]#權重
        return [sum_score]
    if target == "wc":
        sum_score += rf_model_wc.predict(df_chromosome)[0]#權重
        return [sum_score]
    #---------------------全局benchmark優化---------------------------#
    if target == "all":
        sum_score += score_formula(rf_model_km.predict(df_chromosome)[0],204.188,67.699) * 1#權重
        sum_score += score_formula(rf_model_nw.predict(df_chromosome)[0],270.744,90.472) * 1
        sum_score += score_formula(rf_model_pg.predict(df_chromosome)[0],253.533,81.831) * 1
        sum_score += score_formula(rf_model_ts.predict(df_chromosome)[0],117.427,64.393) * 1
        sum_score += score_formula(rf_model_wc.predict(df_chromosome)[0],101.12,54.513) * 1
        return [sum_score/6]

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
# 初始化族群
def init_population(pop_size):
    population = []
    for i in range(pop_size):
        chromosome = create_inital_people()
        chromosome['fitness'] = fitness(chromosome)
        population.append(chromosome)
    return population

# 選擇個體
def selection(population):
    # 隨機取前 10%的人
    pop_20_percent = sorted(population, key=lambda chromosome: chromosome['fitness'][0])[:len(population) // 10]
    chromosome = random.choice(pop_20_percent)
    return chromosome

# 執行交配
def crossover(parent1, parent2):
    # 一點交叉法
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
    child['fitness'] = fitness(child)
    return child

# 執行突變
def mutation(chromosome, mutation_rate):
    # 均勻突變法
    if random.random() < mutation_rate:
        chromosome = create_inital_people()
        chromosome['fitness'] = fitness(chromosome)
def early_stop(p,next_p):
    global early_stop_count
    best_p = sorted(p, key=lambda chromosome: chromosome['fitness'][0])[0]['fitness'][0]
    best_next_p = sorted(next_p, key=lambda chromosome: chromosome['fitness'][0])[0]['fitness'][0]
    if best_next_p < best_p:
        early_stop_count = 0
    else:
        early_stop_count += 1 
# 主函數
def genetic_algorithm(pop_size, max_gen, elitism_size, mutation_rate):
    # 初始化族群
    population = init_population(pop_size)
    
    # 進行迭代
    for i in range(max_gen):
        # 選擇精英
        elites = sorted(population, key=lambda chromosome: chromosome['fitness'][0])[:elitism_size]
        # 複製精英到下一代族群
        next_pop = elites

        # 產生下一代族群
        while len(next_pop) < pop_size:
            # 選擇兩個父代進行交配
            parent1 = selection(population)
            parent2 = selection(population)
            child = crossover(parent1, parent2)
            # 對子代進行突變
            mutation(child, mutation_rate)
            # 將子代加入族群
            next_pop.append(child)

        prv_population = population
        population = next_pop
        early_stop(prv_population,population)
        if early_stop_count == 10:
            print("最終迭代次數: ", i)
            return sorted(population, key=lambda chromosome: chromosome['fitness'][0])
    # 返回最優解
    best_solution = sorted(population, key=lambda chromosome: chromosome['fitness'][0])
    #print(sorted(population, key=lambda chromosome: chromosome['fitness'][0]))
    return best_solution

# 執行遺傳演算法 pop_size 人口數
best_solution = genetic_algorithm(pop_size=10000, max_gen=100, elitism_size = 1500, mutation_rate=0.2)
print("次次優解：", best_solution[2])
print("次優解：", best_solution[1])
print("最優解：", best_solution[0])