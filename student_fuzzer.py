import random
from fuzzingbook import GreyboxFuzzer as gbf
from fuzzingbook import Coverage as cv
from fuzzingbook import MutationFuzzer as mf
from typing import List, Set, Any, Tuple, Dict, Union
import traceback
import numpy as np
import time
import coverage
from bug import entrypoint
from bug import get_initial_corpus

class Particle:
    def __init__(self, num_strategies):
        self.position = [random.random() for _ in range(num_strategies)]  # 随机初始化位置
        self.velocity = [random.random() for _ in range(num_strategies)]  # 随机初始化速度
        self.best_position = list(self.position)  # 最佳位置初始化
        self.best_score = float('inf')  # 最佳得分初始化

class PSO:
    def __init__(self, num_particles, num_strategies):
        self.particles = [Particle(num_strategies) for _ in range(num_particles)]
        self.global_best_position = [random.random() for _ in range(num_strategies)]
        self.global_best_score = float('inf')

    def optimize(self, feedback_data):
        W = 0.5  # 惯性权重
        V_MAX = 1.0  # 最大速度
        V_MIN = -1.0  # 最小速度
        C1, C2 = 2.05, 2.05  # 学习因子
        for particle in self.particles:
            for i in range(len(particle.position)):
                r1, r2 = random.random(), random.random()
                particle.velocity[i] = (W * particle.velocity[i] +
                                        C1 * r1 * (particle.best_position[i] - particle.position[i]) +
                                        C2 * r2 * (self.global_best_position[i] - particle.position[i]))
                # 限制速度
                particle.velocity[i] = max(V_MIN, min(V_MAX, particle.velocity[i]))
                particle.position[i] += particle.velocity[i]

            score = self.evaluate(particle, feedback_data)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = list(particle.position)
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = list(particle.position)
        return self.global_best_position

    def evaluate(self, particle, feedback_data):
        total_score = 0
 # 遍历所有反馈数据
        for feedback in feedback_data:
    # 假设 feedback 是一个元组，形如 (new_coverage, found_bugs)
            new_coverage = int(feedback[0]) if feedback[0] is not None else 0
 # 将 'PASS' 视为 0 错误，其他情况视为发现错误
            found_bugs = 0 if feedback[1] == 'PASS' else 1
# 累加每次迭代的分数
            total_score += new_coverage - 63 * found_bugs
        return total_score

# MOpt 基于的调度类
class MOptSchedule:
    def __init__(self, initial_probabilities):
        self.mutation_probabilities = initial_probabilities
        self.pso = PSO(num_particles=5, num_strategies=5)
        self.feedback_data = []

    def update_mutation_probabilities(self):
        optimized_probs = self.pso.optimize(self.feedback_data)
        self.mutation_probabilities = optimized_probs

    def record_feedback(self, feedback):
        self.feedback_data.append(feedback)

    def select_mutation_strategy(self):
        # 一个简单的示例，选择概率最高的策略
        max_prob = max(self.mutation_probabilities)
        return self.mutation_probabilities.index(max_prob)

class MyFuzzer:
    def __init__(self, seed_inputs, schedule, mutator, max_trials=999999):
        self.seed_inputs = seed_inputs  # 初始种子输入
        self.schedule = schedule  # MOpt 调度策略
        self.mutator = mutator  # 变异器
        self.max_trials = max_trials
        self.current_trials = 0

    def is_done(self):
        #print(f"cur_trial: {self.current_trials}")
        #print(f"max_trial: {self.max_trials}")
        return self.current_trials >= self.max_trials

    def run_one_cycle(self, runner):
        
        # 选择一个变异策略
        strategy = self.schedule.select_mutation_strategy()

        # 从种子输入中随机选择一个输入进行变异
        seed_input = random.choice(self.seed_inputs)
        mutated_input = self.mutator.mutate(seed_input)

        # 运行测试并获取结果
        
        result = runner.run(mutated_input)
        feedback = self.analyze_result(result)
        self.schedule.record_feedback(feedback)
        
        # 更新测试次数
        self.current_trials += 1
        #print(f"Total trials: {self.current_trials}")
        end_time = time.time()
        total_time = end_time - start_time

        print(f"Total fuzzing time: {total_time} seconds")
        
        return result
    
    def analyze_result(self, result):
        # 示例：从结果中提取反馈信息
        new_coverage = result["new_coverage"] if "new_coverage" in result else 0
        found_bugs = result["found_bugs"] if "found_bugs" in result else 0
        return new_coverage, found_bugs
# 主程序
if __name__ == "__main__":
    # 初始种子输入
    seed_inputs = get_initial_corpus()
    initial_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    schedule = MOptSchedule(initial_probs)
    #line_runner = MyFunctionCoverageRunner(entrypoint)
    line_runner = mf.FunctionCoverageRunner(entrypoint)
    fuzzer = MyFuzzer(seed_inputs, schedule, gbf.Mutator())
    start_time = time.time()
    while not fuzzer.is_done():
        feedback = fuzzer.run_one_cycle(line_runner)
        schedule.record_feedback(feedback)
        schedule.update_mutation_probabilities()
    