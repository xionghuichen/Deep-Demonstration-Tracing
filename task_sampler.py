import numpy as np
import random
import copy
import os


class TaskSampler:
    def __init__(self, task_repeated, task_random_sample_prob) -> None:
        self.task_repeated = task_repeated
        self.task_random_sample_prob = task_random_sample_prob
        self.task_repeat_counter = 0
        self.eps = 1e-6

    def construct_taskset(self, iid_train_task_ids):
        self.iid_train_task_ids = iid_train_task_ids
        self.failure_rates = {}
        for tid in self.iid_train_task_ids:
            self.failure_rates[tid] = 1

    def sample_next(self):
        update = False
        if self.task_repeat_counter % self.task_repeated == 0:
            update = True
            if np.random.random_sample() < self.task_random_sample_prob:
                task_id = random.sample(list(self.iid_train_task_ids), k=1)[0]
            else:
                failure_rate_list = np.array(
                    [
                        self.failure_rates[tid] if tid in self.failure_rates else 1
                        for tid in self.iid_train_task_ids
                    ]
                )
                task_sample_prob = failure_rate_list + self.eps
                if sum(task_sample_prob) != 0:
                    task_sample_prob = task_sample_prob / sum(task_sample_prob)
                    task_id = np.random.choice(
                        self.iid_train_task_ids, p=task_sample_prob, replace=False
                    )
                else:
                    task_id = random.sample(list(self.iid_train_task_ids), k=1)[
                        0
                    ]  # degrade
        return task_id, update

    def update_failure_rate(self, task_id, failure_rate):
        self.failure_rates[task_id] = failure_rate
        self.task_repeat_counter += 1
        return
