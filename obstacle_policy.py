import numpy as np

class BaseObstaclePolicy:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, obs:np.ndarray, action:np.ndarray, obstacle_try:int, obstacle_success:bool):
        # return action, obstacle_try, obstacle_success, info
        raise NotImplementedError


class pick_place_v2_drop_block(BaseObstaclePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h_start = 0.021
        self.h_after_free_fall = 0.02005
        assert self.h_start > self.h_after_free_fall

    def __call__(self, obs:np.ndarray, action:np.ndarray, obstacle_try:int, obstacle_success:bool):
        # only drop once
        # always start in 0.017
        if obs[6]>=self.h_start and obs[3]<=0.98 and not obstacle_success:  # obs 3 is gripper open state, 1 is totally open
            obstacle_try += 1
            action[3] = -(0.90+0.1*np.random.rand()) # open the gripper
            print('drop', obs[6])
        if obstacle_try>0 and obs[6]<self.h_after_free_fall:
            obstacle_success=True
        return action, obstacle_try, obstacle_success, {}


class basket_ball_v2_drop_ball(BaseObstaclePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        eps= 5e-4
        self.h_start = 0.035
        self.h_after_free_fall = self.h_start - eps
        assert self.h_start > self.h_after_free_fall

    def __call__(self, obs:np.ndarray, action:np.ndarray, obstacle_try:int, obstacle_success:bool):
        # only drop once
        # always start in 0.029872163199400722, but h will decrease range to ~0.024 for elastic ball reason when we first grasp the ball 
        if obs[6]>=self.h_start and obs[3]<=0.98 and not obstacle_success:  # obs 3 is gripper open state, 1 is totally open
            obstacle_try += 1
            action[3] = -(0.90+0.1*np.random.rand()) # open the gripper
            print('drop', obs[6])
        if obstacle_try>0 and obs[6]<self.h_after_free_fall:
            obstacle_success=True
        return action, obstacle_try, obstacle_success, {}


class peg_insert_side_v2_drop_peg(BaseObstaclePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        eps= 5e-4
        self.h_start = 0.035
        self.h_after_free_fall = self.h_start - eps
        assert self.h_start > self.h_after_free_fall

    def __call__(self, obs:np.ndarray, action:np.ndarray, obstacle_try:int, obstacle_success:bool):
        # only drop once
        # always start in  0.02999999566214617
        if obs[6]>=self.h_start and obs[3]<=0.98 and not obstacle_success:  # obs 3 is gripper open state, 1 is totally open
            obstacle_try += 1
            action[3] = -(0.90+0.1*np.random.rand()) # open the gripper
            print('drop', obs[6])
        if obstacle_try>0 and obs[6]<self.h_after_free_fall:
            obstacle_success=True
        return action, obstacle_try, obstacle_success, {}

class sudden_random_action(BaseObstaclePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.try_threshold = 5

    def __call__(self, obs:np.ndarray, action:np.ndarray, obstacle_try:int, obstacle_success:bool):
        # only drop once
        # 5 sudden action
        if np.random.rand()<0.4:  # 0.9 prob not to do sudden action
            return action, obstacle_try, obstacle_success, {}
        
        action=action.copy()
        gripper_action = action[-1]
        action = 1.0-2*np.random.rand(4)
        action[-1]=gripper_action

        obstacle_try += 1
        if obstacle_try>self.try_threshold:
            obstacle_success=True
        return action, obstacle_try, obstacle_success, {}
# "pick-place-hole-v2" "pick-place-wall-v2" "shelf-place-v2" "basketball-v2" "peg-insert-side-v2"


def Initial_Obstacle_Policy(env_name:str, *args, **kwargs):
    map_dict = {
        'pick-place-v2': pick_place_v2_drop_block,
        'pick-place-hole-v2': pick_place_v2_drop_block,
        'pick-place-wall-v2': pick_place_v2_drop_block,
        'shelf-place-v2': pick_place_v2_drop_block,
        'basketball-v2': basket_ball_v2_drop_ball,
        'peg-insert-side-v2': peg_insert_side_v2_drop_peg,
        'button-press-v2': sudden_random_action,
        'sweep-v2': sudden_random_action,
        'reach-v2': sudden_random_action,
        'push-v2': sudden_random_action,
    }
    return map_dict[env_name]()


if __name__ == '__main__':
    obs, action, obstacle_try, obstacle_success = np.zeros(7), np.zeros(4), 0, False
    obs[6]=100
    policy = Initial_Obstacle_Policy('pick-place-v2')
    print(policy(obs, action, obstacle_try, obstacle_success))