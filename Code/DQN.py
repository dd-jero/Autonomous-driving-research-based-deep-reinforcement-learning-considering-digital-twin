from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda
import torch.backends.cudnn as cudnn

import numpy as np
import cupy as cp
import random
import datetime
from collections import deque
from Car_gym import Car_gym
import pandas as pd
import gc

import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('best_model/')
load = False
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./best_model/{date_time}.pkl"
load_path = f"./best_model/obstacle.pkl"

class Q_network(nn.Module):
    def __init__(self):
        super(Q_network, self).__init__()

        # Ray 추가할때 사용
        # self.vector_fc = nn.Linear(95, 512)

        self.image_cnn = nn.Sequential(
            # nn.Conv2d(입력채널 수, 출력 채널수, kernel_size=필터 크기, stride=필터 이동간격)
            # 64 x 64 x 6 -> 30 x 30 x 32 (0326: 4장 -> 6장)
            nn.Conv2d(6, 32, kernel_size=6, stride=2, groups=1, bias=True),
            # GELU : 더 유연한 RELU (0에 가까운 음수 값을 살려 상태에 대한 다양성 고려)
            nn.GELU(),
            # 30 x 30 x 32 -> 13 x 13 x 64 
            nn.Conv2d(32, 64, kernel_size=6, stride=2, groups=1, bias=True),
            nn.GELU(),
            # 14 x 14 x 64 -> 6 x 6 x 64 
            nn.Conv2d(64, 64, kernel_size=4, stride=2, groups=1,bias=True),
            nn.GELU(),
            # 6 x 6 x 64  -> 3 x 3 x 64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, groups=1, bias=True),
            nn.GELU()
        )
        # 3x3x64 -> 1 x 576
        # 1 x 576 + 1 x 5 -> 1 x 581
        self.image_fc = nn.Linear(581, 256)
        # 1 x 581 -> 1 x 256
        self.fc_action1 = nn.Linear(256, 256)
        # 1 x 256 -> 1 x 16
        self.fc_action2 = nn.Linear(256, 16) 
    
    def forward(self, camera, ray):
        camera_obs = camera
        ray_obs = ray
        batch = camera_obs.size(0)  

        # self.image_cnn(camera_obs) : 
        # torch.view(batch, -1) : self.image_cnn(camera_obs) tensor를 (1, ?)의 2차원 tensor로 변경
        # camera_obs 사이즈(image_cnn 이후): torch.Size([1, 64, 4, 4])
        # camera_obs 사이즈(view(batch, -1) 이후) : torch.Size([1, 1024])
        camera_obs = self.image_cnn(camera_obs).view(batch, -1)

        # camera, ray obs를 합치기 위해 torch shape을 (1, ?) -> (?, 1) 로 변경
        # col vector -> row vector화 : torch.cat 사용하기 위해
        camera_obs = camera_obs.view(-1, batch)
        ray_obs = ray_obs.view(-1, batch)

        # x에 ray 성분 추가
        x = torch.cat([camera_obs, ray_obs], dim=0) 
        # row vector -> col vector화 : fully connected layer 적용하기 위해
        x = x.view(batch, -1)

        # print("현재 x tensor의 shape : ", x.shape)
        # print("-------------------------------------")

        x = F.gelu(self.image_fc(x))
        # print("현재 camera sensor : ", x.shape)
        # print("----------------------------------")

        x = F.gelu(self.fc_action1(x))
        Q_values = self.fc_action2(x)

        # 16개의 행동에 대한 기대치 출력
        return Q_values

class Agent:
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy_net = Q_network().to(device)
        self.Q_target_net = Q_network().to(device)
    
        cudnn.enabled = True # GPU 가속 활성화
        cudnn.benchmark = True # 입력 사이즈에 따른 GPU 연산 알고리즘 선택 

        self.learning_rate = 0.0003 #  optimizer로 역전파 진행 시 파라미터를 얼마 업데이트 할지 

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # Q의 nn parameter를 target network에 불러오기
        self.Q_target_net.load_state_dict(self.policy_net.state_dict())

        # epsilon
        self.epsilon = 1
        # epsilon decay 값
        self.epsilon_decay = 0.00001

        self.device = device
        
        self.data_buffer = deque(maxlen=100000)

        self.x_epoch = list()
        # max Q value 32개(batch size) 평균
        self.y_max_Q_avg = list()

        self.y_loss = list()

        self.writer = SummaryWriter('best_model/')

        if load == True:
            print("Load trained model..")
            self.load_model()

    def epsilon_greedy(self, Q_values):
        # 난수 생성, 
        # epsilon보다 작은 값일 경우
        if np.random.rand() <= self.epsilon:
            # action을 random하게 선택
            action = random.randrange(16)

            return action

        # epsilon보다 큰 값일 경우
        else:
            # 학습된 Q value 값중 가장 큰 action 선택
            return Q_values.argmax().item()
    
    # def policy(self, obs):
    # def train_policcy(self, obs):

    # model 저장
    def save_model(self):
        torch.save({
                    'state': self.policy_net.state_dict(), 
                    'optim': self.optimizer.state_dict()},
                    save_path)
        return None

    # model 불러오기
    def load_model(self):
        checkpoint = torch.load(load_path)
        self.policy_net.load_state_dict(checkpoint['state'])
        self.Q_target_net.load_state_dict(checkpoint['state'])
        self.optimizer.load_state_dict(checkpoint['optim'])
        return None

    def store_trajectory(self, traj):
        self.data_buffer.append(traj)

    # 1. resizing : 64 * 64, gray scale로
    def re_scale_frame(self, obs):
        obs = cp.array(obs)  # cupy 배열로 변환
        obs = cp.asnumpy(obs)  # 다시 numpy 배열로 변환
        obs = resize(rgb2gray(obs), (64, 64))
        return obs

    # 2. image 6개씩 쌓기
    def init_image_obs(self, obs):
        obs = self.re_scale_frame(obs)
        frame_obs = [obs for _ in range(6)]
        frame_obs = np.stack(frame_obs, axis=0)
        frame_obs = cp.array(frame_obs)  # cupy 배열로 변환
        return frame_obs
    
    # 3. 6장 쌓인 Image return
    def init_obs(self, obs):
        return self.init_image_obs(obs)
    
    def camera_obs(self, obs):
        camera_obs = cp.array(obs)  # cupy 배열로 변환
        camera_obs = cp.expand_dims(camera_obs, axis=0)
        camera_obs = torch.from_numpy(cp.asnumpy(camera_obs)).to(self.device)  # GPU로 전송
        return camera_obs

    def ray_obs(self, obs):
        ray_obs = cp.array(obs)  # cupy 배열로 변환
        ray_obs = cp.expand_dims(ray_obs, axis=0)
        ray_obs = torch.from_numpy(cp.asnumpy(ray_obs)).to(self.device)  # GPU로 전송
        return ray_obs

    # numpy 변환은 cpu 연산으로 한 결과에만 적용 가능? => GPU에서 Numpy 연산이 가능하도록 cupy 라이브러리 활용 
    def ray_obs_cpu(self, obs):
        obs_gpu = cp.asarray(obs)
        obs_gpu = cp.reshape(obs_gpu, (1,-1))
        return cp.asnumpy(obs_gpu)

    # FIFO, 6개씩 쌓기
    # step 증가함에 따라 맨 앞 frame 교체
    # 111111 -> 111112 -> 111123 -> 111234 -> 112345 -> ...

    def accumulated_image_obs(self, obs, new_frame):
        temp_obs = obs[1:, :, :]
        new_frame = self.re_scale_frame(new_frame)
        temp_obs = cp.array(temp_obs)  # cupy 배열로 변환
        new_frame = cp.array(new_frame)  # cupy 배열로 변환
        new_frame = cp.expand_dims(new_frame, axis=0)
        frame_obs = cp.concatenate((temp_obs, new_frame), axis=0)
        frame_obs = cp.asnumpy(frame_obs)  # 다시 numpy 배열로 변환
        return frame_obs

    def accumlated_all_obs(self, obs, next_obs):
        return self.accumulated_image_obs(obs, next_obs)

    def convert_action(self, action):
        return action

    # action 선택, discrete action 16개 존재
    # obs shape : torch.Size([1, 4, 64, 64])
    def train_policy(self, obs_camera, obs_ray):
        Q_values = self.policy_net(obs_camera, obs_ray)
        action = self.epsilon_greedy(Q_values)
        return self.convert_action(action), action

    def batch_torch_obs(self, obs):
        obs = [cp.asarray(ob) for ob in obs]  # obs의 모든 요소를 cupy 배열로 변환
        obs = cp.stack(obs, axis=0)  # obs를 축 0을 기준으로 스택
        obs = cp.squeeze(obs, axis=0) if obs.shape[0] == 1 else obs  # 첫 번째 축 제거
        obs = cp.asnumpy(obs)  # 다시 numpy 배열로 변환
        obs = torch.from_numpy(obs).to(self.device)  # torch tensor로 변환
        return obs

    def batch_ray_obs(self, obs):
        obs = cp.asarray(obs)  # cupy 배열로 변환
        obs = cp.expand_dims(obs, axis=0)  # 새로운 축 추가
        obs = torch.from_numpy(cp.asnumpy(obs)).to(self.device)  # torch tensor로 변환
        return obs

    def batch_signal_obs(self, obs):
        obs = cp.asarray(obs)  # cupy 배열로 변환
        obs = cp.expand_dims(obs, axis=0)  # 새로운 축 추가
        obs = torch.from_numpy(cp.asnumpy(obs)).to(self.device).to(torch.uint8)  # torch tensor로 변환
        return obs

    # update target network 
    # 저장되어있던 를 target network에 복사
    def update_target(self):
        self.Q_target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, step, update_target):

        # discount factor
        gamma = 0.95 # 미래가치(타겟값)을 얼마나 반영할 지 

        # epsilon decaying
        self.epsilon -= self.epsilon_decay
        # min of epsilon : 0.05
        self.epsilon = max(self.epsilon, 0.05)
        
        # replay buffer에서 무작위로 64개의 샘플을 뽑아 미니 배치 학습
        random_mini_batch = random.sample(self.data_buffer, 64)

        # data 분배 (각 리스트에 분배 후 텐서로 변환하여 모델의 입력으로 사용)
        # 현재 state, action, reward, next state, 게임종료
        obs_camera_list, obs_ray_list, action_list, reward_list, next_obs_camera_list, next_obs_ray_list, mask_list = [], [], [], [], [], [], []

        for all_obs in random_mini_batch:
            s_c, s_r, a, r, next_s_c, next_s_r, mask= all_obs
            obs_camera_list.append(s_c)
            obs_ray_list.append(s_r)
            action_list.append(a)
            reward_list.append(r)
            next_obs_camera_list.append(next_s_c)
            next_obs_ray_list.append(next_s_r)
            mask_list.append(mask)

        # tensor
        obses_camera = self.batch_torch_obs(obs_camera_list)
        obses_ray = self.batch_ray_obs(obs_ray_list)

        actions = torch.LongTensor(action_list).unsqueeze(1).to(self.device)
        rewards = torch.Tensor(reward_list).to(self.device)
        
        next_obses_camera = self.batch_torch_obs(next_obs_camera_list)
        next_obses_ray = self.batch_ray_obs(next_obs_ray_list)
        
        masks = torch.Tensor(mask_list).to(self.device)

        #print("signal:",obs_signal.shape)

        # get Q-value
        Q_values = self.policy_net(obses_camera, obses_ray) # Q-Network로 예측 Q-value 출력
        del obses_camera, obses_ray
 
        # 추정값
        q_value = Q_values.gather(1, actions).view(-1)
        # print(q_value)
        # print('--------------------------------------')
        
        # get target, y(타겟값) 구하기 위한 다음 state에서의 max Q value
        # target network에서 next state에서의 max Q value -> 상수값
        with torch.no_grad(): # Target Network는 파라미터 업데이트가 일어나지 않으므로 이 메소드를 사용해 연산 최적화 
            target_q_value = self.Q_target_net(next_obses_camera, next_obses_ray).max(1)[0] # Target Network로 Target Q-value(출력)

        # 타겟값(y)
        Y = rewards + masks * gamma * target_q_value 

        # loss 정의
        loss = F.mse_loss(q_value, Y.detach())
        # writer.add_scalar("Loss",loss, step)

        # 10,000번의 episode동안 몇 번의 target network update가 있는지
        # target network update 마다 max Q-value / loss function 분포
        if step % update_target == 0:
            self.x_epoch.append(step//update_target)

            # tensor -> list
            # max Q-value 분포
            tensor_to_list_q_value = target_q_value.tolist()
            # max_Q 값들(batch size : 32개)의 평균 값 
            list_q_value_avg = sum(tensor_to_list_q_value)/len(tensor_to_list_q_value)
            # writer.add_scalar("max Q-value (avg)",list_q_value_avg, step)
            self.y_max_Q_avg.append(list_q_value_avg)

            # loss 평균 분포(reduction = mean)
            loss_in_list = loss.tolist()
            self.y_loss.append(loss_in_list)

        # backward 시작
        # Adam optimizer를 통해 Q-network의 model.parameters()의 loss function 최적화
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        gc.collect() # 메모리 최적화

def main():

    engine_configuration_channel = EngineConfigurationChannel() # 유니티와 파이썬을 연결하는 메인 채널
    # 파일 위치 지정 중요!
    env = Car_gym(
        time_scale=1.0, port=11300, filename='230519_project.exe') # Car_gym에서 서브 채널을 열어 실질적 데이터 통신 
    
    score = 0
    # episode당 step
    episode_step = 0
    # 전체 누적 step
    step = 0

    initial_exploration = 5000 # 초기 replay buffer 채우기 위한 탐험 -> 5000개의 샘플을 저장한 후 학습 시작 
    
    update_target = 4000 # 4000 step마다 타겟 네트워크의 파라미터 업데이트 

    # episode당 이동한 거리
    episode_distance = 0
    total_epi_dis = 0
    total_step = 0

    y_epi_dis = list()
    x_episode = list()
    y_epi_avg =list()

    # x축 : 학습 과정에서 epoch 수, y축 : maxQ 값의 변화
    # x축 : step 수                y축 : loss 값
    # x축 : episode 수,            y축 : episode당 step수(주행시간)

    # Agent() class 초기화: __init__(self) 자동으로 호출
    agent = Agent()

    if load:
        agent.load_model()

    for epi in range(4501):

        speed_count_1 = 0
        speed_count_2 = 0   
        speed_count_3 = 0
        deceleration = 0

        obs = env.reset() # 상태 데이터를 불러옴
        # 3차원 (1, 84, 84, 3) : camera sensor
        obs_camera = obs[0] # 첫 번째 상태: 카메라 센서의 차량 전방 상황 


        obs_camera = torch.Tensor(obs_camera)
        # print("main 함수에서 camera tensor의 shape : ", obs_camera.shape)
        # print("-------------------------------------")        
        
        # 3차원 (1, 84, 84, 3) -> 2차원 (84, 84, 3)
        obs_camera = torch.Tensor(obs_camera).squeeze(dim=0)
        # (84, 84, 3) -> (64, 64, 1) -> 4장씩 쌓아 (64, 64, 4)
        # 같은 Image 4장 쌓기 -> 이후 action에 따라 환경이 바뀌고, 다른 Image data 쌓임
        obs_camera = agent.init_obs(obs_camera)


        # 1차원                : ray sensor
        obs_ray = obs[1] # 두 번째 상태: 레이 센서의 오브젝트 충돌 거리 
        obs_ray_tensor = [obs_ray[1], obs_ray[7], obs_ray[9], obs_ray[35], obs_ray[37]]
        obs_ray_tensor = torch.Tensor(obs_ray_tensor)
        # 0 3 4 17 18
        # obs_ray[1] : 정면
        # obs_ray[7] : 우측 대각
        # obs_ray[9] : 좌측 대각
        # obs_ray[35] : 우측
        # obs_ray[37] : 좌측

        while True:
            # print('누적 step: ', step)

            # action 선택
            action, dis_action = agent.train_policy(agent.camera_obs(obs_camera), agent.ray_obs(obs_ray_tensor)) # Q-network로 액션 결정 
            # action에 따른 step()
            next_obs, reward, done = env.step(action) # 액션에 따른 다음 상태, 보상, 에피소드 종료 유/무 확인

            # action당 이동 거리 측정 -> 한 Episode당 이동한 거리 측정
            if not done:
                if action == 0 or action == 3 or action == 6 or action ==  9 or action ==  12:    
                    episode_distance += 5.5
                    speed_count_1 += 1
                elif action == 1 or action == 4 or action == 7 or action == 10 or action ==  13: 
                    episode_distance += 9.0
                    speed_count_2 += 1
                elif action == 2 or action == 5 or action == 8 or action == 11 or action ==  14: 
                    episode_distance += 12.5
                    speed_count_3 += 1
                elif action == 15: # 이전 속도의 70%만 이용하는 감속 행동 
                    deceleration += 1
            else:
                if epi % 100 == 0:
                    y_epi_dis.append(total_epi_dis // 100)
                    total_epi_dis = 0
                else:
                    total_epi_dis += episode_distance

            # state는 camera sensor로 얻은 Image만
            next_obs_camera = next_obs[0]
            next_obs_ray = next_obs[1]

            #print(next_obs_ray.shape)
            next_obs_ray_tensor = [next_obs_ray[1], next_obs_ray[7], next_obs_ray[9], next_obs_ray[35], next_obs_ray[37]]
            next_obs_ray_tensor = torch.Tensor(next_obs_ray_tensor)

            next_obs_camera = torch.Tensor(next_obs_camera).squeeze(dim=0)
            # step이 증가함에 따라 6장 중 1장씩 밀기(FIFO)
            next_obs_camera = agent.accumlated_all_obs(obs_camera, next_obs_camera)

            mask = 0 if done else 1
            # print("%d번째 step에서의 reward : %f, action speed : %f"%(step, reward, action_speed))
            score += reward
            # reward_score += reward

            # maxlen = 100,000인 data buffer(경험 데이터)에 저장
            # 현재 상태(camera, ray), 현재 행동, 보상, 다음 상태(camera, ray), 종료 유/무를 하나의 샘플로 묶어 replay buffer에 저장 
            agent.store_trajectory((obs_camera, agent.ray_obs_cpu(obs_ray_tensor), dis_action, reward, next_obs_camera, agent.ray_obs_cpu(next_obs_ray_tensor), mask))

            # 다음 상태를 현재 상태로 업데이트 
            obs_camera = next_obs_camera
            obs_ray_tensor = next_obs_ray_tensor

            #if os.path.exists('best_model/trained_model.pkl'):
            #        agent.load_model()


            # 초기 5000step 이후 학습 시작 
            if step > initial_exploration:
                agent.train(step, update_target)
                
                # 1000 step마다 Q-Network 파라미터  저장
                if step % 1000 == 0:
                    agent.save_model()
                # update_target 마다 저장된 모델을 타겟 네트워크로 불러오기
                if step % update_target == 0:
                    agent.update_target()

            episode_step += 1
            step += 1

            if done: # 에피소드 종료 시
                break

            cuda.empty_cache() # GPU 캐시 정리 

        if (epi + 1) % 1 == 0:
            print('%d 번째 episode의 총 step: %d'% (epi + 1, episode_step))
            print('deceleration:  %d 번'% deceleration)
            print('Speed  5.5:    %d 번'% speed_count_1)
            print('Speed  9.0:    %d 번'% speed_count_2)
            print('Speed 12.5:    %d 번'% speed_count_3)
            print('True_score:    %f'% score)
            print('Total step:    %d\n'% step)
            #writer.add_scalar("Score",score, epi)
            # agent.write_summary(score, agent.epsilon, epi)

        # 100 episode까지의 step 전체
        total_step = total_step + episode_step
        # 100 episode 마다
        if epi % 100 == 0:
            x_episode.append(epi)
            # second의 평균
            y_epi_avg.append(total_step//100)
            #writer.add_scalar("driving time(sec)",total_step//100, epi)
            total_step = 0

        score = 0
        # reward_score = 0
        episode_step = 0
        episode_distance = 0

    # 엑셀로 평균 주행 시간 (Step), 평균 주행 거리, Max Q-value, loss에 대한 데이터 저장 
    results_df = pd.DataFrame({'Episode' : x_episode, 'y_epi_avg' : y_epi_avg})
    results_df.to_excel('./results/time_results.xlsx', index=False)
    
    results_df = pd.DataFrame({'Episode' : x_episode, 'y_epi_dis' : y_epi_dis})
    results_df.to_excel('./results/distance_results.xlsx', index=False)


    results_df = pd.DataFrame({'Target Update' : agent.x_epoch, 'Max Q value' : agent.y_max_Q_avg})
    results_df.to_excel('./results/Q_value_results.xlsx', index=False)

    results_df = pd.DataFrame({'Target Update' : agent.x_epoch, 'Loss' : agent.y_loss})
    results_df.to_excel('./results/Loss_results.xlsx', index=False)

    # 모든 에피소드 이후 그래프 출력 
    # 100 episode 마다 episode 종료시까지의 평균 주행시간(second)
    plt.figure(1)
    plt.plot(x_episode, y_epi_avg)
    plt.xlabel('episode')
    plt.ylabel('driving time(sec)')

    # target update(2000 step) 마다 maxQ 값 변화
    plt.figure(2)
    plt.plot(agent.x_epoch, agent.y_max_Q_avg)
    plt.xlabel('target update')
    plt.ylabel('max Q value')

    # target update(2000 step) 마다 loss 값 변화
    plt.figure(3)
    plt.plot(agent.x_epoch, agent.y_loss)
    plt.xlabel('target update')
    plt.ylabel('loss')

    # 100 episode마다 평균 주행시간
    plt.figure(4)
    plt.plot(x_episode, y_epi_dis)
    plt.xlabel('episode')
    plt.ylabel('move distance')
    plt.show()

if __name__ == '__main__':
    main()

