# dqnAgents.py (수정본의 핵심만 반영)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, csv, random, pickle, math
import graphicsDisplay  # 추가: 프레임 저장 제어
from game import Directions
from tqdm import tqdm
from learningAgents import ReinforcementAgent

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, state_dim, size=100000):
        self.size = size
        self.ptr = 0
        self.full = False
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, 1), dtype=np.int64)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.next_states = np.zeros((size, state_dim), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        # print(f"[DEBUG] done type={type(done)}, value={done}")
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = np.float32(done)
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        max_size = self.size if self.full else self.ptr
        if max_size < batch_size:
            idx = np.random.choice(max_size, max_size, replace=False)
        else:
            idx = np.random.choice(max_size, batch_size, replace=False)
        return (self.states[idx].astype(np.float32),
                self.actions[idx].astype(np.int64),
                self.rewards[idx].astype(np.float32),
                self.next_states[idx].astype(np.float32),
                self.dones[idx].astype(np.float32).reshape(-1))

# -------------------------
# 상태 전처리
# -------------------------
def state_to_tensor(state):
    from util import manhattanDistance
    pacman = state.getPacmanPosition()
    ghosts = state.getGhostPositions()
    food = state.getFood().asList()
    capsules = state.getCapsules()

    # 거리 특징
    ghost_dists = [manhattanDistance(pacman, g) for g in ghosts]
    nearest_ghost = min(ghost_dists) if ghost_dists else 0.0
    nearest_food = min((manhattanDistance(pacman, f) for f in food), default=0.0)
    nearest_capsule = min((manhattanDistance(pacman, c) for c in capsules), default=0.0)

    # 방향성 특징(유령 상대 좌표 평균)
    if ghosts:
        mean_ghost_dx = np.mean([g[0] - pacman[0] for g in ghosts])
        mean_ghost_dy = np.mean([g[1] - pacman[1] for g in ghosts])
    else:
        mean_ghost_dx, mean_ghost_dy = 0.0, 0.0

    # 스케일 작은 카운트만 사용 (점수는 제거)
    features = np.array([
        pacman[0], pacman[1],
        nearest_ghost, nearest_food, nearest_capsule,
        mean_ghost_dx, mean_ghost_dy,
        len(food), len(capsules)
    ], dtype=np.float32)

    # 간단한 정규화(선택): 위치/거리 범위를 대략적인 값으로 나눠서 스케일 축소
    features[:5] = features[:5] / 10.0
    return features


def _clip_reward(r, clip=10.0):
    return float(np.clip(r, -clip, clip))


# -------------------------
# MLP 기반 DQN
# -------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.fc(x)
    

class DuelingMultiHeadDQN(nn.Module):
    """
    Multi-head encoder + Dueling head for 1D Pacman features.
    Assumes state vector indices:
      0:2   -> pacman (x,y)
      2:5   -> distances (nearest_ghost, nearest_food, nearest_capsule)
      5:7   -> ghost relative direction mean (dx, dy)
      7:9   -> counts (len(food), len(capsules))
    """
    def __init__(self, state_dim: int, action_dim: int,
                 head_dim: int = 64, fused_dim: int = 256,
                 dropout: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Defensive: if state_dim is not 9, fallback to single-head MLP
        self.use_multi = (state_dim >= 9)

        if self.use_multi:
            # Group slice indices (adjust if your state changes)
            self.s_pos = slice(0, 2)
            self.s_dists = slice(2, 5)
            self.s_gdir = slice(5, 7)
            self.s_counts = slice(7, 9)

            # Per-group encoders
            self.enc_pos = nn.Sequential(
                nn.Linear(2, head_dim),
                nn.ReLU(),
                nn.LayerNorm(head_dim),
                nn.Linear(head_dim, head_dim),
                nn.ReLU(),
            )
            self.enc_dists = nn.Sequential(
                nn.Linear(3, head_dim),
                nn.ReLU(),
                nn.LayerNorm(head_dim),
                nn.Linear(head_dim, head_dim),
                nn.ReLU(),
            )
            self.enc_gdir = nn.Sequential(
                nn.Linear(2, head_dim),
                nn.ReLU(),
                nn.LayerNorm(head_dim),
                nn.Linear(head_dim, head_dim),
                nn.ReLU(),
            )
            self.enc_counts = nn.Sequential(
                nn.Linear(2, head_dim),
                nn.ReLU(),
                nn.LayerNorm(head_dim),
                nn.Linear(head_dim, head_dim),
                nn.ReLU(),
            )

            fused_in = head_dim * 4
        else:
            # Fallback encoder for unknown state_dim
            self.enc_fallback = nn.Sequential(
                nn.Linear(state_dim, fused_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(fused_dim // 2),
                nn.Linear(fused_dim // 2, fused_dim // 2),
                nn.ReLU(),
            )
            fused_in = fused_dim // 2

        # Fusion trunk
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
        )

        # Dueling heads
        self.val_stream = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Linear(fused_dim // 2, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Linear(fused_dim // 2, action_dim)
        )

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, state_dim)
        if self.use_multi and x.size(1) >= 9:
            pos = x[:, self.s_pos]
            dists = x[:, self.s_dists]
            gdir = x[:, self.s_gdir]
            counts = x[:, self.s_counts]

            h_pos = self.enc_pos(pos)
            h_dists = self.enc_dists(dists)
            h_gdir = self.enc_gdir(gdir)
            h_counts = self.enc_counts(counts)

            h = torch.cat([h_pos, h_dists, h_gdir, h_counts], dim=1)
        else:
            h = self.enc_fallback(x)

        z = self.fuse(h)
        v = self.val_stream(z)                  # (batch, 1)
        a = self.adv_stream(z)                  # (batch, action_dim)
        a_mean = a.mean(dim=1, keepdim=True)    # dueling mean-subtraction
        q = v + (a - a_mean)
        return q


# -------------------------
# DQN Agent
# -------------------------
class DQNAgent(ReinforcementAgent):
    def __init__(self,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                 gamma=0.99, lr=1e-3, memory_size=10000,
                 target_update_freq=1000,
                 numTraining=0,
                 output_path=None,
                 save_points=None,
                 weights_path=None,
                 layout=None):
        
        super().__init__(
            actionFn=lambda state: state.getLegalActions(),
            numTraining=numTraining,
            epsilon=epsilon,
            alpha=lr,
            gamma=gamma
        )

        self.model = None
        self.target_model = None
        self.optimizer = None

        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.lr = float(lr)

        self.memory_size = int(memory_size)
        self.batch_size = 64
        self.target_update_freq = int(target_update_freq)
        self.total_steps = 0

        self.numTraining = int(numTraining)
        self.training_mode = True

        self.actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        self.action_dim = len(self.actions)

        self.prev_state_vec = None
        self.prev_action_idx = None
        self.prev_score = None
        self.episode_reward = 0.0
        self.episode_steps = 0

        self.output_path = output_path
        self.weights_path = weights_path
        self.layout_name = layout

        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            if not os.path.exists(self.output_path):
                with open(self.output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["episode", "reward", "win"])

        self.save_points = [int(x) for x in str(save_points).split(";")] if save_points else []
        self.progress_bar = None
        self.total_episodes = numTraining  # 학습 에피소드 수 저장

    def _lazy_init(self, state):
        if self.model is not None:
            return
        state_dim = len(state_to_tensor(state))
        self.model = DuelingMultiHeadDQN(state_dim, self.action_dim, head_dim=64, fused_dim=256, dropout=0.05)
        self.target_model = DuelingMultiHeadDQN(state_dim, self.action_dim, head_dim=64, fused_dim=256, dropout=0.05)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(state_dim, size=self.memory_size)
        
        if self.weights_path and os.path.exists(self.weights_path):
            self.load(self.weights_path)
            
    def load(self, path):
        """Pretrained 모델 파라미터 로드"""
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.target_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # 단순 state_dict만 저장된 경우
            self.model.load_state_dict(checkpoint)
            self.target_model.load_state_dict(checkpoint)
        print(f"[INFO] Loaded pretrained weights from {path}")

    def _action_to_idx(self, a):
        try:
            return self.actions.index(a)
        except ValueError:
            return self.actions.index(Directions.STOP)

    def registerInitialState(self, state):
        # 에피소드 시작만 직접 수행 (부모의 시작 메시지 출력은 건너뜀)
        self.startEpisode()

        # save_point 여부 확인 (오프셋 없이 현재 에피소드 번호 기준)
        self.recording = self.episodesSoFar in self.save_points
        if self.recording:
            self.actionHistory = []

        # 모델/메모리 초기화
        self._lazy_init(state)

        # tqdm 진행바 생성 (단 한 번만)
        if self.progress_bar is None and self.numTraining > 0:
            self.progress_bar = tqdm(total=self.numTraining, desc="Training Progress", ncols=80)

        # 학습/테스트 모드 자동 전환
        self.training_mode = (self.episodesSoFar <= self.numTraining)

        # 에피소드 상태 초기화
        self.prev_state_vec = state_to_tensor(state)
        self.prev_action_idx = None
        self.prev_score = state.getScore()
        self.episode_reward = 0.0
        self.episode_steps = 0

    def getAction(self, state):
        self._lazy_init(state)

        legal = state.getLegalActions()
        if not legal:
            return Directions.STOP

        # ε-greedy (테스트 모드면 ε 최소값)
        eps = self.epsilon if self.training_mode else self.epsilon_min
        if random.random() < eps:
            action = random.choice(legal)
            action_idx = self._action_to_idx(action)
        else:
            state_vec = state_to_tensor(state)
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze(0).numpy()
            legal_idxs = [self._action_to_idx(a) for a in legal]
            action_idx = max(((idx, q_values[idx]) for idx in legal_idxs), key=lambda x: x[1])[0]
            action = self.actions[action_idx]

        # 학습 모드에서 전이 추가 + 스텝 업데이트
        if self.training_mode and self.prev_action_idx is not None and self.prev_state_vec is not None:
            curr_score = state.getScore()
            reward = _clip_reward(curr_score - self.prev_score)
            next_state_vec = state_to_tensor(state)
            done = False
            self.memory.add(self.prev_state_vec, self.prev_action_idx, reward, next_state_vec, float(done))
            self.episode_reward += reward
            self.prev_state_vec = next_state_vec
            self.prev_score = curr_score
            self._update_step()

        # 현재 액션 저장
        self.prev_action_idx = action_idx
        self.episode_steps += 1

        if getattr(self, "recording", False):
            self.actionHistory.append(action)
        return action        


    def _update_step(self):
        if (self.memory.full is False and self.memory.ptr < max(self.batch_size, 1000)):
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int64).view(-1, 1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).view(-1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32).view(-1)

        q_values_all = self.model(states)
        q_values = q_values_all.gather(1, actions).squeeze(-1)

        with torch.no_grad():
            next_q_online = self.model(next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target_all = self.target_model(next_states)
            next_q_values = next_q_target_all.gather(1, next_actions).squeeze(-1)
            one_minus_dones = 1.0 - dones
            target = rewards + self.gamma * next_q_values * one_minus_dones

        loss = nn.SmoothL1Loss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % max(500, self.target_update_freq) == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def final(self, state):
        # 마지막 전이 처리
        if self.training_mode and self.prev_action_idx is not None and self.prev_state_vec is not None:
            curr_score = state.getScore()
            reward = _clip_reward(curr_score - self.prev_score)
            next_state_vec = state_to_tensor(state)
            done = True
            self.memory.add(self.prev_state_vec, self.prev_action_idx, reward, next_state_vec, float(done))
            self.episode_reward += reward
            for _ in range(4):
                self._update_step()

        # CSV 기록
        win = 1 if state.isWin() else 0
        if self.output_path:
            with open(self.output_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.episodesSoFar, f"{self.episode_reward:.3f}", win])

        # ε decay
        if self.training_mode:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # save_point일 때만 action 기록 저장
        if getattr(self, "recording", False):
            os.makedirs("records/DQNAgent", exist_ok=True)
            fname = f"records/DQNAgent/record-ep{self.episodesSoFar}.pkl"
            with open(fname, "wb") as f:
                pickle.dump({"layout": self.layout_name, "actions": self.actionHistory}, f)
            self.recording = False
            self.actionHistory = []

        # 에피소드 종료 (부모의 집계만 맞춰주기 위해 직접 호출)
        self.stopEpisode()

        # 마지막 에피소드에서만 가중치 저장
        if self.episodesSoFar >= self.numTraining and self.weights_path:
            os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.weights_path)

        # 상태 리셋 + tqdm 업데이트
        self.prev_state_vec = None
        self.prev_action_idx = None
        self.prev_score = None
        self.episode_reward = 0.0
        self.episode_steps = 0

        if self.training_mode and self.progress_bar is not None:
            self.progress_bar.update(1)
