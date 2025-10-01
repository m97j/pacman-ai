# Reinforcement Learning for Pacman

본 디렉토리는 **UC Berkeley CS188 Reinforcement Learning 과제(Project 3)**에 해당하는,  
파일들을(Value Iteration, Q-Learning, Approximate Q-Learning)구현 하였고 이를 기반으로 DQN Agent까지로 확장한 결과물 입니다.  
기존 과제 구현내용에 추가적으로 **DQN(Deep Q-Network) Agent**를 직접 설계/구현하여 Pacman 게임 환경에서 학습 및 테스트한 결과를 포함합니다.

👉 [Approximate Q Agent vs DQN Agent 비교 시각화 사이트](https://pacman-ai-deploy.onrender.com)

---

## 📌 과제 개요

- **Value Iteration & Q-Learning**: GridWorld, Crawler, Pacman 환경에서의 기본 RL 알고리즘 구현.
- **Approximate Q Agent**: Feature 기반 근사 Q-learning.  
  - CS188 과제에서 제공된 가장 “지능적인” 에이전트.  
  - 하지만 실제 Pacman 게임(ghost 포함)에서는 **승률 0.02 내외**로 한계 확인.
- **Autograder**: 모든 문제(Q1~Q9, Extra Credit 포함) 만점 통과.

---

## 📂 구현 문제별 매핑

- **valueIterationAgents.py**  
  - Q1. Value Iteration  
  - Q4. Prioritized Sweeping Value Iteration (Extra Credit)

- **analysis.py**  
  - Q2. Bridge Crossing Analysis  
  - Q3. Policies  
  - Q7. Bridge Crossing Revisited

- **qlearningAgents.py**  
  - Q5. Q-Learning  
  - Q6. Epsilon Greedy  
  - Q8. Q-Learning과 Pacman  
  - Q9. Approximate Q-Learning

---

## 🚀 DQN Agent 확장

Approximate Q Agent의 한계를 보완하기 위해, **딥러닝 기반 DQN Agent**를 새롭게 추가했습니다.

### 핵심 특징
- **게임 엔진 기반 시뮬레이션 실행**  
  - 기존 Pacman 엔진에 `saveFramePng` 기능 추가.  
  - `run.py`에서 `--dqn`, `--save-frames` 플래그로 데이터 수집 및 학습 가능.
- **Replay Buffer, Target Network** 등 안정적 학습 기법 적용.
- **Medium GridWorld** 환경에서 **0.8 이상의 승률** 달성.
- Pacman 실제 맵에서도 Approx Q Agent 대비 **현저히 높은 성능** 확인.

### 모델 구조: `DuelingMultiHeadDQN`
- **Multi-head encoder**: 상태 벡터를 위치/거리/유령 방향/아이템 개수 등 그룹별로 인코딩.  
- **Fusion trunk**: 그룹별 feature를 통합 후 dropout + layer norm으로 안정화.  
- **Dueling head**: Value stream + Advantage stream 분리 → Q-value 계산.  
- **방어적 fallback**: state_dim이 달라질 경우 일반 MLP로 자동 전환.

---

## 📊 결과 비교

- **Approximate Q Agent**  
  - Feature 기반 근사 Q-learning.  
  - GridWorld에서는 준수한 성능, Pacman 실제 맵에서는 승률 0.02 내외.

- **DQN Agent**  
  - Multi-head + Dueling 구조 적용.  
  - Medium GridWorld에서 **0.8+ 승률**.  
  - Pacman 실제 맵에서도 Approx Q 대비 월등한 성능.

👉 시각화 대시보드에서 학습 곡선, 승률, 점수 분포, 리플레이 영상 확인 가능.

---

## 🛠️ 기술적 기여

- **기존 코드베이스 유지 + 최신화**  
  - Python 최신 버전에 맞게 문법 수정, 구조 개선.  
  - autograder 기준 모든 문제 만점 통과.

- **딥러닝 기반 RL 확장**  
  - Approx Q Agent → DQN Agent로 확장.  
  - RL과 DL 융합 경험.

- **시각화 및 분석 파이프라인 구축**  
  - 학습/테스트 결과를 웹 대시보드로 배포.  
  - 협업 및 성능 분석에 직관적 활용 가능.

- **실무 친화적 실행 구조**  
  - `run.py` 플래그 기반 실행 (`--dqn`, `--save-frames`).  
  - 연구/실험/배포 환경에서 유연하게 사용 가능.

---

## 🎮 프로젝트를 통한 배운 점

이 프로젝트는 단순 과제 구현을 넘어, **미리 작성된 게임 엔진(pacman) 코드 기반 지능정 rl 알고리즘 구현 및 딥러닝 융합 경험**을 보여줍니다.

- 강화학습 알고리즘(Value Iteration, Q-Learning, Approx Q) 직접 구현 경험.  
- PyTorch 기반 DQN 모델 설계 및 최적화.  
- 게임 엔진 수정 및 데이터 수집 파이프라인 구축.  
- 웹 시각화 서버 배포로 결과 공유 및 분석 역량 증명.  
- 기존 코드베이스 유지보수 + 최신화 능력.

---

## 🔗 참고
- UC Berkeley CS188 Pacman AI Project 3 (Reinforcement Learning)  
- Mnih et al., 2015, "Human-level control through deep reinforcement learning"


---
