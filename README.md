![Based on Berkeley CS188](https://img.shields.io/badge/Based_on-Berkeley_CS188-blue?style=flat-square)

# 🧠 Pacman AI Agents – UC Berkeley CS188 기반 확장 프로젝트

> 교육용 목적으로 작성된 개인 확장 프로젝트입니다.  
> 원본 라이선스 및 저작권은 UC Berkeley에 있으며, 본 코드에는 해당 고지를 포함하고 있습니다.

---

## ✨ Highlights

- 다양한 탐색(Search) · 판단(Adversarial) · 학습(RL) 알고리즘 직접 **설계·구현**  
- 디렉토리별 모듈화로 **재사용성과 가독성** 강화  
- 파라미터 실험 기반의 정책 성향 분석 경험 보유  

---

## 📋 Table of Contents

- [📚 프로젝트 개요](#-프로젝트-개요)  
- [🧩 구현한 주요 기능](#-구현한-주요-기능)  
  - [🔍 Search Agent](#-search-agent-search-searchpy-searchagentspy)  
  - [👾 Adversarial Agent](#-adversarial-agent)  
  - [🤖 Reinforcement Learning Agents](#-reinforcement-learning-agents)  
- [🛠️ 기술 스택 및 실행 환경](#🛠-기술-스택-및-실행-환경)  
- [⚠️ 실행 안내](#-실행-안내)  

---

## 📚 프로젝트 개요

- **기반 과제:** UC Berkeley CS188: Introduction to Artificial Intelligence  
- **목표:** 공식 Pacman 코드베이스 기반의 탐색, 판단, 강화학습 알고리즘 직접 구현  
- **주요 경험:**  
  - 타인의 복잡한 **코드베이스 이해 및 구조 분석 및 확장**  
  - 기존 함수/클래스(`util.py`, `GameState` 등)를 활용한 구조화된 개발 경험  
  - 자료구조 및 알고리즘을 게임 내에 통합 적용  

---

## 🧩 구현한 주요 기능 

### 🔍 Search Agent (`search.py`, `searchAgents.py`)

*Pacman이 다양한 맵에서 목표를 찾아가는 탐색 알고리즘 구현*

#### • depthFirstSearch (DFS)
- **구현 내용:** 자율적으로 스택 기반의 그래프 탐색 구조 구현  
- **핵심 로직:**  
  - 초기 상태 (`getStartState`)를 스택에 푸시  
  - 방문 상태(`visited`) 관리  
  - `getSuccessors()`로 다음 상태 및 방향 수집  
  - 목표 상태 발견 시 `path` 반환  

#### • breadthFirstSearch (BFS)
- **구현 내용:** 큐 기반 너비 우선 탐색 구현  
- **핵심 로직:**  
  - FIFO 큐를 활용한 수준별 탐색  
  - 중복 방문 체크 및 경로 추적  

#### • uniformCostSearch (UCS)
- **구현 내용:** 우선순위 큐 사용으로 비용 기반 최적 경로 탐색 구현  
- **핵심 로직:**  
  - `(state, path, cost)` 튜플 저장 및 비용 우선 큐 삽입  
  - `getSuccessors()` 사용해 누적 비용 반영  
  - 목표 상태 도달 시 최소 비용 경로 제공  

#### • aStarSearch
- **구현 내용:** 휴리스틱(`heuristic`) 함수와 함께 A* 탐색 알고리즘 구현  
- **핵심 로직:**  
  - PriorityQueue(`g + h`) 방식으로 노드 확장  
  - 방문 기록에 비용 비교하여 재방문 방지  
  - Manhattan/Euclidean 또는 사용자 커스텀 휴리스틱 적용 가능  

---

### 🧭 SearchAgent 클래스 작동 방식

- `searchAgents.py`의 `SearchAgent` 클래스 내부에서,  
  `registerInitialState()` 메서드는  
  1. **문제 정의** (`searchType`)  
  2. **검색 함수 호출** (`self.searchFunction(problem)`)  
  3. **검색 결과를 `actions` 리스트에 저장**  
  4. `getAction()` 시 `actions[]`를 순차적으로 반환  

- 다양한 문제(`PositionSearchProblem`, `CornersProblem`, `FoodSearchProblem`)에 동일한 검색 로직 재사용 가능  
- 커스텀 휴리스틱 및 문제 정의를 통해 **재사용성과 가독성** 향상  

---

### 👾 Adversarial Agent (`multiAgents.py`)

*Pacman과 유령 간 경쟁적 상황에서 최적 의사결정을 위한 Minimax, Alpha-Beta, Expectimax 알고리즘 구현*

#### 👻 ReflexAgent
- `YOUR CODE HERE` 아래 부분에 직접 작성한 평가 함수 포함  
- **Food 및 유령까지 거리 계산**하여 식량과 생존을 균형 있게 고려  
- `foodScore`와 `ghostScore` 조합으로 안전성과 효율성 균형 반영  

#### 🎯 MultiAgentSearchAgent 클래스 계층
- **MinimaxAgent**, **AlphaBetaAgent**, **ExpectimaxAgent** 작성  
- 모든 에이전트는 `YOUR CODE HERE` 아래 재귀 루프(`minimaxVal`, `abPruning`, `expectimaxVal`)로 동작  

##### • MinimaxAgent
- **Max 단계 (Pacman)**과 **Min 단계 (유령)** 명확 구분  
- 상태 깊이(`depth`) 및 agentIndex 기반 재귀 탐색  
- 최적 행동 경로(`bestAction`) 직접 계산  

##### • AlphaBetaAgent
- Minimax + **알파-베타 가지치기** 구현  
- `alpha`, `beta` 값 활용해 불필요한 브랜치 탐색 제거  
- 성능 최적화에 초점  

##### • ExpectimaxAgent
- Minimax 대신 **Expectimax** 탐색 구조 구현  
- 불확실한 유령 행동 고려 (행동 확률 = `1 / len(actions)`)  
- 평균 기대값 기반 의사결정 구조 구현  
- `expectimaxVal()` 내부에 상세 주석 포함  

#### 🎯 betterEvaluationFunction
- **Pacman 행동을 정교하게 추정하는 함수 직접 구현**  
- 주요 요소:  
  - 음식 개수, 거리 고려 (`foodDist`, `leftFoods`)  
  - 유령과 거리 및 상태 고려 (`ghostDist`, `scaredGhost`)  
  - 포인트 기반 가중치 조정 (식량 획득 + 보상/패널티)  
  - `leftFoods == 0` 시 +500 보상  
  - 많은 음식 존재 시 페널티 포함  

---

### 🤖 Reinforcement Learning Agents

*MDP 기반 강화학습(Value Iteration, Q‑Learning)으로 정책을 스스로 학습하는 AI 구조 구현*

#### 🔁 ValueIterationAgent (`valueIterationAgents.py`)
- **Value Iteration 알고리즘**으로 MDP 상태 가치 함수 계산  
- 핵심 로직:  
  - `runValueIteration()`에서 반복 수행  
  - 각 상태 가능한 액션 순회하며 Q‑value 계산 → 최대값 선택  
  - **`computeQValueFromValues()`**로 상태-행동 가치 계산  
  - **`computeActionFromValues()`**로 최적 액션 반환  

#### 💡 QLearningAgent (`qlearningAgents.py`)
- **Online Q‑Learning 기반 에이전트 구현**  
- 핵심 요소:  
  - `qvalues` Counter를 통한 상태‑행동 가치 저장  
  - `update()`에서 Bellman 방정식 기반 Q‑value 갱신  
  - `getAction()`에서 ε‑greedy 정책 구현하여 탐험/활용 균형 유지  
  - `computeActionFromQValues()`로 학습 기반 최적 행동 선택  
- **ApproximateQAgent** 구현: feature extractor 활용 Q‑value 근사 학습 구조  

#### 🧪 Analysis (`analysis.py`)
- 학습 실험용 매개변수 조정 경험:  

| 질문        | 조정 내용               | 결과                          |
|-------------|------------------------|-------------------------------|
| question2   | discount=0.9, noise=0.01 | 표준 정책 설정                 |
| question3a–3e | discount, noise, livingReward 변화 | 정책 보수성/공격성/지속성 변화 |
| question8   | 학습률 등 실험 불가 상태 → `NOT POSSIBLE` 반환 | 탐험 없으면 정책 수렴 불가 처리 |

- RL 알고리즘의 **파라미터 변화에 따른 정책 특성** 이해 및 실험 경험 보유  

---

> 📝 `YOUR CODE HERE` 아래 작성된 코드가 모두 **직접 구현한 핵심 AI 로직**입니다.  
> 해당 코드가 없으면 에이전트는 탐색/예측/학습 기능을 수행하지 못합니다.

---

## 🛠️ 기술 스택

| 항목         | 내용                                                       |
|--------------|------------------------------------------------------------|
| **언어**     | Python                                                     |
| **알고리즘** | DFS · BFS · UCS · A* · Minimax · Alpha-Beta · Expectimax · Value Iteration · Q-Learning |
| **구조**     | 함수 기반 + 클래스 기반 모듈화 구현                       |
| **실행 환경**| 로컬 GUI 기반 Pacman 실행 (원본 코드 별도 필요, 실행 환경 미포함) |

---

## ⚠️ 실행 안내

- 본 리포지토리상의 코드는 **전체 Pacman 코드가 포함되어 있지 않아 단독 실행 불가**합니다.  

---
