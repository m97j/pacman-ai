# pacman-ai

# 🧠 Pacman AI Agents – CS188 기반 확장 프로젝트

> 기존 Pacman 게임 코드 기반에서 다양한 AI 에이전트를 구현한 프로젝트입니다.  
> **Search**, **Q-Learning**, **Expectimax** 등을 통해 다양한 맵과 상황에서 Pacman이 스스로 경로를 찾고 유령을 회피하거나 추적합니다.

<br/>

##  프로젝트 개요

- **기반 과제:** UC Berkeley CS188: Introduction to Artificial Intelligence  
- **목표:** 제한된 구조의 기존 게임 환경 내에서 주어진 요구사항에 맞춰 다양한 AI 로직을 구현
- **주요 경험:**  
  - 타인이 만든 복잡한 코드베이스 이해 및 분석  
  - 기존 함수 및 클래스 구조를 활용하여 AI 로직을 기능 확장 형태로 구현  
  - 자료구조 및 알고리즘을 실제 게임 환경에 적용  

<br/>

## 🧩 구현한 주요 기능

###  1. Search Agent
- **DFS, BFS, UCS, A\*** 기반으로 Pacman이 미로를 탐색하며 목표 지점 도달
- `search.py`, `searchAgents.py` 에서 구현
- 각 알고리즘의 동작 경로 시각화 가능

###  2. Adversarial Agent (Expectimax)
- 유령이 확률적으로 움직일 경우를 고려한 의사결정 구현
- 상대의 전략을 고려한 최적 경로 탐색

###  3. Q-Learning Agent
- 강화학습을 이용하여 보상 기반으로 움직임 최적화
- 학습을 통해 시간이 지날수록 유령 회피 및 음식 수집 성능 향상

<br/>

## 🛠️ 기술 스택

| 항목 | 내용 |
|------|------|
| 언어 | Python |
| 알고리즘 | DFS, BFS, UCS, A*, Expectimax, Q-Learning |
| 구조 | 함수 기반 구현 + 클래스 기반 확장 |
| 환경 | 로컬 실행 (GUI 내장) |

<br/>

##  실행 방법

```bash
# 기본 탐색 에이전트 실행 예시
python pacman.py -l mediumMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

# Q-Learning 에이전트 실행
python pacman.py -p QLearningAgent -x 200 -n 201 -l smallGrid
