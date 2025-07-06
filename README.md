![Based on Berkeley CS188](https://img.shields.io/badge/Based_on-Berkeley_CS188-blue?style=flat-square)

# 🧠 Pacman AI Agents – UC Berkeley CS188 기반 확장 프로젝트

> 교육용 목적으로 작성된 개인 확장 프로젝트입니다.  
> 원본 라이선스 및 저작권은 UC Berkeley에 있으며, 본 코드에는 해당 고지를 포함하고 있습니다.

---

## 📚 프로젝트 개요

- **기반 과제:** UC Berkeley CS188: Introduction to Artificial Intelligence  
- **목표:** 기존 Pacman 코드 기반에서 다양한 AI 로직(Search, Q‑Learning, Expectimax 등)을 구현  
- **주요 경험:**  
  - 타인의 복잡한 코드베이스 이해 및 구조 분석  
  - 기존 함수/클래스(util.py, GameState 등)를 활용한 실무형 기능 확장  
  - 자료구조 및 알고리즘을 게임 내에 통합 적용

---

## 🧩 구현한 주요 기능

### 1. Search Agent
- **DFS, BFS, UCS, A\***에 기반한 경로 탐색 기능
- `search.py`, `searchAgents.py`에서 구현
- 시각화를 통해 경로의 흐름 및 성능 분석 가능

### 2. Adversarial Agent (Expectimax)
- 유령의 확률적 움직임을 예측하여 최적의 의사결정 수행

### 3. Q-Learning Agent
- 강화학습 기반의 Pacman 행동 최적화
- 보상 구조에 따라 유령 회피 및 음식 수집 능력 향상


---

> 📝 `YOUR CODE HERE` 아래에 작성된 코드들이 모두 **직접 작성한 핵심 AI 로직**입니다.  
> 이 로직이 없는 경우 에이전트는 탐색/예측/학습 기능을 수행하지 못합니다.

---
---

## 🛠️ 기술 스택

| 항목       | 내용                                                       |
|------------|------------------------------------------------------------|
| **언어**       | Python                                                     |
| **알고리즘**   | DFS · BFS · UCS · A\* · Expectimax · Q‑Learning           |
| **구조**       | 함수 기반 + 클래스 기반 모듈화 구현                       |
| **실행 환경**  | 로컬 GUI 기반 Pacman 실행 (원본 코드 필요, 실행 환경 별도 불포함) |

---

## ⚠️ 실행 안내

본 리포지토리는 **전체 Pacman 코드가 포함되지 않아 실행되지 않습니다.**  

---

