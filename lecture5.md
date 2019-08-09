# Lecture 5



## 두가지 언어학적 구조

​	(1) 구-구조 문법
​	(2) 의존 문법



## 구-구조문법: 언어학 분야에서의 주요 접근법

NP (명사구): det + (adj) + noun, PP (전치사구): prep + NP 등  

단어를 기본유닛으로 삼는 여러 종류의 구를 재귀적으로 쌓은 구조로 문장을 해석



## 의존문법: 전산 언어학 분야에서의 주요 접근법

A -> B (B가 A에 의존한다)

단어를 기본유닛으로 삼아 서로간 의존성을 가장 저수준까지 탐색

 

## 의존문법의 불확실성

자연어의 구조는 의미/문맥에 의존하여 경제적이지만 불확실성 발생

​	* 의미를 이해하려면 구조를 알아야하지만 구조를 찾으려면 의미를 이해해야 합니다.

종류

​	전치사구 연결 불확실성
​	호응범위 불확실성
​	형용사적 수식언 불확실성
​	동사구 연결 불확실성



## 의존성 파싱 트리뱅크

디자인 특성

​	ROOT를 더해서 최상부 노드(보통 동사부)가 의존하는 유일한 노드로 작용토록 합니다.
​	acyclic graph

데이터

​	Universal Dependencies: http://universaldependencies.org/

장점

​	룰 기반 표현법에 비해서 재사용이 가능
​	불확실한 구조에 대해서 정/부정 평가가 가능
​	범용성
​	구조의 빈도 계산 가능



## 파싱에 사용하는 정보 (Dependency Conditioning Preferences)

1 두 단어 사이 친밀도
2 의존성 거리
​3 사이에 오는 단어 (의존성은 보통 동사나 마침표를 가로지르지 않습니다.)
​4 헤드의 결합가 (헤드의 양 쪽에 보통 몇 개의 의존부가 나타나는지)

해당 문장에 대한 의존성 파싱을 하지 않은 상태에서 두 단어 사이 친밀도, 의존성 거리, 헤드의 결합가를 어떻게 계산할까 궁금했는데, 아마도 수많은 이진 조건으로 이루어진 피처가 이러한 정보를 표현하는 듯 싶습니다.

**>질문**: 의존 화살표가 크로스하는 경우와 하지 않는 경우 트리구조 상에서 변화가 발생하지는 않습니다. 예를 들면 I will give a talk on bootstrapping tomorrow와 I will give a talk tomorrow on bootstrapping은 트리구조상에서는 완벽히 동일합니다. 그렇다면 왜 이런 크로싱을 이슈로 여길까요?



## Transition-Based Parsing

1 버퍼에서 스택으로 Shift 액션을 통해서 단어를 넘기면서, 
2 스택 내 루트를 제외한 단어 쌍이 발생할 경우, 
	단어 쌍에 대해 LeftArc  Reduction 이나 RightArc Reduction 중 하나를 수행합니다.
	다시 말해, 의존성과 라벨을 의존성 뱅크에 넣으면서 자식 부분을 스택에서 제외합니다.

3 루트와 마지막 단어가 남는 경우,
	루트와 마지막 단어와의 의존성과 라벨을 의존성 뱅크에 넣고 스택을 비우는 동시에 의존성 뱅크를 완성합니다.

모든 가능한 선택을 탐색하거나 다이나믹 프로그래밍을 사용해서 보다 효과적으로 탐색할 수 있습니다.

**>질문**: 이 모델은 '학습'이라는 개념이 있나요? 예를 들면 정답이 없는 unseen 데이터를 받아서 모든 가능한 선택을 늘어놓은 후에는 (정답이 주어지지 않기에) accuracy 계산을 할 수 없는데 가능한 트리 중 가장 좋은 트리를 어떻게 고를 수 있나요?



## MaltParser

인풋

​	수백만 이진 조건 특성	

모델

​	3가지 (혹은 라벨*2+1) 액션  중 어느 작업을 수행할 지 SVM 등의 분류기를 학습

결과

​	작업 시퀀스에 대한 서치를 하지 않아도 상당한 정확도 (빔 서치를 할 경우 정확도 향상)

단점

​	인풋이 sparse, incomplete, expensive



## 평가방법

라벨을 제외하고 의존성만 평가

​	Acc = # of correct dependencies / # of dependencies

라벨까지 평가

​	Acc = # of correct dependencies-labels/# of dependencies-labels



## 분산 표현 (Distributed Representations)

인풋
	단어임베딩: d-차원 벡터 표현 
	POS 임베딩: d-차원 벡터 표현 
	의존성 라벨: d-차원 벡터 표현

토큰 추출 (모든 토큰을 단어, POS, 의존성 라벨을 concat한 3d-벡터로 표현)
	스택단어 1
	스택단어 2
	버퍼단어 1
	스택단어 1의 왼쪽 의존성
	스택단어 1의 오른쪽 의존성
	스택단어 2의 왼쪽 의존성
	스택단어 2의 오른쪽 의존성



## 모델 구조

Input layer: x
	3d-벡터

Hidden layer: h
	h = ReLU(Wx+b1)

Output layer: y
	y = softmax(Uh+b2)
	3차원 확률분포 혹은 (2*|라벨|+1)차원 확률분포