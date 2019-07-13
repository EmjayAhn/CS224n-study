### Main Idea of Word2Vec

각 단어는 주변에 출현하는 단어들로 정의됩니다.

학습과정에서 벡터는 p(주변단어|중심단어)가 높도록 변화됩니다.

비슷한 단어들을 주변에 갖는 단어들은 임베딩 공간에서 서로 가까이 위치하게 됩니다.



### 두 행렬: U와 V 

[이전 질문] 
dL/du와 dL/dv를 따로 구해서 u와  v를 각각 업데이트하면 사실상 네트워크 둘을 트레이닝 하는 것으로 보이는데, 왜 u와 v로 나누는 것이 더 간단한가요?

[대답]
만약 한 행렬만 쓸 경우, 중심단어가 주변단어 후보로 한 번 등장하게 되는데, 2승 텀이 되고 미분 계산이 더 어려워집니다.

모든 단어가 'the'나 'a'와 같이 빈번한 단어에 대한 높은 점곱을 가지게 됩니다.
	해결법 1: PCA 후 첫번째 component를 제거 (빈도를 나타내므로)
	해결법 2: Negative Sampling



### 최적화

$\theta^{new} = \theta^{old} - \alpha\nabla_{\theta} J(\theta)$

보통 SGD로 최적화

```python
while True: 
	window = sample_window(corpus)
	theta_grad = evaluate_gradient(J, window, theta)
	theta = theta - alpha * theta_grad
```



### Sparsity 문제

윈도우 크기가 5일 때, 최대 11개의 단어가 등장, 배치 사이즈가 32이면 최대 출현 단어는 352개가 됩니다. 
	dimension_size * batch_size * num_words = 300 * 32 * 11 = 105,600개 값

반면 업데이트 해야하는 파라미터는 U와 V 각각 dimension_size * vocabulary_size로 2* 300 * 2,000,000, 약 12억개.

##### 해결법1

U와 V에서  해당하는 단어의 행만 업데이트 합니다.

- <span style="color:blue">질문 01: 어떻게 특정 행만 골라서 업데이트 할 수 있나요?</span>
- Debbie : 등장한 row의 gradient를 계산해서 matrix add, subtraction 으로 gradient update한다는 말인 것 같습니다. 
- <span style="color:blue">질문 02: 윈도우 크기가 5일 때 매 배치에 대해서 V는 1 행 U는 5 행 비대칭적으로 학습되지 않을까요?</span>

##### 해결법2

단어 벡터에 대해서 해쉬를 보존합니다.

- <span style="color:blue">질문 03: 이게 무슨 뜻일까요??</span>
- Debbie : hash embedding이라는 게 있는데, feature를 제한된 숫자로 변환해주는 함수를 사용하는 방법이라고 합니다. python dict가 사용하는 hash와는 다른 개념이고, hash embedding이 여러 종류가 있고 Bloom Embedding등이 NLP에서 사용된다고 합니다. 


### Negative Sampling (SG)

문제를 다르게
Softmax: 2백만 단어 &rarr; Binary 로지스틱 회귀: 참 쌍 (주변단어-중심단어) vs. 거짓 쌍(무작위단어-중심단어)

- <span style="color:blue">질문 04: $J(\theta) = \frac{1}{T}\sum_{t=1}^TJ_t(\theta)$ 라면 t는 무엇을 의미하나요? (p13)</span> 

$J_{neg-sample}(o, v_c, U) = -log(\sigma(u_o^Tv_c)) - \sum_{i=1}^k(log(\sigma(u_k^Tv_c)) $

$P(w) = count(w)^{3/4} / Z$ 의 확률에 따라 k개 (보통 15)의 단어를 주변단어 외 단어에서 골라 거짓 쌍을 만듭니다.
빈번한 단어가 더 많이 뽑히되 3/4승 때문에 그 빈도가 상대적으로 플랫하게 됩니다.



### 동시발생 행렬

$X_{ij}$: $word_i$와 $word_j$가 윈도우 내 동시발생한 경우를 카운트합니다.

##### SVD

$X = U\Sigma V^T$ (X가 symmetric이므로 EigenDecomposition과 동일)
	$Sigma$의 특이값의 수를 조정해 차원 축소를 할 수 있습니다.

##### Rohde 트릭 

​	카운트 스케일링(빈번한 단어 capping)을 합니다.
​	윈도우 조정 (유사한 단어를 더 많이 카운트 합니다.)
​	카운트 대신 피어슨 코릴레이션을 사용합니다.

- <span style="color:blue">질문 05: 윈도우 조정은 Word2Vec에도 사용되었다고 하는데 Negative Sampling의 P(w)를 말한 것일까요?</span> 



### Glove

동시발생 확률의 비율(=벡터간 차이)로 의미를 인코딩할 수 있습니다.

방향을 암시하는 단어들의 비율은 커지고, 나머지 관련이 없거나 의미가 일정한 단어의 비율은 1에 가까워집니다.

동시발생 확률의 비율이 임베딩 공간에서 선형적이 되도록 하는 것이 목표입니다.

##### 계산

점곱이 동시발생 확률의 로그가 되도록 하고, 로그 동시발생 확률의 비율을 구하면 자연스럽게 벡터간 차가 나옵니다. 

$w_i \cdot w_j = logP(i|j)$ 

$w_x \cdot (w_a - w_b) = \frac{logP(x|a)}{logP(x|b)}$ (예: a는 '얼음', b가 '증기'이면 주요 x는 고체와 기체)

$J = \sum_{i,j=1}^Vf(X_{ij})(w_i^Tw_j + b_i + b_j -logX_{ij})^2$
	$(w_i^Tw_j - logX_{ij})^2$단어 i와 j 사이의 점곱이 동시발생 확률의 로그와 비슷해야 합니다.
	 $f(X_{ij})$는 $P(w)^{3/4}$와 비슷한 역할을 수행하여 빈번한 단어쌍이 모델에 지나치게 큰 영향을 끼치지 못하도록 합니다.



### 다의어

##### 해결법 1

클러스터링을 통해 bank를 bank1, bank2, ..., bank5로 나눕니다.
	의미간 구분이 불분명한 경우가 있습니다.

##### 해결법 2

$v_{bank} = \alpha_1v_{bank_1} + \alpha_2v_{bank_2} + etc. $ : weighted average (superposition) of sub-meanings
	$\alpha_1 = f_1/(f_1+f_2+...)$

##### Sparsity 결과

스파스 코딩을 통해 의미를 분해할 수 있습니다.



### 평가방식

코사인 유사도

아날로지 빈칸

Word Vector Analogies
	syntactic: superlatives etc.
	semantic: city-state etc.

WordSim353



### 하이퍼 파라미터

차원수: 300

윈도우 크기: 5~10

학습시간: 24 hrs

데이터: Wikipedia



### 
- Debbie 질문 + (위의 질문 1,3에 답변은 https://eda-ai-lab.tistory.com/122?category=706160 참고하였습니다)
- p14 수식 맨 오른쪽에 sigmoid안의 마이너스 이유 좀더 구체적으로?
