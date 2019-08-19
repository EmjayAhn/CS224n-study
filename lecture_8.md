# Lecture 8



## Machine Translation

source language **x** &rarr; target language **y**
	$argmax_{y}P(y|x)$ 			주어진 소스 언어 문장에 대해서 가장 높은 확률로 대응되는 타겟 언어 문장을 찾습니다.
	$argmax_{y}P(x|y)P(y)$	베이즈룰을 사용해서 두 컴포넌트로 분해합니다.
		$P(x|y)$는 번역 모델: 병렬 말뭉치를 이용하여 fidelity를 높이는 방식으로 훈련됩니다. 	 
		$P(y)$는 언어 모델: 한 언어 말뭉치를 이용하여 fluency를 높이는 방식으로 훈련됩니다.

<span style="color:blue">$P(x|y)$와 $P(y|x)$는 동일한 난이도의 작업일텐데 굳이 언어 모델을 더 해서 문제를 디자인 하는 것은 보다 결과물이 개선되어서 일까요?</span>

<span style="color:green">SMT는 이렇게 두 컴포넌트로 작업해야 하지만 NMT는 직접적으로 $P(y|x)$를 모델링해서 유리합니다.</span>



## Alignment

alignment는 소스 언어와 타겟 언어 사이의 단어 수준의 대응입니다.

$argmax_{y}P(x|y)P(y)$에 alignment a를 추가해서 $argmax_{y}P(x,a|y)P(y)$로 나타낼 수 있습니다.

일 대 일 대응 뿐 아니라 many-to-one, one-to-many, many-to-many 대응도 가능하고 일방향이 아닐 수도 있습니다.

<span style="color:blue">MT에서 alignment의 존재 의의는 무엇일까요? 다시 말해 문제를 $argmax_{y}P(x,a|y)P(y)$ 나타내어서 얻는 이점이 어떤 것일까요?</span>



## SMT Decoding (MT의 디코더 부분과도 동일)

$argmax_{y}P(x|y)P(y)$에서 $argmax_{y}$를 어떻게 계산할 수 잇을까요?

​	가능한 모든 y에 대한 확률을 계산한다? &rarr; 지나친 계산 비용
​	휴리스틱 서치 (낮은 확률의 가설은 배제)



## Seq2Seq MT (Test Time)

<p24>

- '조건적' '언어 모델' (디코더가 y의 다음 단어를 예측 & 소스 언어 x가 conditioning) &rarr; $argmax_{y}P(y|x)$ 직접 추론
  	$P(y|x) = P(y_{1}|x)P(y_{2}|y_{1},x)P(y_{3}|y_{1},y_{2},x)...P(y_{T}|y_{1},...,Y_{T-1},x)$

- 인코더 부분에서 마지막 부분은 hidden state 하나로 출력
  	*tf.keras.layers.rnn/lstm/gru에서 return_sequences와 return_state으로 조정 가능합니다.*

- 인퍼런스 혹은 테스트 시 디코더 부분은 Free Running

- 소스 언어와 타겟 언어의 임베딩은 따로.



## Seq2Seq MT (Train Time)

<p27>

- 학습 시 디코더 부분은 Teacher Forcing  (Professor Forcing이나 스케쥴 러닝으로 대체 가능합니다)
  https://arxiv.org/pdf/1610.09038.pdf

- 손실함수 J (**예측단어가 아닌, 정답단어**의 negative log likelihood)는 디코더 타임스텝 t마다 매번 계산 후 평균
  $J = \frac{1}{T}\sum_{t=1}^TJ_{t}$
- end-to-end: 백프롭이 전체 시스템에 대해서 흐릅니다.

*학생질문 1 : 디코더가 문장을 너무 일찍 끝내면 어떻게 하나요? &rarr; Teacher Forcing 하기 때문에 일찍 끝날 일 없습니다.*

*학생질문 2 : end-to-end모델의 인코더와 디코더 부분을 따로 학습할 수는 없나요? &rarr; 한 시스템을 전체적으로 트레이닝 하는 것이 더 깔끔하지만 프리트레이닝은 따로 할 수 있습니다 (예. 디코더로 언어모델링  먼저 학습)*

*학생질문 3 : 소스 언어 문장과 타겟 언어 문장 길이는 항상 고정되어 있어야 하나요? &rarr; 말뭉치에서는 당연히 문장길이가 제각각이지만, 학습시에는 배치가 일정한 사이즈의 텐서로 이루어져야 편하기에 배치 중 가장 긴 문장 길이에 맞추어 제로 패딩하거나 할 수 있습니다. **패딩한 부분에서 히든 스테이트를 계산하여 사용하면 안됩니다!***

<span style="color:green">제로 패딩을 함으로써 rnn mask가 해결됩니다. 다시 말해, 인풋이 0인 부분의 경우 mask_zero=True로 세팅하면 tensorflow가 단순히 그 전의 히든 스테이트를 복사해서 넘기게 됩니다.</span>

*학생질문 4 : 소스 언어와 타겟 언어는 항상 1:1 대응이어야 하나요? &rarr; 조경현 교수님 강의에서도 멀티 언어가 나와서 레고처럼 디코더와 인코더를 결합해서 사용해봤다고 하는데요, 아비님은 이런 모델의 경우 훈련에 주의가 필요하다고 합니다.*

*학생질문 5 : 단어 임베딩도 동일한 말뭉치에서 훈련시켜야 하나요? &rarr; 프리 트레인된 단어 임베딩을 사용하거나, 파인 튜닝하거나, 아예 새로 트레이닝하는 것도 가능합니다.*



## Decoding Search

#### Greedy Decoding

디코더 타임 스텝 t에서, 주어진 조건 하 (그 전 인퍼런스) 가장 확률이 높은 단어를 고릅니다.

- 한 번 실수가 치명적일 수 있습니다.

#### Exhaustive Decoding

앞서 나온 모든 가능한 y에 대한 확률을 찾는 것과 동일한 작업이며 실질적을 불가능.

#### Beam Search Decoding

<p32>

가장 로그 확률 점수가 높은 k개의 부분 번역(가정)을 추적합니다. (k는 보통 5에서 10)
	$score(y_{1},...,y_{t}) = logP_{LM}(y_{1},...,y_{t}|x)=\sum_{i=1}^tlogP_{LM}(y_{i}|y_{1},...,y_{i-1},x)$

- k개의 가장 높은 확률의 단어를 보존하는 첫 스텝을 제외하고는 k^2 개의 단어의 확률을 비교하고 그 중 가장 높은 스코어의 k 개의 단어만 남깁니다.

- 추적 중인 k개의 시퀀스에서 EOS가 각기 다른 타임스텝에서 발생할 수 있습니다.
  완성된 시퀀스는 보존해두고 나머지 k-1개의 시퀀스로 서치를 진행합니다.
- 빔 서치는 미리 정해둔 T에서 무조건 끝내거나, k 개의 시퀀스 중 N 개가 EOS를 생성하면 종료할 수 있습니다.
- 스코어는 시퀀스 길이로 노말라이즈 해줍니다. $\frac{1}{t}\sum_{i=1}^tlogP_{LM}(y_{i}|y_{1},...,y_{i-1},x)$

*학생질문 1 : 왜 애초에 시퀀스 길이로 노말라이즈한 스코어로 빔 서치하지 않나요? &rarr; 빔 서치 진행 중일 때는 어차피 모든 시퀀스의 길이가 같아서 노말라이즈 하는 것과 동일한 효과가 납니다.*

<span style="color:green">빔서치를  실제로 구현하려면 디코더의 infer 함수를 autoregressive하게 call 할 때 , 위와 같이 빔 서치 종료조건으로  while 룹을 구성해서 k 개의 시퀀스를 추적할 수 있을 것 같습니다. </span>



## BLEU SCORE

n-gram precision에 대한 유사도 점수 (짧은 번역은 페널티) 



## Attention

<p58>

#### Problem with Vanilla RNN

하나의 hidden state로 전체 문장을 나타내려면 information bottleneck이 발생합니다.

RNN구조가 벌써 리소스를 많이 사용하기에 hidden state의 사이즈를 키우는 것도 곤란합니다.



#### Seq2seq with Attention

<p61>

1 각 디코더 타임 t에서 query로 어느 인코더 스텝의 value가 중요한지 질문합니다. 
	그 중요도는 attention score (스칼라)로 나타냅니다. (예. 인코더 히든 스테이트와 디코더 히든 스테이트의 점곱)
2 인코더 스텝 개수 만큼의 attention score에 소프트맥스를 해서 attention distribution을 나타냅니다. 
3 그 값으로 인코더 히든 스테이트의 weighted sum을 구해서 디코더 타임 t의 최종 히든 스테이트를 구합니다.

<span style="color:green">query와 value는 각각 디코더 히든 스테이트와 인코더 히든 스테이트 그 자체일 수 도 있지만, 그들을 표상하는 어떤 텐서(예. 그들의 리니어 프로젝션)라도 적용가능합니다.</span>



#### Attention Score

위 세 스텝 중 나머지 둘은 변하지 않지만 attention score를 구하는 방법에는 여러가지가 있습니다.

- Dot-product attention: $e_{i} = s^Th_{i}$
- Multiplicative attention: $e_{i} = s^TWh_{i}$
- Additive (Bahdanau) attention: $e_{i} = v^Ttanh(W_{1}h_{i}+W_{2}s)$



<span style="color:green">Multiplicative Attention과 Additive Attention의 경우 인코더 유닛 사이즈와 디코더 유닛 사이즈가 달라도 됩니다.</span>

<span style="color:green">Additive Attention의 경우 제 3의 디멘션을 또 다른 파라미터로 나타낼 수 있습니다.</span>

<span style="color:green">모든 스코어는 상수를 반환합니다.</span>



https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention







