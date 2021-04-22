---
layout: post
title:  "Decision Tree"
date: 2021-04-22
use_math: true
categories: [Machine Learning]
project: NA
---

### Decision Tree (의사결정나무) 알고리즘 개요

> - Machine Learning (머신러닝)
> - Supervised Learning (지도학습)
> - Classification (분류)
> - Prediction (예측)



### Decision Tree 특성

> - 분석 결과에 대한 근거 파악 가능 
> - 각 Feature가 분류에 영향을 미치는 정도 파악 가능 (Feature의 중요도 파악가능)
> - Feature가 많더라도 중요도가 낮은 Feature는 자동적으로 분류 기준에서 제외됨 (Feature선정의 중요도가 낮음)
> - 과도한 분할로 인하여 과잉적합 문제 가능성이 높음 (Pruning \| Stoping rule 의 방법으로 개선 가능)
> - 트리가 복잡해 질수록 아래 쪽 노드에 포함되는 데이터가 작아짐 (데이터 단편화) 통계적으로 의미 있는 결정 어려움



### Decision Tree의 작동 방식

의사결정 나무는 의사결정 기준을 나무로 형상화한 알고리즘이다. 의사 결정 기준의 경우 불순척도(엔트로피, 지니지수)를 최소화하는 방향으로 결정된다. 

의사결정 기준에 의거하여 순차적으로 데이터를 2분할(데이터의 개수가아닌 개념적 2분할)하며 분류를 진행하고, 별도의 처리를 하지 않을시 모든데이터가 각각 하나의 범주로 분류됨으로 가지치기 혹은 정지규칙을 이용하여 과잉적합을 방지해야 한다.



### Decision Tree에서의 불순 척도

불순척도란 데이터를 분할한 후 데이터들이 얼마나 잘 나뉘어 지었는지를 나타내는 단위이다..

불순척도가 낮을수록 불확실성이 감소하고 이에따라 정보의 획득이 발생된다. 의사결정나무에서는 이러한 정보의 획득을 최대화(불순척도의 최소화)하는 방향으로 분할을 진행한다.

분할전의 불순척도와 분할된 집단에 대한 불순척도의 가중평균을 비교하여 분할하는것이 적합한지 알 수 있다.

일반적으로 불순척도로는 엔트로피와 지니지수가 이용된다.

<center><img src="{{ "/images/entropy_and_jini.png" | absolute_url }}" width = 'auto' height = 'auto' alt="" /></center>



> ### **엔트로피(Entropy)**
>
> - 0 - 1 사이의 값을 갖으며, p = 0.5 (완전한 반반 상태)일 경우 최대의 불순척도를 갖는다.  (그림 참고)
>   
>   
>   
> $$
>   Entropy = -\sum^c_{i = 1}p(i|t)log_2p(i|t)
> $$
> 
>   - **c** = 분류된 범주&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**i** = 개별 데이터의 수&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**t** = 범주내의 데이터 수&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**p(i\|t)** = t개중 i일 확률
>
> 
>
> ##### **엔트로피 계산 예시**
>
> - **A(21개)와 B(17개)로 구분되는 데이터가 있을경우 분할 전의 엔트로피 계산**
>
>   
>
> $$
> \begin{align*}
> Entropy &= -[(21/38)log_2(21/38) + (17/38)log_2(17/38)]\\
> &= 0.9919...
> \end{align*}
> $$
>
> 
>
> - **X = a 라는 분류 기준으로 데이터를 분할한 결과가 다음과 같을때 엔트로피의 계산**
>   
>   - Left  :   A = 11개&nbsp;&nbsp;&nbsp;B = 1개&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Right  :  A = 10개&nbsp;&nbsp;&nbsp;B = 16개 
>   
>     
>
> $$
> \begin{align*}
> Entropy_{\_Left} &= -[(11/12) log_2(11/12) + (1/12) log_2(1/12)] \\
> &= 0.4138...\\
> \\
> Entropy_{\_Right} &= -[(10/26) log_2(10/26) + (16/26) log_2(16/26)] \\
> &= 0.9612...\\
> \\
> Entropy_{\_W-avg} &= (12/38)*0.4138+(26/38)*0.9612\\
> &= 0.7883...\\
> \\
> Information\_Gain_{(X=a)} &= 0.9919-0.7883\\
> &= 0.2036
> 
> \end{align*}
> $$
>
> - 분류 후의 엔트로피가 분류전의 엔트로피보다 0.2096 만큼 작다.
>
> - X = a라는 기준으로 분류를 함으로서 0.2096의 정보를 얻었다(불확실성이 감소했다). 
>
> 
>
> - **Y = b 라는 분류 기준으로 데이터를 분할한 결과가 다음과 같을때 엔트로피의 계산**
>   
>   - Upper : A = 14개&nbsp;&nbsp;&nbsp;B = 7개&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Lower : A = 7개&nbsp;&nbsp;&nbsp;B = 10개
>   
>     
>
> $$
> \begin{align*}
> Entropy_{\_Upper} &= -[(14/21) log_2(14/21) + (7/21) log_2(7/21)] \\
> &= 0.9182...\\
> \\
> Entropy_{\_Lower} &= -[(7/17) log_2(7/17) + (10/17) log_2(10/17)] \\
> &= 0.9774...\\
> \\
> Entropy_{\_W-avg} &= (21/38)*0.9182+(17/38)*0.9774\\
> &= 0.9446...\\
> \\
> Information\_Gain_{(Y=b)} &= 0.9919-0.9446\\
> &= 0.0473
> 
> \end{align*}
> $$
>
> - 분류 후의 엔트로피가 분류 전의 엔트로피보다 0.0532 만큼 작다.
> - Y = b라는 기준으로 분류를 함으로서 0.2096의 정보를 얻었다(불확실성이 감소했다).
>
> 
>
> - X=a 와 Y=b 두가지 분류기준 모두 분류 전에비해 불확실성이 낮기에 두가지 기준 모두 적합하지만, 의사결정나무는 X = a의 기준으로 분할하는것이 더 많은 정보를 얻기에 X = a로 분할하게된다.
>
> 
>
> 
>
> ### 지니 지수(Gini Score)
>
> - 0 - 0.5 사이의 값을 갖으며, p = 0.5 (완전한 반반 상태)일 경우 최대의 불순척도를 갖는다.  (그림 참고)
>   
>   
>   
> $$
>   Gini = 1-\sum^c_{i = 1}p(i|t)^2
> $$
> 
> 
>   - **c** = 분류된 범주&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**i** = 개별 데이터의 수&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**t** = 범주내의 데이터 수&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**p(i\|t)** = t개중 i일 확률
>
> 
>
> **지니 지수 계산 예시**
>
> - **A(21개)와 B(17개)로 구분되는 데이터가 있을경우 분할 전의 지니지수 계산**
>
>   
>
> $$
> \begin{align*}
> Geni &=1 -[(21/38)^2 + (17/38)^2]\\
> &= 0.4944...
> \end{align*}
> $$
>
> 
>
> - **X = a 라는 분류 기준으로 데이터를 분할한 결과가 다음과 같을때 지니지수 계산**
>   
>   - Left  :   A = 11개&nbsp;&nbsp;&nbsp;B = 1개&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Right  :  A = 10개&nbsp;&nbsp;&nbsp;B = 16개 
>   
>     
>
> $$
> \begin{align*}
> Gini_{\_Left} &=1 -[(11/12)^2 + (1/12)^2] \\
> &= 0.1527...\\
> \\
> Gini_{\_Right} &=1 -[(10/26)^2 + (16/26)^2] \\
> &= 0.4733...\\
> \\
> Gini_{\_W-avg} &= (12/38)*0.1527+(26/38)*0.4733\\
> &= 0.3720...\\
> \\
> Information\_Gain_{(X=a)} &= 0.4944-0.3720\\
> &= 0.1224
> 
> \end{align*}
> $$
>
> - 분류 후의 지니지수가 분류전의 지니지수보다 0.1224 만큼 작다.
>
> - X = a라는 기준으로 분류를 함으로서 0.1224의 정보를 얻었다(불확실성이 감소했다). 
>
> 
>
> - **Y = b 라는 분류 기준으로 데이터를 분할한 결과가 다음과 같을때 지니지수의 계산**
>   
>   - Upper : A = 14개&nbsp;&nbsp;&nbsp;B = 7개&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Lower : A = 7개&nbsp;&nbsp;&nbsp;B = 10개
>   
>     
>
> $$
> \begin{align*}
> Gini_{\_Upper} &=1 -[(14/21)^2 + (7/21)^2] \\
> &= 0.4444...\\
> \\
> Gini_{\_Lower} &=1 -[(7/17)^2 + (10/17)^2] \\
> &= 0.4844...\\
> \\
> Gini_{\_W-avg} &= (21/38)*0.4444+(17/38)*0.4844\\
> &= 0.4622...\\
> \\
> Information\_Gain_{(Y=b)} &= 0.4944-0.4622\\
> &= 0.0322
> 
> \end{align*}
> $$
>
> - 분류 후의 지니지수가 분류 전의 지니지수보다 0.0322 만큼 작다.
> - Y = b라는 기준으로 분류를 함으로서 0.0322 의 정보를 얻었다(불확실성이 감소했다).
>
> 
>
> - X=a 와 Y=b 두가지 분류기준 모두 분류 전에비해 불확실성이 낮기에 두가지 기준 모두 적합하지만, 의사결정나무는 X = a의 기준으로 분할하는것이 더 많은 정보를 얻기에 X = a로 분할하게된다.



### 가지치기(Pruning)와 정지기준(Stopping rule)

가지치기와 정지기준은 의사결정나무가 너무 복잡해 지지 않도록(과잉적합) 단순화시는 작업이며, 이는 일반화 특성을 향상시킨다고 할 수 있다.

**사전 가지치기(정지기준)**

Pre-pruning이라고도 불리는 사전 가지치기는 트리의 깊이, 마지막 노드의 최소 데이터수, 불순척도 등의 임계치를 이용한 정지기준으로 사전에 의사결정나무의 전개를 제한 하는것을 의미한다.

**사후 가지치기**

Post-pruning이라고도 불리는 사후 가지치기는 먼저 트리를 최대한으로 전개한 후 (마지막노드의 불순척도가 최소가 되도록), 완전히 전개된 트리를 위쪽방향으로 다듬어가는 절차를 수행(Trimming)하는 것을 의미한다. 일반적으로 Cross Validation 시험 오차가 최소가 되는 분할 수준으로 트리를 줄이게 된다.



### Practice (python)

[Cancer data](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/D_Tree/DTree(cancer).py)

[Post pruning](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/D_Tree/DTree(post_prune).py)

[Pre pruning](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/D_Tree/DTree(pre_prune).py)