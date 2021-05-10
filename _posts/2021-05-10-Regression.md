---
layout: post
title:  "Regression"
date: 2021-05-07
use_math: true
categories: [Machine Learning]
project: NA
---

### Linear Regression 개요

> - Machine Learning (머신러닝)
> - Supervised Learning (지도학습)
> - Prediction (예측)



### Linear Regression 특성

> - OLS (Ordinary Least Square) & TLS (Roral Least Square)
> - 전통적인 예측 기법
> - 회귀 직선을 이용한 예측



### Linear Regression의 작동 방식

데이터를 가장 잘 성명할 수 있는 선을 이용하여 예측을 하는 방식으로, 종속변수 y와 한개 이상의 독립변수 x의 관계를 선으로 표현한 후, 실상의 독립변수를 이 선에 대입하여 예측값 Y를 구한다.



> $$
> y = wx+b
> $$



**OLS(Ordinaty Least Square)**

OLS는 각 데이터 $(x_i,y_i)$ 에서 y축에 평행한 방향으로 회귀직선과 만나는 거리$(d)$의 총 합이 최소(데이터를 가장 잘 설명하는, Loss가 최소화 되는)가 되는 회귀직선을 찾는 방식(MES: Mean Square Error 를 이용한다).

X축이 독립변수이고, y축이 종속변수 일때 유효한 방식이다.



> $$
> \begin{align*}
> &d_i = y_i-\hat{y_i}\quad(개별\;d_i\;계산)\\
> &loss = \frac{1}{n}\sum_{i=1}^nd_i^2\quad(MSE)
> \end{align*}
> $$



**TLS(Total Least Square)**

TLS는 각 데이터 $(x_i,y_i)$와 회귀직선의 수직 거리$(d)$의 총 합이 최소(데이터를 가장 잘 설명하는, Loss가 최소화 되는)가 되는 회귀직선을 찾는 방식(MES: Mean Square Error 를 이용한다).

x축, y축 모두가 독립변수일 때 유효한 방식이다. 기준축이 없음으로 x축,y축 모두 오차를 포함 할 수 있다.



> $$
> \begin{align*}
> &d_i = \frac{|wx_i-y_i+b|}{\sqrt{w^2+1}}\\
> &loss = \frac{1}{n}\sum_{i=1}^nd_i^2\quad(MSE)
> \end{align*}
> $$



**설명력 측정**

회귀 직선을 구한 후, 도출된 직선이 데이터들을 얼마나 잘 설명하는지 알아보기 위하여 $R^2score$(R-square)를 이용한다.

SSE : explained sum of square (예측값과 실제값간 차이$\;\to\;\sum_{i=1}^n(y_i- \hat{y_i})^2$)

SSR: residual sum of square (예측값과 평균간 차이$\;\to\;\sum_{i=1}^n(\hat{y_i}-\bar{y_i})^2$)

SST: total sum of square (편차 제곱 합$\;\to\;\sum_{i=1}^n(y_i- \bar{y})^2$)

> $$
> R^2score = \frac{SSR}{SST}=1-\frac{SSE}{SST}
> $$



위의 방식으로 구해진 $R^2\;score$는 0~1사이의 값을 갖으며, 1에 가까울수록 도출된 회귀선이 데이터를 잘 설명하는 것이다. ($R^2\;score$가 1이라면 모든 데이터가 회귀선 위에 위치함) 



### Practice (python)

[Random data](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/Regression/linear_Regression.py)

[Boston data](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/Regression/linear_Regression(boston).py)

[OLS_TensorFlow](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/Regression/Regression(OLS).py)

[TLS_TensorFlow](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/Regression/Regression(TLS).py)



---------------------------------------------



### Logistic Regression 개요

> - Machine Learning (머신러닝)
> - Supervised Learning (지도학습)
> - Binary classification (이진 분류)



### Logistic Regression 특성

> - Cross Entropy,  Sigmoid
> - 연속적 이진분류를 이용하여 다중분류 가능



### Logistic Regression의 작동 방식

범주형의 종속변수를 갖는 데이터에 대하여, 0~1 사이의 값을 갖는 Sigmoid 함수($y=\frac{1}{1+e^{-x}}$)와 Cross Entropy(CE)를 이용하여 분류를 진행한다. CE를 이용하여(CE를 최소화, 즉 CE = Loss function) 가중치(w)와 편향( b)을 찾아 회귀 선을 완선시킨다. 단 선 상의 결과를 그대로 해석하는 것이 아닌, 0.5를 기준으로 True or False와 같이 범주형 라벨로 해석한다.



> $$
> y = \frac{1}{1+e^{-(wx+b)}}
> $$



### Cross Entropy

Cross Entropy는 정보의 가치를 발생 확률로서 측정하는 것에서 시작한다. 발생확률이 높은 정보인 'Covid-19로 인하여 경제성장률이 저조할 것이다.'가 있을때, 이 정보는 발생확률이 높기에 누구나 다 알고 있는 정보이다. 따라서 이 정보의 가치는 낮다고 할 수 있다. 

Cross Entropy는 Claude Shannon이 제안한 정보량을 계량화하는 방법이다. 확률이 큰 사건이 발생한다는 정보는 정보량이 작고, 확률이 작은 사건의 정보는 정보량이 크다는 아이디어로, 정보량은 확률에 반비례한다고 정의하였다. 또한, 독립적인 두 사건이 발생할 확률은 서로 곱해지지만 정보량은 서로 더해지는 것이 자연스러움으로 정보량을 확률의 역수에 로그를 취한 형태로 제안하였다. 그리고 발생할 수 있는 모든 사건들의 평균을 정보량으로 정의했다. 이렇게 정의된 정보량은 물리학에서 사용하는 볼츠만 엔트로피와 동일한 형태이다.

>$$
>정보량\to log(\frac{1}{p})\to \sum_ip_ilog(\frac{1}{p_i}) = -\sum_ip_ilog(p_i) = H(p)
>$$



- 분포의 형태를 전혀 알 수 없는 어떠한 확률분포 p를 찾는 경우, 잘 알고있는 분포 q를 이용하여 근사적으로 p를 찾을 수 있다.
- 근사분포 q의 정보량과 원 분포 p의 정보량 차이는 $\Delta_i$로 표현할 수 있고 이를 정보의 손실량이라 할 수 있다. 이때 정보 손실량의 기댓값을 KL Divergence라고 한다.
- KL Divergence를 이용하면 두 확률분포 p,q의 유사성을 정량적으로 측정할 수 있다.

>$$
>KL\;Divergence\\
>\;\\
>\Delta_i = -log(q_i)+log(p_i) \to E(\Delta_i) = \sum_ip_i\Delta_i = -\sum_ip_ilog(q_i) + \sum_ip_ilog(p_i) = D_{KL}(p||q)
>$$

- $D_{KL}(p||q)$는 대칭적 구조가 아니다. 이것을 대칭 구조로 변형한 것을 Jensen-Shannon Divergence(JSD)라고 한다.

>$$
>\begin{align*}
>D_{KL}(p||q) &\neq D_{KL}(q||p)\\
>JSD(p||q)&=JSD(q||p)\qquad
>JSD(p||q) =\frac{1}{2}D_{KL}(p||\frac{p+q}{2})+\frac{1}{2}D_{KL}(q||\frac{p+q}{2})\qquad 
>\end{align*}
>$$

- q분포를 이용하여 p분포를 추정하려면, q분포의 파라메터를 추정해 가면서 $D_{KL}(p||q)$가 최소가 되는 지점을 찾으면 된다. 이때 KL Divergence의 첫번째 항은 q와 무관함으로 두번째 항만 최소화 시키면 된다 (두 번째 항을 Cross Entropy라고 한다). 

>$$
>Cross\;Entropy = -\sum_ip_ilog(q_i) = H(p,q) \qquad (D_{KL}(p||q) = H(p)-H(p,q))
>$$







### Practice (python)

[Cancer data](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/D_Tree/DTree(cancer).py)

[Post pruning](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/D_Tree/DTree(post_prune).py)

[Pre pruning](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/D_Tree/DTree(pre_prune).py)