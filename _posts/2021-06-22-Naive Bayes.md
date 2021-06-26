---
layout: post
title:  "Naive Bayes"
date: 2021-06-22
use_math: true
categories: [Machine Learning]
project: NA
---

### Naive Bayes 개요

> - Machine Learning (머신러닝)
> - Supervised Learning (지도학습)
> - Classificatin (분류)



### Naive Bayes 특성

> - Naive : 데이터의 Feature들이 서로 독립임을 가정
> - Bayes정리 이용
> - 확률 기반 분류 기법



### Naive Bayes의 작동 방식

시험데이터 (X)의 P(Y = yes \| X)와 P(Y = no \| X)를 계산해서 확률이 큰 Y를 시험데이터의 target으로 분류한다. 즉 계산결과 P(Y = no \| X) 의 확률이 더 크다면 시험데이터는 Y = no로 분류한다. 이를 위하여 통계학에서의 Bayes정리가 이용된다.



### Bayes 정리

Bayes정리는 확률을 기반으로 새로운 정보가 주어질때마다 확률을 조정하여, 특정결과에 대한 확률을 계산하는 이론이다. (두 확률 변수의 사전 확률과 사후 확률 사이의 관계를 나타내는 **정리**다. **베이즈** 확률론 해석에 따르면 **베이즈 정리**는 사전확률로부터 사후확률을 구할 수 있다. - wiki)

&nbsp;
$$
P(A|B)=\frac{P(B|A)\cdot P(A)}{P(B)}
$$
&nbsp;

[베이즈 정리 이해에 도움이 되는 글](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=vinci22c&logNo=220346570972)




### 예시를 통한 Naive Bayes의 이해

다음과 같은 데이터가 있을때 Naive Bayes가 어떠한 과정을 통해 분류를 진행하는지 알아보자

<center><img src="{{ "/images/NaiveBayes_ex.png" | absolute_url }}" width = 'auto' height = 'auto' alt="" /></center>

<center><img src="{{ "/images/NaiveBayes_ex_y.png" | absolute_url }}" width = 'auto' height = 'auto' alt="" /></center>

- 위에서 언급하였듯이 시험 데이터의 feature들을 이용하여 아래 두확률을 계산한 후 보다 더 큰 확률의 Y를 선택하게된다.

> $$
> P(Y=NO|X) \to P(Default=No|Home=No,Marital=Married,Annual=120K)\\
> P(Y=YES|X) \to P(Default=YES|Home=No,Marital=Married,Annual=120K)
> $$

- 이때 위 두확률을 계산하기 위해서 Bayes정리를 이용한다.

> $$
> P(Y=NO|X) = \frac{P(X|Y=NO)\cdot P(NO)}{P(X)} \qquad P(Y=YES|X) = \frac{P(X|Y=YES)\cdot P(YES)}{P(X)}
> $$

- 학습 데이터를 이용하여 위 공식의 분자항을 먼저 계산한다.

>$$
>P(X|Y=NO)=P(Home=NO|NO)*P(Marital=Married|NO)*P(Annual=120K|NO)\\
>P(X|Y=YES)=P(Home=NO|YES)*P(Marital=Married|YES)*P(Annual=120K|YES)
>$$

- 위 식에서 $P(Home=NO\|NO\;or\;YES)$와 $P(Marital=Married\|NO\;or\;YES)$는 명목형 임으로 학습데이터를 이용하여 다음과 같이 쉽게 추정할 수 있다.

> $$
> P(Home=NO|NO) = \frac{4}{7}\qquad P(Home=NO|YES)=\frac{3}{3}\\
> P(Marital=Married|NO) = \frac{4}{7} \qquad P(Marital=Married|YES) = \frac{0}{3}
> $$

- 단 $P(Annual=120K\|NO\;or\;YES)$의 경우 연속적인 실수값 이기에 정규분포를 이용하여 계산한다.

> $$
> P(Annual=120K|NO) \to \frac{1}{\sqrt{2\pi\sigma}}exp[-\frac{(x-\mu)^2}{2\sigma^2}]\\
> \;\\
> \;\\
> \begin{align*}
> \mu_{no} &= \frac{125+100+70+\cdots+75}{7}=110\\
> \sigma_{no}^2 &=\frac{(125-110)^2+\cdots+(75-110)^2}{7-1} = 2975\\
> \sigma_{no} &= \sqrt{\sigma^2} = 54.54\\
> \end{align*}
> \;\\
> \;\\
> \;\\
> P(Annual=120K|NO) \to \frac{1}{\sqrt{2\pi*54.54}}exp[-\frac{(120-110)^2}{2*2975}] = 0.007193\\
> $$

&nbsp;

> $$
> P(Annual=120K|YES) \to \frac{1}{\sqrt{2\pi\sigma}}exp[-\frac{(x-\mu)^2}{2\sigma^2}]\\
> \;\\
> \;\\
> \begin{align*}
> \mu_{no} &= \frac{95+85+90}{3}=90\\
> \sigma_{no}^2 &=\frac{(95-90)^2+(85-90)^2+(90-90)^2}{3-1} = 25\\
> \sigma_{no} &= \sqrt{\sigma^2} = 5\\
> \end{align*}
> \;\\
> \;\\
> \;\\
> P(Annual=120K|YES) \to \frac{1}{\sqrt{2\pi*5}}exp[-\frac{(120-90)^2}{2*25}] = 1.2 \times 10^{-9}
> $$

- 위의 3가지 요소(Feature (X))를 이용하여 $P(X\|Y=NO\;or\;YES)$의 확률을 계산한다.

>$$
>\begin{align*}
>P(X|Y=NO) &= \frac{4}{7} \times\frac{4}{7} \times0.007193 = 0.002349\\
>P(X|Y=YES) &= \frac{3}{3} \times \frac{0}{3} \times 1.2\times10^{-9} = 0
>\end{align*}
>$$

- Bayes 정리를 이용하여 시험데이터의 target(Y)이 YES 혹은 NO일 확률을 계산한다. 

> $$
> P(Y=NO|X) = \frac{P(X|Y=NO)\cdot P(NO)}{P(X)} = \frac{0.002349 \times \frac{7}{10}}{P(X)} = \frac{0.001644}{P(X)}\\
> \;\\
> \;\\
> \;\\
> P(Y=YES|X)= \frac{P(X|Y=YES)\cdot P(YES)}{P(X)} = \frac{0\times\frac{3}{10}}{P(X)} = \frac{0}{P(X)}\\
> \;\\
> \;\\
> \;\\
> P(Y=NO|X) > P(Y=YES|X)
> $$

- 최종 결과를 보면 NO일 확률이 YES일 확률보다 크기에 최종적으로 target은 NO로 분류된다.



### m-Estimates

- 위의 예시에서 Y = Yes인 샘플 중 Martial = Married가 없기 떄문에 아래의 확률은 0이 된다. 이와 더불어 만약 Y = Np인 샘플 중에서도 Martial = Married가 없다면, Martial = Married의 특성을 갖고 있는 시험 데이터는 분류할 수 없게된다(두 확률이 모두 0이기 때문).

> $$
> P(Martial=Married|Y=YES) = \frac{0}{3} \;\to\; p(X|Y=YES)=\frac{3}{3}\times\frac{0}{3}\times1.2\times10^{-9}=0
> $$

- 이 문제를 보완하기 위해서는 m-Estimates방법을 사용한다. (m과 p를 사용하여 조건부 확률 계산식을 조정)
  - m : the equivalent sample size (arbitrary)&nbsp;&nbsp;&nbsp;&nbsp;(아래에서 m = 3을 적용)
  - p : a prior estimate (arbitrary)&nbsp;&nbsp;&nbsp;&nbsp;(아래에서  Y=No : p = 2/3, Y=Yes : p = 1/3을 적용)

> 
> $$
> P(Martial = Married|Y=YES) = \frac{0+mp}{3+m}\\
> P(Martial = Married|Y=YES) = \frac{0+3\cdot\frac{1}{3}}{3+3} = \frac{1}{6} \;\;\to\;\; P(X|Y=YES)=\frac{4}{6}\times\frac{1}{6}\times1.2\times10^{-9} = 1.33\times10^{-10}
> $$

- ps) [Laplace 혹은 Lidstone smoothing 기법도 있다.](https://scikit-learn.org/stable/modules/naive_bayes.html)



### Scikit Learn 상의 Naive Bayes

**Gaussian NB** : 연속적인 모든 데이터

**BernoulliNB** : 이진 데이터

**MultinomialNB** : 빈도 데이터 (카운트 기반)



### Practice (python)

[Iris data](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/master/Naive_Bayes/NBayes(iris).py)

[Income data Case1 (Gaussian + Multinomial)](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/master/Naive_Bayes/NBayes(mix-1).py)

[Income data Case2 (Gaussian + Multinomial)](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/master/Naive_Bayes/NBayes(mix-2).py)