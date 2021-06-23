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



예시를 통하여 Bayes정리를 이해해보면 다음과 같다.

**어제 A라는 주식을 구매하였는데, 오늘 뉴스를 확인해보니 A에 대한 기사를 확인하였다. 이런경우 A라는 주식이 오를 것에 대한 기대확률은 몃%로 조정되어야 하는가?**



(사전확률)

기사가 등재되기 전 A주식이 오를 것에 대한 기대확률 70%



(사건발생)

기사 등재 이전 주식이 오를것이라는 조건하에서 기사 등재이후 주식이 오를확률을 30%

기사 등재 이전 주식이 떨어질 것이라는 조건하에서 기사 등재 이후 주식이 오를 확률 10%



주식이 오를 것이라는 조건 + 기사 등재 후 주식 상승 확률 = 70% * 30% = 21%

주식이 떨어질 것이라는 조건 + 기사 등재 후 주식 상승 확률 = 30% * 10% = 3%



이떄 기사 확인 후 주식이 오를 확률은 









### Practice (python)

[Random data](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/Regression/linear_Regression.py)

[Boston data](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/Regression/linear_Regression(boston).py)

[OLS_TensorFlow](https://github.com/Hyunjun-Bruce-Lee/ML_study/blob/main/Regression/Regression(OLS).py)