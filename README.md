# 실습 3
기계 학습,  2015년 봄

## Setup
셋업은 실습 1에서와 같을 것입니다. [practical 1 repository](https://github.com/oxford-cs-ml-2015/practical1)을 참고해 주십시오, 그리고 마지막 시간에 알려드린대로 그 스크립트를 실행해 주십시오.

우리는 그래프를 그리기 위해 토치 패키지 `gnuplot`을 사용할 것입니다. 또는 그것을 설치하는 대신 iTorch를 사용할 수도 있습니다. (나중에 시간이 있으면, 집에서 직접 설치해 실행해 보시길 추천합니다).

## Materials
글의 "소개" 부분에서 언급된 예제를 위해 `simple_example.lua`를 보십시오. 그리고 이전 과제의 템플릿을 위해 `practical3.lua`를 보십시오.

우리는 MNIST라 불리는 데이터세트에 있는 손으로 쓴 숫자들을 분류할 것입니다. 그 데이터의 모습은 다음과 같습니다:

![mnist](https://github.com/oxford-cs-ml-2015/practical3/raw/master/mnist.png)

우리가 가진 MNIST 버전에서 각 데이터포인트는 32x32 영상입니다. 제공된 코드는 이것을 전체가 raw 픽셀 값들로 구성된 한 벡터로 바꿀 것입니다. `simple_example.lua`에는, 훈련/시험 세트들에서 어떻게 한 숫자를 보여주는지를 그림으로 보여주는 "UNCOMMENT(주석 해제)"라 쓰인 줄이 하나 있습니다. 

# 이 실습들을 위한 코스 페이지를 보십시오.
<https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/>

