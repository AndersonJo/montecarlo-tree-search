

# Introduction

Montecarlo tree search의 예제를 보여줍니다. 
데모로서 OpenAI의 택시문제를 풉니다. 
어떻게 움직여야 하는지, 승객이 어디 있는지, 목적지가 어디인지 아무것도 가르쳐주지 않습니다. 
단순히 목적지에 데려다주면 점수를 받고, 잘못된 승객을 태우면 점수를 잃도록 만들었습니다. 

# Result

* **[https://youtu.be/CsxUwp94wko](https://youtu.be/CsxUwp94wko)**

![Demo](images/demo.gif)

택시 (노란색)는 승객 (파란색 R)을 탑승시킨뒤에 목적지 (핑크색 B)에 내리면 점수를 얻습니다. 



# Requirements

* Python 3.x
* OpenAI gym
* Numpy
* tqdm



# Training

```
python mcts.py
```

the following screen is an actual training logs. 

```

```
