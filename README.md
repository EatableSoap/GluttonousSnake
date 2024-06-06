# 北京理工大学人工智能2024大作业
运行此项目你需要安装requirement.txt里的相关库

```cmd
# 转到根文件夹下，运行
pip install -r requirements.txt
```



#### 经典的贪吃蛇游戏实现

```
运行Snake文件夹下的Snake.py
操作方式为wasd控制上下左右，双击可冲刺
```

#### 使用监督学习方法，用手指控制贪吃蛇移动

```
├─Finger
│  ├─ dataset.py 数据集打包工具箱
│  ├─ FingerControl.py 输入对应IP即可使用手指识别
│  ├─ FingerModel.py  手指识别设计和训练部分
```



#### 采取进化算法优化的神经网络方式控制贪吃蛇移动

```
├─GA
│  │  GA_Show.py 运行可使用Model\Best文件下的最优模型
│  │  GA_train.py 模型训练
│  │  Net.py 网络设计
│  │  SnakeClass_NoGraph.py 没有视觉效果的训练蛇类
│  │  
│  ├─Model
│  │  ├─All
│  │  └─Best
│  │          1500_2863.pkl
│  │          1500_2863_population.pkl
```



#### 强化学习控制贪吃蛇移动

```
├─RL
│  │  agent.py 智能体设计
│  │  game.py 奖励机制设计和更新
│  │  model.py 模型设计
│  │  show.py 运行即可看最优模型效果
│  │  
│  ├─model
│  │      model_best.7z 记得解压模型
│  │      train_data.pkl 训练过程的一些数据
```

