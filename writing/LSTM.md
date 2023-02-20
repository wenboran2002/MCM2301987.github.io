3.1

为了解决问题1，我们采用LSTM模型进行时间序列预测，因为题目要求进行区间预测，所以我们采用了一种特殊的序列处理方式，并预测了2023年1月1号到3月1号的人数。接着我们根据wordle解题策略以及生活经验，提出4个可能影响困难模式选择比例以及wordle难度的单词属性，并建立一套量化规则。之后我们绘制了热图并详细分析了热图结果。
在这一小节，我们首先对LSTM进行了介绍(3.1.1),然后介绍了我们建模的具体细节及参数选择(3.1.2),最后我们介绍了单词属性的选择及量化，以及热图分析的结果

To solve problem 1, we used an LSTM model for time series prediction, and because the question required interval prediction, we used a special sequence treatment and predicted the number of people from January 1 to March 1, 2023. Then we proposed four word attributes that might affect the proportion of difficult mode choices as well as wordle difficulty based on wordle solution strategies as well as life experiences, and established a set of quantification rules. After that we drew a heat map and analyzed the heat map results in detail.
In this subsection, we first introduce the LSTM (3.1.1), then introduce the details of our modeling and parameter selection (3.1.2), and finally we introduce the selection and quantification of word attributes and the results of the heat map analysis


3.1.2 Inplementation details:

训练数据集为359,首先训练模型预测下一天数据的能力，训练完后，我们使用历史数据预测第360天的数据，预测后，我们暂时将第360天的数据看做真是数据，放入历史数据中，再用它预测第361天的数据，依次类推，最终预测完后面的数据。在模型训练过程中，我们使用的模型学习率为$10^{-3}$,采用MSE损失，Adam优化算法。
为了得到预测区间，我们采取一种特殊的方法处理序列。以三天为一个短序列，分别用短序列的最大值和最小值替换掉三天的数据，得到时间序列曲线的两条包络线($max-pre,min-pre$)
,分别对极大值平滑处理过的数据和极小值平滑处理过的数据进行LSTM预测，接着对原始序列进行LSTM预测，得到结果如图。我们发现，原始序列预测的结果正好在极大平滑序列预测和极小序列预测之间，符合预期。最终我们得到预测区间为[20734,28653],原始序列预测为$21367$

The training data set is 359 in size. First, the model is trained to predict the data of the next day. After the training, we use historical data to predict the data of the first day. In the process of model training, we used a model learning rate of $10^{-3}$, and we used MSE loss and Adam optimization algorithm.
To get the prediction interval, we take a special approach to the sequence. Take three days as a short series, replace the three days' data with the maximum and minimum values of the short series respectively, and obtain the two envelope lines of the time series curve ($max-pre,min-pre$).
, LSTM prediction was carried out on data smoothed by maximum value and data smoothed by minimum value respectively, and then LSTM prediction was carried out on the original sequence, and the results were shown in the figure below. We find that the original sequence prediction results are right between the maximal smooth sequence prediction and the minimal sequence prediction, which is in line with the expectation. Finally, we get the forecast interval of [20734,28653] and the original sequence forecast of $21,367