4 distribution prediction based on WM-R

在这一小节中，我们提出了一种根据单词本身信息预测wordle尝试次数分布的模型，Wordle MLP-Regressor(WM-R)
这是一种基于Multi-Layer Perception(MLP)的模型,在输入端加入Wordle编码器提取单词本身特征作为输入。在这一小节，我们首先介绍了模型的构成与实施细节，并对"eerie"进行了回归预测(4.1)，然后我们对模型的不确定因素做了解释，并对模型的可靠性进行了分析(4.2)。

In this section, we proposed a model for predicting wordle trial times distribution based on information of words themselves, Wordle MLP-Regressor(WM-R)
This is a model based on Multi-Layer Perception(MLP), in which a Wordle encoder is added to the input side to extract the features of the word itself. In this section, the constitution and implementation details of the model are first introduced, and then we made the regression prediction of word "eerie"  (4.1). Then, the uncertainty factors of the model are explained, and we analyzed the reliability of the model (4.2).

4.1 model and implementation:

我们的模型输入是一个单词，首先经过一个one-hot编码器得到单词的特征图,维数是$(5\times 26)$.5代表wordle的单词由5个字母组成,26代表每个单词的one-hot向量。之后进行flatten,输入建立好的MLP网络，最后接一个regressor,得到结果。

在具体实施过程中,我们搭建了3个隐层，并在每个线性映射后面接入ReLU层，最后得到一个7维的输出，代表尝试次数 (1, 2, 3, 4, 5, 6, X)的百分比。学习过程采用学习率0.01,MSE损失函数，Adam优化器。训练集大小为250,测试集大小为109

我们对单词"eerie"进行预测，得到结果为(0.1%, 5.17%, 19.46%, 29.31%, 25.65%, 15.31%, 4.09%)

Our model input is a word, first through a one-hot encoder to get the feature map of the word, dimension is $(5\times 26)$.'5' represents the wordle word is composed of 5 letters,'26' represents the one-hot vector of each word. Then, with flatten, enter established MLP networks and finally, with regressor, obtain the results.

In the specific implementation process, we set up three hidden layers, and connected ReLU layer behind each linear mapping, and finally got a 7-dimensional output, representing the percentage of attempts (1, 2, 3, 4, 5, 6, X). Learning process using learning rate 0.01,MSE loss function, Adam optimizer.The training set size is 250 and the test set size is 109

To forecast the word "eerie", we get the result for (0.1%, 5.17%, 19.46%, 29.31%, 25.65%, 15.31%, 4.09%)

4.2 uncertainty and reliability analysis 

对于模型的不确定因素，我们首先将测试集分为高频(f-score>5.8)和低频(f-score<=5.8),再分别计算两个测试集的Mean Absolute Error(MAE),分别为0.0351和0.0359,说明单词的频率对模型有一定的影响因素。然后我们将测试集按DL-score分为两部分(>0.55和<0.55>)计算MAE，分别为0.0347和0.0361说明DL-score对模型也有一定的影响.我们模型的MAE始终保持在一个较低的值，说明我们的模型可靠性较强。

For the uncertainties of the model, First, we divided the test set into high frequency (f-score>5.8) and low frequency (f-score<=5.8), and then calculated the Mean Absolute Error(MAE) of the two test sets respectively, which were 0.0351 and 0.0359, indicating that the frequency of words had a certain influence on the model. Then we divided the test set into two parts (>0.55 and <0.55>) according to DL-score to calculate MAE, which were 0.0347 and 0.0361 respectively, indicating that DL-score also had a certain influence on the model. MAE of our model is always kept at a low value, which indicates that our model has strong reliability.


5.classification based on WM2.0

在这一节中，我们首先构建了对单词难度分类的标准(5.1)。之后我们基于WM-x模型，在输入端和网络内部进一步改进，得到升级后的WM2.0-x(5.2,5.3)。我们改进后的WM2.0考虑了单词的属性特征和语义特征，依据section 3提出的重要属性，以及使用了GoogleNews-vectors-negative300预训练好的词向量(5.2),得到了较好的分类效果。之后我们进行了Ablation Study,判断语义信息以及单词属性信息的加入对模型效果的提升。最后我们讨论了模型的准确性(5.4)

In this section, we first construct a criterion for word difficulty classification (5.1). Later, based on the WM-x model, we further improved the input terminal and the network interior to obtain the upgraded WM2.0-x(5.2,5.3). Our improved WM2.0 takes into account the attribute features and semantic features of words, the important attributes proposed in section 3, and the pre-trained word vectors (5.2) of GoogleNews-vectors-negative300 into consideration, and obtains a better classification effect. We then performed Ablation Study to determine whether the addition of semantic information and word attribute information improved the model's performance. Finally, we discussed the accuracy of the model (5.4).


5.2 Word2Vec

Word Vectors are used for semantic parsing, to extract meaning from text to enable natural language understanding. For a language model to be able to predict the meaning of text, it needs to be aware of the contextual similarity of words.The vectors created by Word Embedding preserve these similarities, so words that regularly occur nearby in text will also be in close proximity in vector space. 

下面这张图片解释了word embeddings
的含义
The following picture explains the meaning of word embeddings

![](![](2023-02-19-23-30-19.png).png)
[explanation]Word Embeddings are similarities based on context, which might be gender, tense, geography or something else entirely. The lines shown are just mathematical vectors, so see how you could move ‘across’ in embedding space from “Man” to “Queen” by subtracting “King” and adding “Woman”.


在wordle任务中，我们认为仅仅根据字母本身的排列特征以及单词的属性是不够的，单词的语义特征可能也会影响wordle的答题情况。单词的语义特征结合了单词使用频率和单词组成的信息。比如，一些常用的单词向量距离较近，难度也更加简单。因此我们根据GoogleNews-vectors-negative300预训练好的词向量构建训练集输入。

In the wordle task, we believe that it is not enough to base the answer on the arrangement characteristics of the letters themselves and the properties of the words; the semantic characteristics of the words may also influence the wordle answer. The semantic features of words combine information about the frequency of word usage and word composition. For example, some commonly used word vectors are closer together and lower in difficulty. Therefore we construct the training set input based on the pre-trained word vectors from GoogleNews-vectors-negative300.

5.3 Model:

升级后的WM-x模型由3个并行的MLP组成，各自最后一层隐层输出的特征向量进行加权叠加，再进入一个全连接层得到输出。对于本题的分类问题，我们在输出后面接入一个分类器,得到WM-C模型。3个并行的MLP输入分别为单词的one-hot向量($x_1,dim=130$)，单词的属性向量($x_2,dim=3$)，和word-embedding得到的词向量($x_1,dim=300$)。其中属性向量我们选择了较为重要的3个属性分数(F-score,V-score,DL-score)。每个MLP由3层全连接层组成，每一层后接Leaky_ReLU,在合并特征进入全连接层后，加入Batchnorm层，再进入ReLU层，最后进行Dropout,防止过拟合。
在训练过程中，我们还是选择训练集大小为250,优化算法选择AdamW,learning rate=
$10^{-2}$,weight decay=$10^{-2}$
损失函数我们没有使用WM1.0中的MSE损失，而是使用了SmoothL1Loss,因为我们发现它的训练效果更好。
我们设定batch_size为64，训练了10个epoch

在训练集上的损失和准确率如下图所示。
The upgraded WM-x model consists of 3 parallel MLPs, where the feature vectors output from each last hidden layer are weighted and superimposed, and then enter a fully connected layer to obtain the output. the 3 parallel MLP inputs are the one-hot vector of words ($x_1,dim=130$), the attribute vector of words ($x_2,dim=3$), and the word- embedding of the word vector ($x_1,dim=300$). Among the attribute vectors we chose the more important 3 attribute scores (F-score, V-score, DL-score). Each MLP consists of 3 fully connected layers, each followed by Leaky_ReLU, after merging the features into the fully connected layer, adding the Batchnorm layer, then into the ReLU layer, and finally Dropout, to prevent overfitting.
In the training process, we still choose the training set size of 250, and the optimization algorithm choose AdamW, learning rate=
$10^{-2}$,weight decay=$10^{-2}$
Instead of using the MSE loss from WM1.0, we used SmoothL1Loss because we found it to be more effective for training.
We set the batch_size to 64 and trained 10 epochs

The loss and accuracy on the training set are shown below.


对于"EERIE"的预测结果为(2.0126e-01, 7.5815e-01, 5.7570e-04),分别表示分类为hard,normal,easy的概率，所以得出"EERIE"属于normal类单词

5.4 Evaluation

对于模型的评估指标，我们主要选择了准确率，在测试集上，我们的WM2.0-C在分类任务上的准确率最终达到了62.04%。我们绘制了混淆矩阵，并分析了其他指标，见下图和表



在这之后，为了验证我们模型的合理性，我们进行了消融学习。去掉Semantic Encoder得到WM1,发现准确率下降了约4%,去掉Attributes Encoder得到WM2发现准确率下降了约10%,将Attribute Encoder和Semantic Encoder都去掉得到原始的WM，准确率下降了约13%，证明我们的模型设计是合理的。

5.4 Evaluation

For the evaluation metrics of the model, we mainly chose the accuracy rate, and on the test set, our WM2.0-C ended up with 62.04% accuracy on the classification task. We plotted the confusion matrix and analyzed other metrics, as shown in the following figure and table



After this, to validate the soundness of our model, we performed ablation learning. Removing the Semantic Encoder to get WM1, we found that the accuracy dropped about 4%, removing the Attributes Encoder to get WM2, we found that the accuracy dropped about 10%, and removing both Attribute Encoder and Semantic Encoder to get the original WM, the accuracy dropped about 13%, proving that Our model design is reasonable.

为了分析单词属性对分类结果的印象，我们重新搭建了一个只有属性向量作为输入，并且在输入时将三个属性值都线性映射到[0,1]之间，只有一个隐层的小网络进行训练，并且将第一个特征矩阵提取出来，绘制特征矩阵的热图，从中可以看出3个属性的影响程度的大小，颜色越浅表示特征重要性越低。我们这样的假设是合理的因为F-score,DL-score,V-score都是与难度大小成负相关。三个属性权重的平均值分别为:(1.617542,1.120998,-0.122229)

To analyze the impression of word attributes on the classification results, we reconstructed a small network with only the attribute vector as the input and mapped all three attribute values linearly between [0,1] with only one hidden layer at the input for training, and extracted the first feature matrix and plotted the heat map of the feature matrix, from which we can see the magnitude of the influence of the 3 attributes, with lighter colors indicating that the feature The lighter the color means the feature is less important. Our assumption is reasonable because the F-score, DL-score, and V-score are all negatively correlated with the difficulty level. The mean values of the three attribute weights are:(1.617542,1.120998,-0.122229)