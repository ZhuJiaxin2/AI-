# 总体学习进度(更新至8.29）

>8.28提交版
## 理论学习

* pytorch各部分常用函数
* 循环神经网络
  * 经典rnn
  * rnn的反向传播推导
  * LSTM、GRU
  * word-embedding基本知识
  * seq2seq结构
* 注意力机制
* （大创学得）模型web部署、服务器部署

## 代码实现

### 1.py

	* 基于torch、glove embedding、lstm+encoder-decoder结构、自定义tokenizer，实现imdb情感分析任务
	* 问题：loss在前5个batch计算正常，从第6个batch开始变为nan，且在后续训练中保持此状态

### 2.py

	* 基于torch、自定义tokenizer，在不使用任何transformers库函数的前提下，自己动手实现bert，完成imdb分类任务
	* 问题：总是提示KeyError，推测是词表的问题，或者词表和词元化列表的对应问题。目前还没有时间去解决它，后续解决后会上传更新。

### 3.py

	* 调用transformers库中的BertTokenizer及BertModel，对bert进行finetune，完成imdb分类任务。
	* 问题：我的电脑是mac M1，拥有10G显存，但此代码运行极其缓慢，差不多几分钟一个step foward。我不知道这是否正常。由于运行实在是太缓慢，而且会导致电脑卡住，我没有运行完成过。

### 总结

	* 每项都写了点，但每项都没有最终结果。
	* 反思：模型理论和实际代码没有建立很好的连接，需要多多动手。

>8.29更新
### 1.py
	* 将optimizer更改为SGD后，loss=nan的现象得到缓解。但在十几二十多个batch后仍然变为nan，并保持此状态。
