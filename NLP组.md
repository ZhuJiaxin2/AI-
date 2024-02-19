## 自然语言处理组(NLP)

### 课程资料与论文阅读

nlp领域并不如cv领域那样容易理解，模型学习难度会更大，学习过程中可以做好相关笔记

- 课程资料：
  

​		[吴恩达深度学习课程第五课](https://www.bilibili.com/video/BV1F4411y7BA)(讲到经典模型处会给出相应的论文名称)，你能够从中学会：

- RNN的BPTT手动推导（理解梯度爆炸和梯度消失的原因）

- RNN的两种经典变体：GRU与LSTM

- word embedding的基本知识

- seq2seq结构

- 论文阅读相关资料：

  - 最早提出Attention mechanism的论文: [ Neural Machine Translation by Jointly Learning to Align and Translate (arxiv.org)](https://arxiv.org/abs/1409.0473)
  - Transformer模型经典论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - BERT原始论文：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  - Bert的应用：<https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html#%E7%94%A8-BERT-fine-tune-%E4%B8%8B%E6%B8%B8%E4%BB%BB%E5%8B%99>
  - 希望你能够从中学会BERT的奇妙之处以及其在情感分类任务上是如何进行fine-tune的

- 代码编写相关资料：

  - Pytorch框架的学习：https://pytorch.org/tutorials/beginner/basics/intro.html

  - 预训练模型权重：[Models - Hugging Face](https://huggingface.co/models)

### 学习路线, 工具网站，相关资料推荐

##### [AI Expert Roadmap (am.ai)](https://i.am.ai/roadmap/#note) 

##### [Mikoto10032/DeepLearning: Deep Learning Tutorial (github.com)](https://github.com/Mikoto10032/DeepLearning)

[Transformer非常经典好的blog](https://jalammar.github.io/illustrated-transformer/)

https://github.com/bentrevett/pytorch-sentiment-analysis 

https://github.com/FudanNLP/nlp-beginner 

https://github.com/graykode/nlp-tutorial/tree/master 

http://nlp.seas.harvard.edu/2018/04/03/attention.html

### 代码应用与实践任务其一

分类任务是最简单也最易学习的应用场景，这个代码任务可以分类任务来帮助大家入门NLP这一领域。

（1） IMDB情感分析任务基本上算是一个已经被刷烂掉了的任务，不过也是很好的入门学习任务

1. 从[官网](https://ai.stanford.edu/~amaas/data/sentiment/)上下载数据集
2. 由于文本的离散特性，往往需要先利用一些库对文本进行tokenize
3. 根据tokenize之后的token得到对应的word embedding
4. 从word embedding开始接入常规的模型训练过程
5. 使用基本的textcnn、lstm实现，准确率不做要求

（2）学习最基本的预训练模型，bert和gpt**选其一**（多做加分，~~全选最好~~）

1. 掌握transformer模型原理

2. 自己动手实现bert/gpt以及第一问实现的tokenizer完成IMDB分类任务

   * 从头开始训练的bert/gpt最后准确率较差属于正常现象，不必过分纠结调参

3. 使用预训练好的bert/gpt以及它们对应的tokenizer进行finetune，完成IMDB分类任务

   * 调用的库不做要求，强烈建议使用transformers库

   * [transformers库官方文档](https://huggingface.co/docs/transformers/training) ，官方文本分类[代码参考](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/)
   * 不能直接使用已经在IMDB上finetune好的模型权重
   * 开源的gpt预训练权重用的别较多的是gpt2，跟gpt模型上没有太大区别

4. 比较bert和gpt的区别

说明：

* 我们不做准确率要求，但希望你在使用非finetune方法（textcnn/lstm/自己实现的bert/gpt）最高能达到90%或者使用finetune方法能达到92%，**前者达到92%或者后者达到95%的正确率可直接获得面试资格**（此项请于截止日期前将结果提交给相关人员核验以确定直通通道资格）
* **没做完或者准确率很低也没关系！**我们希望你每个完成的部分都有完整可运行的代码，**态度最重要**



### 代码应用与实践任务其二

以[GPT-3](https://openai.com/blog/gpt-3-apps)、[Switch Transformer](https://arxiv.org/abs/2101.03961)为代表，布局大模型已成为世界性趋势。未来，人工智能大模型时代即将到来，大模型会形成类似“电网”的基础设施，为社会提供智力能源。

（1）大模型时代首先需要了解大模型，同时当前的大模型喷涌，因此了解大模型的原理和优劣十分重要

1. 理解大模型的原理，同时阐述出来（建议从底向上，从模型架构到训练数据）
2. 调研当前的大模型，比较不同大模型的优劣（不仅是外国的，中国也有[智源“悟道”](https://www.baai.ac.cn/portal/article/index/cid/49/id/518.html)，[复旦“MOSS”](https://github.com/OpenLMLab/MOSS)等非常优秀的大模型）
3. 剖析大模型中的训练技巧，自顶向下理解NLP任务（包括prompt，finetune等）
4. 将上面的内容进行整理

（2）同时，当前是一个开源的时代，虽然离人人都有自己的大模型还很遥远，但是现在我们正有机会使用并训练自己的大模型

- 请首先选择一个大模型进行部署。
  - 参考大模型有[ChatGLM](https://github.com/THUDM/ChatGLM-6B)，[ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)，[MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4)，[Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
  - 建议卡至少10G显存以上
- 进一步选择一个细分主题（即选择一个数据集进行调整）对其进行微调，让其成为一个领域的专家
  - 参考大模型同上
  - 建议卡至少16G显存以上
- 具体的微调方法可以自行寻找，很容易找到

### 要求

- 对于**课程资料与论文阅读**部分，我们仅提供一些NLP领域初期非常经典的工作，不进行考核，但是建议未接触过NLP领域的兄弟们还是看一下。
- 对于**学习路线, 工具网站，相关资料推荐**部分，我们仅提供参考。

- 对于**代码应用与实践任务其一**，适合对于NLP领域比较陌生的兄弟，任务量虽然多，但是对于nlp会有比较底层的了解，对于对NLP领域有一定代码量或者做过相关任务的兄弟不建议选择；对于**代码应用与实践任务其二**，适合于对NLP领域有一定了解的兄弟，通过大模型的部署训练和代码学习，可以从较高层次直接查看NLP的各个任务，相当于直接查看NLP这本大书，同时比较联系时代。总之，在其中选择一个任务完成即可。
- 如果其中某项没有完成，也请不要气馁，面试过程中并不会刻意为难大家，只要展现出你的学习即可
- 详细信息可以询问杨明欣（QQ：1411477833），对于资源要求较高的任务，相信大家都各有方法，提供参考获取方法[autodl资源解决方案](https://www.autodl.com/)，或者联系我想办法。
