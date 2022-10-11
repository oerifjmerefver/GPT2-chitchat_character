# GPT2 for Chinese chitchat 改进

## 项目描述
- 本项目是基于原项目进行了一些极小的改动，原项目地址为[GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)
- 本项目是基于GPT2的中文闲聊机器人，模型实现基于HuggingFace的[transformers](https://github.com/huggingface/transformers)。
- 在生成阶段，使用了Temperature、Top-k Sampling和Nucleus Sampling等，可参考论文[The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)

## 运行环境
python3.6、 transformers==4.2.0、pytorch==1.7.0，python3.7实测可运行

## 项目结构
- data
    - train.txt:默认的原始训练集文件，存放闲聊语料 
    - train.pkl:对原始训练语料进行tokenize之后的文件,存储一个list对象，list的每条数据表示一个多轮对话，表示一条训练数据
- model:存放对话生成的模型
    - epoch40:经过40轮训练之后得到的模型
      - config.json:模型参数的配置文件
      - pytorch_model.bin:模型文件
- vocab
    - vocab.txt:字典文件。默认的字典大小为13317，若需要使用自定义字典，需要将confog.json文件中的vocab_size字段设为相应的大小。
- sample:存放人机闲聊生成的历史聊天记录
- train.py:训练代码
- interact.py:人机交互代码
- preprocess.py:数据预处理代码

## 关于改进
在原来的项目中，如果要机器人回答一些问题，比如"github是什么？"，我们想要的是：
```
q: github是什么？
a：github是一个著名的开源网站，网址为https://github.com
```
如果只是单个问题，也就是max_history_len=1，原项目是完全能够处理的，但是当处理多轮对话时，原项目的结果可能会出现错误，history长度越大，结果越混乱，错误率越高：
```
q：github是什么？
a：github是一个著名的开源网站，网址为https://github/com
q:那你还知道mysql是什么吗？
a：mysql是一款数据管理系统，用于管理关系型数据库
q:很好，那mongodb呢？
a：mysql是一个相当流行的关系型数据库管理系统（错误结果）
```
而且，如果想要给chatbot安排一个角色，让它从角色的角度生成语句，还会出现"角色错误"的情况
```
user：上午好啊
chatbot：上午好
user：你叫什么名字？
chatbot：我的名字是chatbot
user：哦，那你最近过得怎么样，还好吗？
chatbot:你叫什么名字？（错误结果）
```
原因未知，并且修改模型的配置文件config.json中n_layer（隐藏层的数量）、n_embd（嵌入和隐藏状态的维数）、n_head（注意力头数）等调参手段，经验证毫无用处
本项目便进行了一些改进，可能看起来有些奇怪，打个比方，用户的"github是什么？"，使用基于BertTokenizerFast的MyTokenizer类分词器得到的input_ids是这样的:
```
[10575, 3221, 784, 720, 8043]
```
而chatbot的"github是什么？"：
```
[23892, 16538, 14101, 14037, 21360]
```
二者正好差了一个vocab.txt词表的大小，这使得config.json其中的vocab_size是词表大小的两倍，换言之，模型输入的token的id数量翻了一倍
基于此，再对数据集进行预处理，然后训练得到模型，经验证无论是单轮还是多轮效果都很好


### 模型参数简介(详见模型的config.json文件)
- initializer_range: 0.02
- layer_norm_epsilon: 1e-05
- n_ctx: 1024
- n_embd: 384
- n_head: 24
- n_layer: 48
- n_positions: 1024
- vocab_size: 26634
可以给个提示，由于vocab_size翻了一倍，模型的参数会增大很多，这时候想要训练效果较好的模型，同时缩减模型的大小，可以选择将原本的n_embd:768调整为本项目的384或者更小的192，然后适当增加n_layer隐藏层的数量，经我确认{n_embd:384;n_layer:48}和{n_embd:192;n_layer:60}都可以，后者甚至只有200mb不到

## 训练思路
除了MyTokenizer分词器的特性，其它与原项目相同

## 使用方法
### Quick Start
执行如下命令，进行对话
```
python interact.py --no_cuda --model_path model/epoch40 (使用cpu生成，速度相对较慢)
或
python interact.py --model_path model/epoch40 --device 0 (指定0号GPU进行生成，速度相对较快)
```
而且提示，如果用cpu生成，对话轮数越多，cpu占用越大，chatbot应答的时间越长，个人猜测原因是对话过长导致embedding耗费太多。而使用显卡没有这个问题，chatbot应答的时间不会随着对话轮数的改变而改变，唯一的问题是对话轮数越多，占用的显存越大，使用{n_embd:384;n_layer:48}模型当对话轮数到达12以上时，显存占用已经超过3g

###  数据预处理
和原项目相同，在data中创建train.txt，然后使用preprocess.py预处理
```
python preprocess.py --train_path data/train.txt --save_path data/train.pkl
```

### 训练模型
运行train.py,使用预处理后的数据，对模型进行自回归训练，模型保存在根目录下的model文件夹中。
```
python train.py --epochs 40 --batch_size 8 --device 0,1 --train_path data/train.pkl
```
更多的训练参数介绍，可直接看train.py中的set_args()函数中的参数说明

### 人机交互
运行interact.py，使用训练好的模型，进行人机交互，输入Ctrl+Z结束对话
```
python interact.py --no_cuda --model_path path_to_your_model --max_history_len 13（max_history_len最好为奇数，其实应该没有什么影响）
```
关于参数，temperature的数值适当取小比如0.75，结果的精确度确定会提高，topk和topp取默认值即可，尤其是topp最好直接取默认值0，否则chatbot回答的所有结果都将会是一个固定的句子
如果要使用GPU进行生成，则不要调用--no_cuda参数，并且通过--device gpu_id来指定使用哪块GPU。

## Reference
- [GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)
- [The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)
- [transformers](https://github.com/huggingface/transformers)
- [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)
- [DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.xilesou.top/pdf/1911.00536.pdf)