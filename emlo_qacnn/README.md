# ELMO在Answer selection上的应用

## Introduction
本项目的**baseline**是之前复现的一篇paper：[QACNN](https://github.com/WenRichard/CNN-in-Answer-selection)，在此基础结合ELMO进行实验，项目中参数并没有进行太大的改动，基本上和QACNN相同。同时，本项目按照QACNN的实验设置，分别对不同的Margin进行实验，实验结果将以表格形式给出。  

### 环境配置

    Python版本为3.6
    tensorflow版本为1.13
    
### 目录说明
    
    emlo_qacnn文件夹包含的是完成整个问答demo流程所需要的脚本。
        data_helper_emlo.py
            数据处理部分
        model_utils.py
            模型用到的一些工具等
        model_elmo_online.py
            基于tensorflow_hub生成ELMO词向量，需要保证网络情况良好
        model_elmo.py
            基于官方bilm包生成ELMO词向量，ELMo的模型代码发布在github[1]上,我们在调用ELMo预训练模型时主要使用到bilm中的代码
        modelParams文件夹
            elmo_options.json 模型参数文件，需要下载[2]
            elmo_weights.hdf5 模型的结构和权重值，需要下载[2]
            elmo_token_embeddings.hdf5 根据自身的数据生成
            vocab.txt 根据自身的数据生成
        
### 如何使用ELMO"黑匣子"
这部分主要介绍如何将ELMO代码进行复用，快速迁移到不同领域上  
1. data_helper_emlo.py中的Dataset类  
  +  _genVocabFile( )  
   生成词表文件vocab.txt    
  + _genElmoEmbedding( ): 
  调用ELMO源码中的dump_token_embeddings方法，基于字符的表示生成词的向量表示elmo_token_embeddings.hdf5      
  **dump_token_embeddings的各参数介绍：**  
  1.self._vocabFile,  词汇表（训练集，验证集，测试集）  
  2.self._optionFile,  官网下载  
  3.self._weightFile,  官网下载  
  4.self._tokenEmbeddingFile   保存位置  
  
2. qacnn_emlo.py训练阶段  
  + 实例化BiLM对象，得到bilm
  + 调用bilm中的__call__方法生成op对象，得到inputEmbeddingsOp
  + 计算ELMo向量表示，得到elmoInput
  + 定义elmo的batch形式，得到elmo函数

### 实验设置
|Model|CNN share|Dropout|Parameters|Margin|Epoch|MAP|MRR|DATE|  
|-|-|-|-|-|-|-|-|-|    
|QACNN|No|0.5|2115200|0.5|100|0.655|0.673|2019.3.20|  
|QACNN|Yes|0.5|481664|0.5|100|0.684|0.697|2019.3.20|  
|QACNN|Yes|0.5|481664|0.25|100|0.668|0.674|2019.3.20|  
|QACNN|Yes|0.5|481664|0.2|100|0.690|0.695|2019.3.20|  
|QACNN_EMLO|Yes|0.5|476036|0.5|30|0.711|0.729|2019.4.12| 
|QACNN_EMLO|Yes|0.5|476036|0.25|30|0.721|0.735|2019.4.12| 
|QACNN_EMLO|Yes|0.5|476036|0.2|30|0.720|0.733|2019.4.12| 


### 实验分析
ELMO提升了大概3个百分点吧，提升还是挺大的！  

### LINK
[1]https://github.com/allenai/bilm-tf  
[2]https://allennlp.org/elmo  
[[3]ELMO 预训练模型](https://www.cnblogs.com/jiangxinyang/p/10235054.html)

--------------------------------------------------------------
**如果觉得我的工作对您有帮助，请不要吝啬右上角的小星星哦！欢迎Fork和Star！也欢迎一起建设这个项目！**    
**有时间就会更新问答相关项目，有兴趣的同学可以follow一下**  
**留言请在Issues或者email xiezhengwen2013@163.com**

   

