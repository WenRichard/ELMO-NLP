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
            vocab.txt诗句 根据自身的数据生成
        
### 如何使用ELMO"黑匣子"
这部分主要介绍如何将ELMO代码进行复用，迁移到不同领域上  

