
  ------------------------------------------------------------------
  代码文件索引（各 .py 文件开头有更完整的模块说明）
  ------------------------------------------------------------------
  train_eval.py   实验入口：TrainingData 含 test 张量（已上 GPU）但当前只评 train/val；
                  tokenizer_mode / pos_encoding_mode / causal 等见 CLI；按实验分子目录产出；logs/training_log.txt。
  load_dataset.py IMDB CSV、原逻辑 train/val/test 划分、词表、[CLS]+正文+[SEP]、三套 TensorDataset、dataset_to_gpu。
  models.py       手写的 MultiHeadAttention、正弦/RoPE、AttentionClassifier、RNNClassifier、因果 mask。
  visualize.py    四曲线、验证集混淆矩阵、单头注意力热力图（无多头网格、无表格 PNG）。

  一键跑全流程（需 CUDA GPU、imdb_sentiment_data.csv 在路径内）：
    python train_eval.py --mode all
  依次等价于：一次基线训练 + 五组超参扫描 + 三种位置编码对比 + Attention 与 RNN 对比；
  与下文「五个要求」的对应关系见 train_eval.py 模块首段「与 README 的对应」。

  ------------------------------------------------------------------

  深度学习导论作业 3 - 情感分析 

  ⼀、任务说明 
  IMDB电影评论情感分类 是⼀个经典的情感分析数据集，包含IMDB电影评论⽂本，任务是判断评论是正⾯ (positive)还是负⾯(negative)。你可以在  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download 下载对应的csv⽂件。 

  ⼆、实验步骤 
  1. 词表构建 
  将⽂本序列转换为数字序列，需要构建vocabulary。使⽤LLM的预训练tokenizer 
 （如GPT的tokenizer） 映射到从0开始的连续数字序列 

  2. ⽂本向量化 
       截断/填充到统⼀⻓度（建议200） 
       需要<pad>、<unk>，以及 BERT 式 [CLS]、[SEP] 
       注意 ︓在序列右侧填充<pad>；有效序列为 [CLS] + 内容词 + [SEP]，有效⻓度 = 内容词数 + 2（[SEP] 计⼊有效 token） 

  3. 模型构建 - Attention注意力机制

实现位置编码，参考Transformer的sinusoidal位置编码 ︓ 
     x = self.embed(x) + self.pos_encoder.pe 
     实际推理时需要mask掉padding位置。

模型设计 
    可直接调⽤nn.Embedding和nn.MultiheadAttention模块，不强制要求像标准transformer那样使⽤ Encoder- Decoder架构。但是！不能使⽤transformers库直接调⽤现成模型！

     # ⽰例 
     class AttentionClassifier(nn.Module): 
          # TODO:初始化 
          def forward(self, x): 
              x = self.embed(x) + self.pos_encoder.pe # 也可以构造⼀个位置编码类来实现 
              x, attn_weights = self.attn(x, x, x) # 按需添加残差连接和LayerNorm 
              x = x[:, 0, :]  # BERT 分类：取 [CLS] 位置（右 padding 时在第 0 位） 
              x = self.classifier(x) 
              return x 

        输出1个值 ︓⽤F.binary_cross_entropy_with_logits 
        输出2个值 ︓⽤F.cross_entropy 

  4. 第二个模型构建 -  RNN 
使⽤RNN实现同样任务，直接调⽤nn.RNN，⽆需从头实现。⽐较RNN与Attention各自在性能、效率等各⽅⾯的不同之处。 

   三、以下5个要求都要实现：

      1. 超参数的影响  ︓⽐较注意⼒头数，词表⻓度，嵌⼊维度，模型层数，优化器参数，学 习率策略等对模型 性能的影响。⾄少需要⽐较5组超参，每组⾄少有3个不同的参数值。 
      2. ⼿动实现MHA ︓参照Attention Is All You Need的原⽂，不调⽤torch库函数实现多头注意⼒。 
      3. 位置编码分析  ︓有⽆位置编码的性能对⽐ ，并尝试RoPE等其它位置编码⽅式。 
      4. 可视化分析  ︓挑选样本观察不同注意⼒头的attention权重模式差异，绘制注意力矩阵进行可视化。 
      5. ⾃回归MHA ︓实现带因果mask的多头⾃注意⼒，创建下三⾓矩阵mask掉未来的token。 
      6. 本实验在A40显卡（显存44GB）上运行，可以使用加速方法，例如将数据集预加载到GPU等提速方法

   四、评价指标 

  使⽤sklearn.metrics计算  ︓ 
        Accuracy 
        Precision 
        Recall 
        F1-score 

