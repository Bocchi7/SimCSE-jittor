# SimCSE的Jittor复现

## 简介

在自然语言处理中，文本表示是重要的研究主题，好的文本表示需要能编码文本内在的语义特征。在本项目中，我们聚焦于句子粒度的表示，并将文本相似度（Semantic Textual Similarity, STS）任务设置为目标任务。

SimCSE 是基于 BERT 模型微调，使用对比学习的损失函数训练得到的句子嵌入模型。论文中提出的无监督方法中，正样本通过对同一个句子进行两次带 dropout 的前向传播来生成，负样本采用的是同批次中的其它句子。我们在 Jittor 上复现了 SimCSE 模型的**无监督训练版本**，并完成了以下工作：

1. **用 PyTorch 和 Jittor 分别完成原论文方法的复现**。
2. **优化了 Jittor 的训练性能和显存占用**。
3. **迁移到中文数据集进行实验**。
4. **从 BERT (Encoder-Only) 迁移到 GPT-2 (Decoder-Only) 进行实验**。

## 复现方法和额外工作

### Jittor 复现方法和训练效率改进

我们通过使用 jtorch 库、修改 transformers 库、适配 Jittor 的数据集格式，完成了 PyTorch 到 Jittor 框架的迁移。原论文仓库默认采用 CUDA AMP 的 FP16 混合精度框架训练。但由于 Jittor 框架缺乏对 CUDA AMP 混合精度框架的原生支持，我们手动实现了 FP16 混合精度线性层的 Forward 与 Backward 计算过程。

### SimCSE 方法在 GPT2 上的迁移

我们尝试将 SimCSE 中采用的 BERT 架构换成 GPT 架构。我们采用了 GPT2。我们支持平均池化方法和加权池化方法，其中加权池化方法公式如下：
$$
v = \sum_{i=1}^S w_ih_i, \quad {\rm where} \quad w_i = \frac{i}{\sum_{j=1}^S j},
$$
其中$S$为序列长度，$h_i$为各标记的最后隐藏层上的向量，$w_i$为加权池化时所采用的权重，$v$为池化结果。各标记享有的权重和它们所在的位置成正比，保证了能看到越多标记的标记享有越高的权重。我们实现了这种加权池化方法。

## 配置环境教程

注：您可以直接按照下面的说明安装，也可以直接从 [环境和模型权重下载链接](https://cloud.tsinghua.edu.cn/d/a68eb5be2b824fbfb901/) 下载我们配置的 jittor conda 环境。

### 系统环境

- **操作系统**: Ubuntu 20.04.6 LTS x86_64
- **GPU**: RTX3090 * 1
- **GCC 版本**: gcc version 9.4.0

### Jittor 安装

注意：下面的 `python -m` 尽量不要省略，这样能保证安装到的是当前的 python 环境（即安装到当前的 conda env，而不是其他地方）。

```bash
conda create --name SimCSE python=3.7
conda activate SimCSE
sudo apt install libomp-dev

python -m pip install jittor
python -m jittor.test.test_example
```

### SimCSE 环境安装

```bash
# -i 指定用jittor的源， -I 强制重装Jittor版torch
python -m pip install -r requirements.txt -i https://pypi.jittor.org/simple -I
```

**说明：**

1. 理论上，安装 `jtorch` 环境之后，Jittor 就可以无伤兼容 torch 的 API：

   ```bash
   python -c "import torch; print(torch.tensor([1, 2, 3]))"
   ```

   你应当看到输出为 `jt.Var([1 2 3], dtype=int32)`（输出前面可能会跟着一些绿色的信息）

2. 在 `SimCSE-jittor` 目录下测试 transformers 库（我们修改了 transfomers 库，删去了 Jittor 不能直接支持的模型）

   ```bash
   python -c "import torch; import transformers"
   ```

   **不应该**输出类似 `None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.` 的报错信息

   如果出现 `tokenizers` 的报错信息，请重新安装

   ```
   pip uninstall tokenizers
   pip install tokenizers
   ```


<!-- # SimCSE Unsup. 复现

data准备（微调数据集大小~71M）

```
cd data
bash download_wiki.sh
```

加载预训练模型

1. 下载 `export HF_ENDPOINT=https://hf-mirror.com`

   ``

2. 由于 Jittor 的 load 不支持当前的检查点，需要转换一下 `pytorch_model.bin`
   可以用一个标准的 torch 来重新储存 state_dict（注意不是 Jittor 版本）；也可以使用我们转换好的来完成（链接

遇到问题

- pretrained weight 读取
- 数据集处理（dataloader） -->

## 示例训练脚本

> 数据集下载：根据 [Evaluation](#evaluation) 和 [Training](#training) 小节中的说明下载数据集。
>
> 预训练模型权重下载：如果能够访问 huggingface 可以直接运行脚本，脚本会自动下载；如果不能访问 huggingface，可以通过 https://hf-mirror.com 完成预训练权重下载。
>
> 预训练模型权重转换：如果训练脚本在读取权重过程遇到问题，你需要在标准的 PyTorch 环境下使用本仓库的 `transform.py` 进行预训练权重的格式转换。

我们提供了一系列示例训练脚本，例如，`run_unsup_example-FP16.sh`中采用了和官方仓库一样的超参，`run_unsup_example-FP32.sh`是前者将`--fp16`设置去掉的超参。下面给出的各结果都是在这些示例训练脚本下得到的。

我们也有提供评估脚本，如`run_unsup_example_eval.sh`。但这里我们并没有让它们和各示例训练脚本一一对应，所以请记得改`--model_name_or_path`和`--pooler_type`。

## 实验结果示例

[环境和模型权重下载链接](https://cloud.tsinghua.edu.cn/d/a68eb5be2b824fbfb901/) 中保存了我们的 Jittor 复现结果。

### 训练结果复现与效率对比

训练集、验证集、测试集的选取是和原论文一致的。训练集采用的是官方仓库给出的数据集，验证集选用 STS-B 验证集，测试集选用一系列 STS 任务的测试集。


|                | STS-B 验证集 | 训练吞吐率 (it/s) | 显存占用 (GB) |
|----------------|--------------|-------------------|---------------|
| 论文结果       | 82.5         | -                 | -             |
| Jittor-FP32    | 81.7         | 6.62              | 6.16          |
| Jittor-OurFP16 | 83.5         | 8.76              | 4.04          |

下表是我们将训练好的不同模型进一步在 STS 任务上进行测试的结果。可以看出 Jittor 与 PyTorch 的结果基本接近，并且也基本达到了原论文的水平。

|                | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | STS-R | Avg.(test) |
|----------------|-------|-------|-------|-------|-------|-------|-------|------------|
| 论文结果       | 68.40 | 82.41 | 74.38 | 80.91 | 78.56 | 76.85 | 72.23 | 76.25      |
| Jittor-FP32    | 69.34 | 82.07 | 73.54 | 81.57 | 78.47 | 77.68 | 70.01 | 76.10      |
| Jittor-OurFP16 | 67.35 | 80.03 | 73.22 | 82.47 | 77.83 | 77.16 | 70.39 | 75.49      |

### 到中文数据集的迁移

我们在中文语义相似度任务 LCQMC 与 PAWSX 上进行了 SimCSE 模型的训练和评估，来验证其在中文任务上的有效性。

从下表结果可以看出，使用 SimCSE 方法微调中文预训练模型，相比于微调前，仍然能够有效提升模型在文本嵌入任务上的表现。

| 任务   | BERT-base-cls | SimCSE-cls | BERT-base-avg | SimCSE-avg |
|--------|---------------|------------|---------------|------------|
| LCQMC  | 31.81         | 63.55      | 52.54         | 61.75      |
| PAWSX  | 9.87          | 12.37      | 9.39          | 8.66       |

### 从 BERT 迁移到 GPT-2 模型

我们将 SimCSE 模型迁移到 GPT 架构上，并分别测试了使用平均池化和加权平均池化的结果。

从下表结果可以看出，GPT-2 尽管是 Decoder 架构，但仍然可以通过微调，在该任务上取得较好的结果；并且，使用加权平均进行池化，也显著地优于直接平均池化。

|                | STS-B 验证集 | 各 STS 任务测试集均值 |
|----------------|--------------|-----------------------|
| SimCSE 复现结果 | 83.5         | 75.49                 |
| gpt2-small-avg  | 64.5         | 54.17                 |
| gpt2-small-wavg | 76.2         | 65.87                 |
| gpt2-medium-avg | 65.2         | 49.47                 |
| gpt2-medium-wavg | 79.3         | 68.18                 |

---

以下是[官方仓库](https://github.com/princeton-nlp/SimCSE)原本的README内容。

---

## SimCSE: Simple Contrastive Learning of Sentence Embeddings

This repository contains the code and pre-trained models for our paper [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821).

**************************** **Updates** ****************************

<!-- Thanks for your interest in our repo! -->

<!-- Probably you will think this as another *"empty"* repo of a preprint paper 🥱.
Wait a minute! The authors are working day and night 💪, to make the code and models available, so you can explore our state-of-the-art sentence embeddings.
We anticipate the code will be out * **in one week** *. -->

<!-- * 4/26: SimCSE is now on [Gradio Web Demo](https://gradio.app/g/AK391/SimCSE) (Thanks [@AK391](https://github.com/AK391)!). Try it out! -->
* 8/31: Our paper has been accepted to EMNLP! Please check out our [updated paper](https://arxiv.org/pdf/2104.08821.pdf) (with updated numbers and baselines). 
* 5/12: We updated our [unsupervised models](#model-list) with new hyperparameters and better performance.
* 5/10: We released our [sentence embedding tool](#getting-started) and [demo code](./demo).
* 4/23: We released our [training code](#training).
* 4/20: We released our [model checkpoints](#use-our-models-out-of-the-box) and [evaluation code](#evaluation).
* 4/18: We released [our paper](https://arxiv.org/pdf/2104.08821.pdf). Check it out!


## Quick Links

  - [Overview](#overview)
  - [Getting Started](#getting-started)
  - [Model List](#model-list)
  - [Use SimCSE with Huggingface](#use-simcse-with-huggingface)
  - [Train SimCSE](#train-simcse)
    - [Requirements](#requirements)
    - [Evaluation](#evaluation)
    - [Training](#training)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [SimCSE Elsewhere](#simcse-elsewhere)

## Overview

We propose a simple contrastive learning framework that works with both unlabeled and labeled data. Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as noise. Our supervised SimCSE incorporates annotated pairs from NLI datasets into contrastive learning by using `entailment` pairs as positives and `contradiction` pairs as hard negatives. The following figure is an illustration of our models.

![](figure/model.png)

## Getting Started

We provide an easy-to-use sentence embedding tool based on our SimCSE model (see our [Wiki](https://github.com/princeton-nlp/SimCSE/wiki) for detailed usage). To use the tool, first install the `simcse` package from PyPI
```bash
pip install simcse
```

Or directly install it from our code
```bash
python setup.py install
```

Note that if you want to enable GPU encoding, you should install the correct version of PyTorch that supports CUDA. See [PyTorch official website](https://pytorch.org) for instructions.

After installing the package, you can load our model by just two lines of code
```python
from simcse import SimCSE
model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
```
See [model list](#model-list) for a full list of available models. 

Then you can use our model for **encoding sentences into embeddings**
```python
embeddings = model.encode("A woman is reading.")
```

**Compute the cosine similarities** between two groups of sentences
```python
sentences_a = ['A woman is reading.', 'A man is playing a guitar.']
sentences_b = ['He plays guitar.', 'A woman is making a photo.']
similarities = model.similarity(sentences_a, sentences_b)
```

Or build index for a group of sentences and **search** among them
```python
sentences = ['A woman is reading.', 'A man is playing a guitar.']
model.build_index(sentences)
results = model.search("He plays guitar.")
```

We also support [faiss](https://github.com/facebookresearch/faiss), an efficient similarity search library. Just install the package following [instructions](https://github.com/princeton-nlp/SimCSE/wiki/Installation) here and `simcse` will automatically use `faiss` for efficient search.

**WARNING**: We have found that `faiss` did not well support Nvidia AMPERE GPUs (3090 and A100). In that case, you should change to other GPUs or install the CPU version of `faiss` package.

We also provide an easy-to-build [demo website](./demo) to show how SimCSE can be used in sentence retrieval. The code is based on [DensePhrases](https://arxiv.org/abs/2012.12624)' [repo](https://github.com/princeton-nlp/DensePhrases) and [demo](http://densephrases.korea.ac.kr) (a lot of thanks to the authors of DensePhrases). 

## Model List

Our released models are listed as following. You can import these models by using the `simcse` package or using [HuggingFace's Transformers](https://github.com/huggingface/transformers). 
|              Model              | Avg. STS |
|:-------------------------------|:--------:|
|  [princeton-nlp/unsup-simcse-bert-base-uncased](https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased) |   76.25 |
| [princeton-nlp/unsup-simcse-bert-large-uncased](https://huggingface.co/princeton-nlp/unsup-simcse-bert-large-uncased) |   78.41  |
|    [princeton-nlp/unsup-simcse-roberta-base](https://huggingface.co/princeton-nlp/unsup-simcse-roberta-base)    |   76.57  |
|    [princeton-nlp/unsup-simcse-roberta-large](https://huggingface.co/princeton-nlp/unsup-simcse-roberta-large)   |   78.90  |
|   [princeton-nlp/sup-simcse-bert-base-uncased](https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased)  |   81.57  |
|  [princeton-nlp/sup-simcse-bert-large-uncased](https://huggingface.co/princeton-nlp/sup-simcse-bert-large-uncased)  |   82.21  |
|     [princeton-nlp/sup-simcse-roberta-base](https://huggingface.co/princeton-nlp/sup-simcse-roberta-base)     |   82.52  |
|     [princeton-nlp/sup-simcse-roberta-large](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large)    |   83.76  |

Note that the results are slightly better than what we have reported in the current version of the paper after adopting a new set of hyperparameters (for hyperparamters, see the [training](#training) section).

**Naming rules**: `unsup` and `sup` represent "unsupervised" (trained on Wikipedia corpus) and "supervised" (trained on NLI datasets) respectively.

## Use SimCSE with Huggingface

Besides using our provided sentence embedding tool, you can also easily import our models with HuggingFace's `transformers`:
```python
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

# Tokenize input texts
texts = [
    "There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house."
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))
```

If you encounter any problem when directly loading the models by HuggingFace's API, you can also download the models manually from the above table and use `model = AutoModel.from_pretrained({PATH TO THE DOWNLOAD MODEL})`.

## Train SimCSE

In the following section, we describe how to train a SimCSE model by using our code.

### Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.7.1` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.7.1` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```bash
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.7.1
```


Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

### Evaluation
Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval). It evaluates sentence embeddings on semantic textual similarity (STS) tasks and downstream transfer tasks. For STS tasks, our evaluation takes the "all" setting, and report Spearman's correlation. See [our paper](https://arxiv.org/pdf/2104.08821.pdf) (Appendix B) for evaluation details.

Before evaluation, please download the evaluation datasets by running
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

Then come back to the root directory, you can evaluate any `transformers`-based pre-trained models using our evaluation code. For example,
```bash
python evaluation.py \
    --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased \
    --pooler cls \
    --task_set sts \
    --mode test
```
which is expected to output the results in a tabular format:
```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 75.30 | 84.67 | 80.19 | 85.40 | 80.82 |    84.26     |      80.39      | 81.58 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

Arguments for the evaluation script are as follows,

* `--model_name_or_path`: The name or path of a `transformers`-based pre-trained checkpoint. You can directly use the models in the above table, e.g., `princeton-nlp/sup-simcse-bert-base-uncased`.
* `--pooler`: Pooling method. Now we support
    * `cls` (default): Use the representation of `[CLS]` token. A linear+activation layer is applied after the representation (it's in the standard BERT implementation). If you use **supervised SimCSE**, you should use this option.
    * `cls_before_pooler`: Use the representation of `[CLS]` token without the extra linear+activation. If you use **unsupervised SimCSE**, you should take this option.
    * `avg`: Average embeddings of the last layer. If you use checkpoints of SBERT/SRoBERTa ([paper](https://arxiv.org/abs/1908.10084)), you should use this option.
    * `avg_top2`: Average embeddings of the last two layers.
    * `avg_first_last`: Average embeddings of the first and last layers. If you use vanilla BERT or RoBERTa, this works the best. Note that in the paper we reported the average of last layer and the static word embedding; we fixed this to be last and first layer average and it led to better performance. See [this issue](https://github.com/princeton-nlp/SimCSE/issues/285) for a detailed discussion.
* `--mode`: Evaluation mode
    * `test` (default): The default test mode. To faithfully reproduce our results, you should use this option.
    * `dev`: Report the development set results. Note that in STS tasks, only `STS-B` and `SICK-R` have development sets, so we only report their numbers. It also takes a fast mode for transfer tasks, so the running time is much shorter than the `test` mode (though numbers are slightly lower).
    * `fasttest`: It is the same as `test`, but with a fast mode so the running time is much shorter, but the reported numbers may be lower (only for transfer tasks).
* `--task_set`: What set of tasks to evaluate on (if set, it will override `--tasks`)
    * `sts` (default): Evaluate on STS tasks, including `STS 12~16`, `STS-B` and `SICK-R`. This is the most commonly-used set of tasks to evaluate the quality of sentence embeddings.
    * `transfer`: Evaluate on transfer tasks.
    * `full`: Evaluate on both STS and transfer tasks.
    * `na`: Manually set tasks by `--tasks`.
* `--tasks`: Specify which dataset(s) to evaluate on. Will be overridden if `--task_set` is not `na`. See the code for a full list of tasks.

### Training

**Data**

For unsupervised SimCSE, we sample 1 million sentences from English Wikipedia; for supervised SimCSE, we use the SNLI and MNLI datasets. You can run `data/download_wiki.sh` and `data/download_nli.sh` to download the two datasets.

**Training scripts**

We provide example training scripts for both unsupervised and supervised SimCSE. In `run_unsup_example.sh`, we provide a single-GPU (or CPU) example for the unsupervised version, and in `run_sup_example.sh` we give a **multiple-GPU** example for the supervised version. Both scripts call `train.py` for training. We explain the arguments in following:
* `--train_file`: Training file path. We support "txt" files (one line for one sentence) and "csv" files (2-column: pair data with no hard negative; 3-column: pair data with one corresponding hard negative instance). You can use our provided Wikipedia or NLI data, or you can use your own data with the same format.
* `--model_name_or_path`: Pre-trained checkpoints to start with. For now we support BERT-based models (`bert-base-uncased`, `bert-large-uncased`, etc.) and RoBERTa-based models (`RoBERTa-base`, `RoBERTa-large`, etc.).
* `--temp`: Temperature for the contrastive loss.
* `--pooler_type`: Pooling method. It's the same as the `--pooler_type` in the [evaluation part](#evaluation).
* `--mlp_only_train`: We have found that for unsupervised SimCSE, it works better to train the model with MLP layer but test the model without it. You should use this argument when training unsupervised SimCSE models.
* `--hard_negative_weight`: If using hard negatives (i.e., there are 3 columns in the training file), this is the logarithm of the weight. For example, if the weight is 1, then this argument should be set as 0 (default value).
* `--do_mlm`: Whether to use the MLM auxiliary objective. If True:
  * `--mlm_weight`: Weight for the MLM objective.
  * `--mlm_probability`: Masking rate for the MLM objective.

All the other arguments are standard Huggingface's `transformers` training arguments. Some of the often-used arguments are: `--output_dir`, `--learning_rate`, `--per_device_train_batch_size`. In our example scripts, we also set to evaluate the model on the STS-B development set (need to download the dataset following the [evaluation](#evaluation) section) and save the best checkpoint.

For results in the paper, we use Nvidia 3090 GPUs with CUDA 11. Using different types of devices or different versions of CUDA/other softwares may lead to slightly different performance.

**Hyperparameters**

We use the following hyperparamters for training SimCSE:

|               | Unsup. BERT | Unsup. RoBERTa | Sup.      |
|:--------------|:-----------:|:--------------:|:---------:|
| Batch size    | 64          | 512            | 512       |
| Learning rate (base)  | 3e-5 | 1e-5 | 5e-5 |
| Learning rate (large) | 1e-5 | 3e-5 | 1e-5 |


**Convert models**

Our saved checkpoints are slightly different from Huggingface's pre-trained checkpoints. Run `python simcse_to_huggingface.py --path {PATH_TO_CHECKPOINT_FOLDER}` to convert it. After that, you can evaluate it by our [evaluation](#evaluation) code or directly use it [out of the box](#use-our-models-out-of-the-box).



## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Tianyu (`tianyug@cs.princeton.edu`) and Xingcheng (`yxc18@mails.tsinghua.edu.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use SimCSE in your work:

```bibtex
@inproceedings{gao2021simcse,
   title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
   author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
```

## SimCSE Elsewhere

We thank the community's efforts for extending SimCSE!

- [Jianlin Su](https://github.com/bojone) has provided [a Chinese version of SimCSE](https://github.com/bojone/SimCSE).
- [AK391](https://github.com/AK391) integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/SimCSE)
- [Nils Reimers](https://github.com/nreimers) has implemented a `sentence-transformers`-based [training code](https://colab.research.google.com/drive/1gAjXcI4uSxDE_IcvZdswFYVAo7XvPeoU?usp=sharing#scrollTo=UXUsikOc6oiB) for SimCSE.
