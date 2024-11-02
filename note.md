确实,在计算 IDF (Inverse Document Frequency) 时,传统公式 log(N/nk) 可能会导致负值,特别是当某个词在大多数文档中出现时。然而,sklearn 中的 TfidfVectorizer 实现做了一些调整,以确保结果始终为非负值。让我解释一下 sklearn 是如何处理这个问题的:

sklearn 使用的 IDF 公式
sklearn 使用的 IDF 公式稍有不同:

IDF(t) = log((1 + n) / (1 + df(t))) + 1

其中:

n 是文档总数
df(t) 是包含词 t 的文档数
为什么这个公式总是产生非负值
分子 (1 + n) 总是大于分母 (1 + df(t)),因为 n ≥ df(t)
log 的结果可能是负值,但最小也只能是 log(1) = 0
最后加 1 确保了结果始终为正
smooth_idf 参数
TfidfVectorizer 还有一个 smooth_idf 参数,默认为 True。这进一步确保了 IDF 值不会变成无穷大(当一个词只在一个文档中出现时)。

归一化
最后,sklearn 通常会对 TF-IDF 向量进行 L2 归一化,这也有助于保持值在合理范围内。

确保所有特征都有非零权重
