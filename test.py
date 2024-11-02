def vectorized_fik(documents, vocabs):
    # 创建词汇表字典
    vocab_dict = {word: idx for idx, word in enumerate(vocabs)}

    # 初始化稀疏矩阵
    rows, cols, data = [], [], []

    for doc_idx, doc in enumerate(documents):
        word_counts = {}
        for word in doc.split():
            if word in vocab_dict:
                word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in word_counts.items():
            rows.append(doc_idx)
            cols.append(vocab_dict[word])
            data.append(count)

    return csr_matrix((data, (rows, cols)), shape=(len(documents), len(vocabs)))


# csr_matrix 是 scipy.sparse 模块中的一个类，用于存储稀疏矩阵

# np.bincount()
