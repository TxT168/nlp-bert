import pandas as pd

def preprocess():

    all_data = pd.read_csv('data/train.csv', dtype=str)

    # # 删除评论前后空格
    # all_data = all_data.applymap(lambda x: str(x).strip())
    #
    # # 打乱数据-shuffle
    # all_data = all_data.sample(frac=1).reset_index(drop=True)

    # 划分数据集 可以计算一下8:1:1是 6212：777：776
    train_data = all_data.iloc[:6024]
    dev_data = all_data.iloc[6024:6777]
    test_data = all_data.iloc[6777:]

    # 对于训练模型时，BERT内部数据处理时，要求数据集不要表头
    train_data.to_csv('data/train.txt', sep='\t', header=False, index=False)
    dev_data.to_csv('data/dev.txt', sep='\t', header=False, index=False)
    test_data.to_csv('data/test.txt', sep='\t', header=False, index=False)

preprocess()