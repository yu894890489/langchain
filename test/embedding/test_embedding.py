import array
import datetime

from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
device = None
# 检查是否有 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")  # 使用 CPU
    print("Using CPU")

    # from transformers import pipeline
    # question_classifier = pipeline("text-classification",
    #                                                                 model="huaen/question_detection",
    #                                                                 device="cuda:0") % timeit
    # question_classifier('''What is the meaning of life?''')
now = datetime.datetime.now()
# 加载 GraphCodeBERT 模型和分词器
model_name = "D:/workspace/code/model/graphcodebert-base"
# model_name = "D:/workspace/code/model/bce-embedding-base_v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
# 将模型移动到指定设备
model.to(device)

# 获取向量
def get_embedding(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt").to(device)  # 输入也需移动到同一设备

    # 获取 Embedding
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # 结果移回 CPU
    return embedding
import seaborn as sns
def heatmap(similarity_matrix):
    # 绘制热力图
    plt.figure(figsize=(6, 5))  # 设置图像大小
    sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", cbar=True)

    # 添加标题和标签
    plt.title("Similarity Matrix Heatmap")
    plt.xlabel("Code Snippets")
    plt.ylabel("Code Snippets")

    # 显示图像
    plt.show()

# 测试相似度
def test_similar(embeddings):


    from sklearn.metrics.pairwise import cosine_similarity
    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(np.vstack(embeddings))
    print("Similarity Matrix:")
    print(similarity_matrix)
    # 找到每个代码片段的最相似代码片段（排除自身）
    for i in range(len(similarity_matrix)):
        similarities = similarity_matrix[i]
        most_similar_index = np.argmax(similarities * (np.arange(len(similarities)) != i))
        print(
            f"Code {i + 1} is most similar to Code {most_similar_index + 1} with similarity {similarities[most_similar_index]:.2f}")


def tsne(result):
    result = np.vstack(result)

    # 检查样本数量
    n_samples = result.shape[0]
    print(f"Number of samples: {n_samples}")

    # 设置 perplexity
    perplexity = min(30, n_samples - 1)  # 确保 perplexity 小于样本数量

    # 使用 t-SNE 降维到 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(result)

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y)
        plt.text(x, y, f"Code {i + 1}", fontsize=10)

    plt.title("t-SNE Visualization of Code Snippets")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()


# 示例输入
code_snippets = [
    "查询用户年龄大于30的",
    "def subtract(a, b): return a - b",
    "SELECT * FROM users WHERE age > 30;",
    "SELECT name FROM users WHERE country = 'China';"
]
embeddings = [get_embedding(code).cpu().numpy() for code in code_snippets]
test_similar(embeddings)

tsne(embeddings)