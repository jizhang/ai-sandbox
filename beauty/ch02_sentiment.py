import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
EMBEDDING_MODEL = 'text-embedding-3-small'

client = OpenAI(
    base_url=os.getenv('OPENAI_BASE_URL'),
    api_key=os.getenv('OPENAI_API_KEY'),
)

def get_embedding(text):
    text = text.replace('\n', ' ')
    resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    epsilon = 1e-10
    cosine_similarity = dot_product / (norm_a * norm_b + epsilon)
    return cosine_similarity


if __name__ == '__main__':
    positive_review = get_embedding('好评')
    negative_review = get_embedding('差评')

    # positive_example = get_embedding("买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质")
    # negative_example = get_embedding("随意降价，不予价保，服务态度差")
    positive_example = get_embedding('这家餐馆太好吃了，一点都不糟糕')
    negative_example = get_embedding('这家餐馆太糟糕了，一点都不好吃')


    def get_score(sample_embedding):
        positive_similarity = cosine_similarity(sample_embedding, positive_review)
        negative_similarity = cosine_similarity(sample_embedding, negative_review)
        return positive_similarity - negative_similarity


    positive_score = get_score(positive_example)
    negative_score = get_score(negative_example)

    print(f'好评例子的评分：{positive_score}')
    print(f'差评例子的评分：{negative_score}')
