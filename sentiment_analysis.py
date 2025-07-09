import pandas as pd
import jieba
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# --- 1. 配置与路径定义 ---
# 注册 tqdm 到 pandas，这样在 apply 操作时也能看到进度条
tqdm.pandas()

DATA_PATH = './data/waimai.csv'
SAVED_MODELS_DIR = './saved_models/'
VECTORIZER_PATH = os.path.join(SAVED_MODELS_DIR, 'tfidf_vectorizer.pkl')
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'logistic_regression.pkl')


def create_dirs():
    """创建必要的目录"""
    if not os.path.exists(SAVED_MODELS_DIR):
        os.makedirs(SAVED_MODELS_DIR)


# --- 2. 数据加载与预处理 ---
def load_and_preprocess_data(file_path):
    """加载数据，处理缺失值，并进行中文分词"""
    print("开始加载和预处理数据...")
    
    # 加载数据
    df = pd.read_csv(file_path)
    
    # 查看数据基本信息
    print("\n数据预览:")
    print(df.head())
    print("\n数据信息:")
    df.info()
    
    # 查看标签分布
    print("\n标签分布:")
    print(df['label'].value_counts(normalize=True)) # normalize=True 显示百分比

    # 处理缺失值
    df.dropna(inplace=True)
    
    # 中文分词
    print("\n正在对评论进行分词，请稍候...")
    df['review_cut'] = df['review'].progress_apply(lambda x: " ".join(jieba.cut(x)))
    
    print("\n分词完成，预览分词结果:")
    print(df[['review', 'review_cut']].head())
    
    return df


# --- 3. 特征提取与模型训练 ---
def train_model(df):
    """划分数据集，提取TF-IDF特征，并训练逻辑回归模型"""
    print("\n开始训练模型...")
    
    # 划分数据集
    X = df['review_cut']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # 特征提取 (TF-IDF)
    print("\n提取TF-IDF特征...")

    #with open('baidu_stopwords.txt', 'r', encoding='utf-8') as f:
    #    stopwords = [line.strip() for line in f.readlines()]

    # N-gram特征的处理
    # 方案一：只使用一元词
    # vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7)
    # 方案二：使用一元和二元
    # vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    # 方案三：一元二元三元均使用
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        max_features=15000
        # min_df=10,
        # max_df=0.8
        #stop_words=stopwords  # 传入停用词列表
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF矩阵维度 (训练集): {X_train_tfidf.shape}")
    
    # 训练逻辑回归模型
    #print("\n训练逻辑回归模型...")
    #model = LogisticRegression(max_iter=1000, random_state=42)
    #model.fit(X_train_tfidf, y_train)
    
    # 训练 SVM 模型
    print("\n训练 SVM 模型...")
    model = LinearSVC(random_state=42)
    model.fit(X_train_tfidf, y_train)

    # 训练朴素贝叶斯模型
    #print("\n训练朴素贝叶斯模型...")
    #model = MultinomialNB()
    #model.fit(X_train_tfidf, y_train)

    # 保存模型和向量化器
    print("\n保存模型和向量化器到磁盘...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"模型保存在: {MODEL_PATH}")
    print(f"向量化器保存在: {VECTORIZER_PATH}")

    return model, vectorizer, X_test_tfidf, y_test


# --- 4. 模型评估 ---
def evaluate_model(model, X_test_tfidf, y_test):
    """在测试集上评估模型性能"""
    print("\n开始评估模型...")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n测试集正确率 (Accuracy): {accuracy:.4f}")
    
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))

    if accuracy >= 0.88:
        print("\n恭喜！模型性能已达到通过标准 (≥ 0.88)。")
    else:
        print("\n模型性能未达到标准。")


# --- 5. 主执行函数 ---
if __name__ == '__main__':
    create_dirs()
    dataframe = load_and_preprocess_data(DATA_PATH)
    trained_model, tfidf_vectorizer, X_test_features, y_test_labels = train_model(dataframe)
    evaluate_model(trained_model, X_test_features, y_test_labels)

    # --- 如何使用已保存的模型进行新预测的示例 ---
    print("\n--- 使用已保存的模型进行新预测 ---")
    # 加载保存的模型和向量化器
    with open(MODEL_PATH, 'rb') as f:
        loaded_model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        loaded_vectorizer = pickle.load(f)
        
    # 准备新评论
    new_reviews = [
        "这家店的饭菜味道太棒了，送餐速度也很快，下次还点！",
        "等了一个多小时才送到，到手都凉了，味道也很一般，不会再来了。",
        "中规中矩，没什么亮点，可以填饱肚子。"
    ]
    
    # 对新评论进行分词
    new_reviews_cut = [" ".join(jieba.cut(review)) for review in new_reviews]
    
    # 转换成TF-IDF向量
    new_reviews_tfidf = loaded_vectorizer.transform(new_reviews_cut)
    
    # 进行预测
    predictions = loaded_model.predict(new_reviews_tfidf)
    probabilities = loaded_model.predict_proba(new_reviews_tfidf)

    for review, pred, prob in zip(new_reviews, predictions, probabilities):
        sentiment = "正面" if pred == 1 else "负面"
        print(f"\n评论: '{review}'")
        print(f"预测情感: {sentiment} (标签: {pred})")
        print(f"预测概率: [负面: {prob[0]:.4f}, 正面: {prob[1]:.4f}]")