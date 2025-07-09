import pandas as pd
import jieba
import pickle
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import KeyedVectors
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix # 用于拼接稀疏矩阵



# --- 1. 配置与路径定义 ---
# 注册 tqdm 到 pandas，这样在 apply 操作时也能看到进度条
tqdm.pandas()

DATA_PATH = './data/waimai.csv'
WORD_VECTORS_PATH = 'word2vec/light_Tencent_AILab_ChineseEmbedding.bin'
SAVED_MODELS_DIR = './saved_models_fusion/'
VECTORIZER_PATH = os.path.join(SAVED_MODELS_DIR, 'tfidf_vectorizer_best.pkl')
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'best_classifier_fusion.pkl')


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


# --- 3. 特征提取 ---
def extract_features(df):
    print("\n划分数据集并提取TF-IDF特征...")
    X = df['review_cut']
    y = df['label']
    # 使用 stratify=y 保证训练集和测试集中标签分布与原始数据一致
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
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
    
    print(f"特征提取完成。TF-IDF矩阵维度 (训练集): {X_train_tfidf.shape}")
    
    # 保存向量化器，因为它与最佳模型是配对的
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"最优特征的向量化器已保存至: {VECTORIZER_PATH}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test

def load_word_vectors(path):
    """加载预训练的词向量模型"""
    print(f"正在从 {path} 加载预训练词向量，这可能需要几分钟...")
    wv = KeyedVectors.load_word2vec_format(path, binary=True)
    print("词向量加载完成。")
    return wv

def sentence_to_vector(sentence, wv_model):
    """将分词后的句子转换为句子向量（通过词向量平均）"""
    words = sentence.split()
    vectors = [wv_model[word] for word in words if word in wv_model]
    if not vectors:
        # 如果句子中所有词都不在词向量词汇表中，返回一个零向量
        return np.zeros(wv_model.vector_size)
    return np.mean(vectors, axis=0)

def extract_and_combine_features(df, wv_model):
    """提取TF-IDF和词向量特征，并进行融合"""
    print("\n划分数据集...")
    X = df['review_cut']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # === A. 提取 TF-IDF 特征 ===
    print("提取TF-IDF特征...")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=5, max_df=0.7)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print(f"TF-IDF 特征维度: {X_train_tfidf.shape[1]}")
    
    # === B. 提取词向量特征 (句子向量) ===
    print("提取词向量特征 (句子向量)...")
    X_train_w2v = np.array([sentence_to_vector(sentence, wv_model) for sentence in tqdm(X_train, desc="Train W2V")])
    X_test_w2v = np.array([sentence_to_vector(sentence, wv_model) for sentence in tqdm(X_test, desc="Test W2V")])
    print(f"词向量特征维度: {X_train_w2v.shape[1]}")
    
    # === C. 融合特征 ===
    print("融合 TF-IDF 和词向量特征...")
    # hstack 用于水平拼接稀疏矩阵 (csr_matrix) 和稠密数组
    X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_w2v)])
    X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_w2v)])
    print(f"融合后特征维度: {X_train_combined.shape[1]}")
    
    # 保存向量化器以备后用
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"TF-IDF向量化器已保存至: {VECTORIZER_PATH}")
    
    return X_train_combined, X_test_combined, y_train, y_test


# --- 4. 网格搜索、模型训练与评估 ---
def tune_and_evaluate_classifiers(X_train_tfidf, X_test_tfidf, y_train, y_test):
    """对多个分类器进行网格搜索调优，并选出最佳模型进行评估"""
    
    # 定义要测试的分类器和它们的参数网格
    classifiers = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000, solver='liblinear', random_state=42, class_weight='balanced'),
            "params": {
                'C': [0.1, 1, 10, 50],
                'penalty': ['l1', 'l2']
            }
        },
        "Linear SVC": {
            "model": LinearSVC(max_iter=2000, random_state=42, dual=True, class_weight='balanced'),
            "params": {
                'C': [0.01, 0.1, 1, 10]
            }
        },
        "Multinomial NB": {
            "model": MultinomialNB(),
            "params": {
                'alpha': [0.01, 0.1, 0.5, 1.0]
            }
        }
    }
    
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""

    # 遍历每个分类器进行网格搜索
    for name, a_classifier in classifiers.items():
        print(f"\n--- 正在为 {name} 进行网格搜索 ---")
        start_time = time.time()
        
        grid_search = GridSearchCV(
            a_classifier["model"],
            a_classifier["params"],
            cv=5,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train_tfidf, y_train)
        
        end_time = time.time()
        
        print(f"{name} 最佳参数: ", grid_search.best_params_)
        print(f"交叉验证最佳准确率: {grid_search.best_score_:.4f}")
        print(f"搜索耗时: {(end_time - start_time):.2f} 秒")
        
        # 记录并更新全局最佳模型
        if grid_search.best_score_ > best_accuracy:
            best_accuracy = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_model_name = name
    
    print(f"\n--- 网格搜索完成 ---")
    print(f"在所有分类器中，表现最好的是: {best_model_name}")
    print(f"其交叉验证准确率为: {best_accuracy:.4f}")
    
    # --- 使用找到的最佳模型在独立的测试集上进行最终评估 ---
    print("\n--- 使用最佳模型在测试集上进行最终评估 ---")
    y_pred = best_model.predict(X_test_tfidf)
    
    final_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n测试集最终正确率 (Accuracy): {final_accuracy:.4f}")
    
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))

    if final_accuracy >= 0.88:
        print("\n恭喜！最终模型性能已达到通过标准 (≥ 0.88)。")
    else:
        print("\n最终模型性能未达到标准。")
        
    # 保存最佳模型
    print(f"\n保存最佳模型 ({best_model_name}) 到磁盘...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"最佳模型已保存至: {MODEL_PATH}")


# 使用模型融合
def tune_and_evaluate(X_train_feat, X_test_feat, y_train, y_test):
    """对多个分类器进行网格搜索调优"""
    # ... (这部分代码与上一版 tune_and_evaluate_classifiers 函数基本相同)
    # 只是现在传入的特征是融合后的特征 X_train_combined
    # 朴素贝叶斯 (MultinomialNB) 不能处理负值，所以我们从这里移除它，
    # 因为词向量平均后可能出现负值。
    classifiers = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000, solver='liblinear', random_state=42, class_weight='balanced'),
            "params": {'C': [1, 10, 50]}
        },
        "Linear SVC": {
            "model": LinearSVC(max_iter=2000, random_state=42, dual=True, class_weight='balanced'),
            "params": {'C': [0.1, 1, 10]}
        }
    }
    
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""

    for name, a_classifier in classifiers.items():
        print(f"\n--- 正在为 {name} 进行网格搜索 (使用融合特征) ---")
        grid_search = GridSearchCV(a_classifier["model"], a_classifier["params"], cv=3, scoring='accuracy', verbose=2, n_jobs=-1) # cv=3 加快速度
        grid_search.fit(X_train_feat, y_train)
        
        print(f"{name} 最佳参数: ", grid_search.best_params_)
        print(f"交叉验证最佳准确率: {grid_search.best_score_:.4f}")
        
        if grid_search.best_score_ > best_accuracy:
            best_accuracy = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_model_name = name
            
    print(f"\n--- 网格搜索完成，最佳模型为: {best_model_name} ---")
    
    # 使用最佳模型在测试集上评估
    print("\n--- 使用最佳模型在测试集上进行最终评估 ---")
    y_pred = best_model.predict(X_test_feat)
    final_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n测试集最终正确率 (Accuracy): {final_accuracy:.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))

    if final_accuracy >= 0.88:
        print("\n恭喜！融合特征模型性能已达到通过标准 (≥ 0.88)。")
    else:
        print("\n融合特征模型性能仍未达到标准。")
        
    # 保存最佳模型
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"最佳模型已保存至: {MODEL_PATH}")

# --- 5. 主执行函数 ---
if __name__ == '__main__':
    create_dirs()
    # 1. 加载和预处理数据
    dataframe = load_and_preprocess_data(DATA_PATH)
    # 2. 加载词向量模型
    word_vectors = load_word_vectors(WORD_VECTORS_PATH)
    # 3. 提取并融合特征
    X_train_features, X_test_features, y_train_labels, y_test_labels = extract_and_combine_features(dataframe, word_vectors)
    # 4. 调优、训练、评估和保存
    tune_and_evaluate(X_train_features, X_test_features, y_train_labels, y_test_labels)