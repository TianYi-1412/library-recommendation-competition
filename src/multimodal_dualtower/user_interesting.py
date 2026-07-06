
import torch
from data_loader import *
import pickle
from tqdm import tqdm
from collections import defaultdict
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel
import gc
from data_loader import SEED, PATH, HIST_MAX, TARGET_USER_NUM  # 从data_loader导入共享配置

# -------------------------- 全局配置 --------------------------
torch.manual_seed(SEED)

EMB_DIM = 128  # 统一embedding维度
BERT_MAX_LEN = 32  # BERT文本最大长度
LOCAL_M3E_PATH = r"D:\pycharm的文件\1\人工智能\双塔\m3e-base"  # 本地BERT路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------------------- 数据加载函数 --------------------------
def load_processed_data():
    """加载预处理后的数据"""
    with open(f"{PATH}/processed_data.pkl", "rb") as f:
        data_dict = pickle.load(f)
    return data_dict


# -------------------------- 书籍文本特征提取 --------------------------
def extract_book_text_embedding(books, item_encoder):
    """
    提取书籍文本特征：
    1. 拼接"题名+作者+一级分类"作为文本
    2. BERT编码获取768维向量
    3. PCA降维到EMB_DIM维
    """
    # 1. 加载本地BERT模型
    try:
        from sentence_transformers import SentenceTransformer
        m3e_model = SentenceTransformer(LOCAL_M3E_PATH)
        m3e_model.eval()

        # 如果有GPU则使用GPU
        if torch.cuda.is_available():
            m3e_model = m3e_model.to('cuda')
            print("M3E-base模型已加载到GPU")
        else:
            print("M3E-base模型使用CPU")

        print(f"成功加载本地M3E-base模型：{LOCAL_M3E_PATH}")

    except Exception as e:
        # 如果本地模型加载失败，尝试在线下载
        print(f"本地模型加载失败: {e}，尝试在线下载...")
        try:
            m3e_model = SentenceTransformer('moka-ai/m3e-base')
            m3e_model.eval()
            if torch.cuda.is_available():
                m3e_model = m3e_model.to('cuda')
            print("在线M3E-base模型加载成功")
        except Exception as e2:
            raise ImportError(f"M3E-base模型加载失败！错误：{str(e2)}")

        # 2. 构建书籍文本字典
    book_text_dict = books.set_index('book_id').apply(
        lambda r: f"{r['题名']} {r['作者']} {r['一级分类']}", axis=1
    ).to_dict()
    print(f"共处理 {len(book_text_dict)} 本书籍的文本")

    # 3. M3E-base批量编码文本
    item_vec_list = []
    batch_size = 64
    book_ids = [str(bid) for bid in item_encoder.keys()]

    # 准备所有文本
    all_texts = []
    for book_id in book_ids:
        text = book_text_dict.get(book_id, "未知书名 未知作者 未知分类")
        all_texts.append(text)

    # 分批处理
    with torch.no_grad():
        for idx in tqdm(range(0, len(all_texts), batch_size), desc="M3E-base编码书籍文本"):
            batch_texts = all_texts[idx:idx + batch_size]

            # M3E-base编码
            embeddings = m3e_model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=m3e_model.device
            )
            item_vec_list.append(embeddings.cpu().numpy())

    if not item_vec_list:
        raise ValueError("错误：所有书籍文本编码失败，请检查数据！")

    # 合并向量
    item_vec_mat = np.concatenate(item_vec_list, axis=0)
    print(f"M3E-base编码完成，原始向量维度: {item_vec_mat.shape}")

    # 4. PCA降维处理
    if EMB_DIM < item_vec_mat.shape[1]:
        print(f"执行PCA降维: {item_vec_mat.shape[1]} -> {EMB_DIM}")
        pca = PCA(n_components=EMB_DIM, random_state=SEED)
        item_text_emb = pca.fit_transform(item_vec_mat)
    else:
        # 如果目标维度大于等于原始维度，进行零填充
        print(f"进行零填充: {item_vec_mat.shape[1]} -> {EMB_DIM}")
        pad_width = EMB_DIM - item_vec_mat.shape[1]
        if pad_width > 0:
            item_text_emb = np.pad(item_vec_mat,
                                   ((0, 0), (0, pad_width)),
                                   mode='constant',
                                   constant_values=0)
        else:
            item_text_emb = item_vec_mat

    # 释放内存
    del m3e_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"书籍文本特征提取完成：最终维度={item_text_emb.shape}")
    return item_text_emb


# -------------------------- 用户历史序列构建 --------------------------
def build_user_history_sequence(train_data, num_users, user_encoder, item_encoder, HIST_MAX):
    """
    构建用户历史交互序列：
    - 每个用户的交互物品按时间排序
    - 截断/补PAD
    """
    # 1. 转换原始ID到编码ID（统一为字符串键）
    raw2uid = {str(raw): enc for raw, enc in user_encoder.items()}
    raw2iid = {str(raw): enc for raw, enc in item_encoder.items()}

    # 2. 统计用户历史交互（按时间排序）
    user_hist = defaultdict(list)
    for _, row in train_data.iterrows():
        u_raw = str(row['user_id'])
        i_raw = str(row['book_id'])
        if u_raw in raw2uid and i_raw in raw2iid:
            u_enc = raw2uid[u_raw]
            i_enc = raw2iid[i_raw]
            user_hist[u_enc].append(i_enc)

    # 3. 序列处理
    PAD = num_users  # PAD索引设为用户数（避免与物品索引冲突）
    user_seq_dict = {}

    for u_enc in tqdm(range(num_users), desc="构建用户历史序列"):
        hist = user_hist.get(u_enc, [])
        hist_truncated = hist[-HIST_MAX:] if len(hist) > HIST_MAX else hist
        hist_padded = [PAD] * (HIST_MAX - len(hist_truncated)) + hist_truncated
        user_seq_dict[u_enc] = torch.tensor(hist_padded, dtype=torch.long)

    print(f"用户历史序列构建完成：共{len(user_seq_dict)}个用户，序列长度={HIST_MAX}，PAD索引={PAD}")
    return user_seq_dict, PAD


# -------------------------- 用户分类偏好提取 --------------------------
def extract_user_category_preference(inter_train, user_encoder, uid2raw):
    """
    提取用户分类偏好特征：
    - 每个用户对"一级分类"的交互占比
    """
    # 1. 检查数据
    if '一级分类' not in inter_train.columns:
        raise ValueError("交互数据中缺少'一级分类'列，请检查数据预处理阶段是否正确合并了书籍信息")

    user_cat_dist = {}

    for u_enc in tqdm(range(len(user_encoder)), desc="计算用户分类偏好"):
        u_raw = uid2raw[u_enc]
        # 直接从inter_train中获取该用户的所有交互记录
        user_inter = inter_train[inter_train['user_id'] == u_raw]

        if len(user_inter) == 0:
            user_cat_dist[u_enc] = {"未知分类": 1.0}
            continue

        # 计算分类占比
        cat_count = user_inter['一级分类'].value_counts(normalize=True)
        user_cat_dist[u_enc] = cat_count.to_dict()

    print("用户分类偏好特征提取完成")
    print("交互数据列名:", inter_train.columns.tolist())
    print("一级分类缺失值数量:", inter_train['一级分类'].isna().sum())
    return user_cat_dist


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 1. 加载数据
    print("加载预处理数据...")
    data_dict = load_processed_data()
    inter_train = data_dict["inter_train"]
    books = data_dict["books"]
    user_encoder = data_dict["user_encoder"]
    item_encoder = data_dict["item_encoder"]
    uid2raw = data_dict["uid2raw"]  # 获取用户编码反向映射
    num_users = TARGET_USER_NUM  # 固定600用户

    # 2. 提取书籍文本特征
    item_text_emb = extract_book_text_embedding(books, item_encoder)

    # 3. 构建用户历史序列（简化参数传递）
    user_seq_dict, PAD = build_user_history_sequence(
        inter_train, num_users, user_encoder, item_encoder, HIST_MAX
    )

    # 4. 提取用户分类偏好（使用uid2raw并移除重复合并）
    user_cat_dist = extract_user_category_preference(inter_train, user_encoder, uid2raw)

    # 5. 保存特征
    interest_dict = {
        "item_text_emb": item_text_emb,
        "user_seq_dict": user_seq_dict,
        "PAD": PAD,
        "user_cat_dist": user_cat_dist,
        "EMB_DIM": EMB_DIM,
        "HIST_MAX": HIST_MAX
    }

    with open(f"{PATH}/user_interest_features.pkl", "wb") as f:
        pickle.dump(interest_dict, f)

    print("用户兴趣建模模块完成！特征已保存至 ./datasets/user_interest_features.pkl")