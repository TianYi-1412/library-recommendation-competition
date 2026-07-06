import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle
import warnings
from tqdm import tqdm
import time
from collections import Counter

warnings.filterwarnings('ignore')


class BookRecommendationDataset(Dataset):
    """自定义数据集类"""

    def __init__(self, sequences, targets, user_ids, sequence_info=None, max_sequence_length=10):
        self.sequences = sequences
        self.targets = targets
        self.user_ids = user_ids
        self.sequence_info = sequence_info
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        user_id = self.user_ids[idx]

        # 转换为Tensor
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        user_tensor = torch.tensor(user_id, dtype=torch.long)

        return sequence_tensor, target_tensor, user_tensor


class SequenceRecommender(nn.Module):
    """基于序列的图书推荐模型"""

    def __init__(self, num_books, num_users, embedding_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.3, model_type='lstm'):
        super(SequenceRecommender, self).__init__()

        self.num_books = num_books
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_type = model_type

        # 嵌入层
        self.book_embedding = nn.Embedding(num_books, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # 序列模型
        if model_type == 'lstm':
            self.sequence_model = nn.LSTM(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout, bidirectional=False
            )
        elif model_type == 'gru':
            self.sequence_model = nn.GRU(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout, bidirectional=False
            )
        elif model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.sequence_model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 注意力机制
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8, dropout=dropout)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_books)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, sequences, user_ids):
        batch_size = sequences.size(0)
        seq_length = sequences.size(1)

        # 获取图书嵌入
        book_embeds = self.book_embedding(sequences)  # [batch_size, seq_len, embedding_dim]

        # 获取用户嵌入
        user_embeds = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        user_embeds = user_embeds.unsqueeze(1)  # [batch_size, 1, embedding_dim]

        # 序列建模
        if self.model_type in ['lstm', 'gru']:
            # LSTM/GRU处理
            sequence_output, _ = self.sequence_model(book_embeds)
            sequence_representation = sequence_output[:, -1, :]  # 取最后一个时间步
        else:
            # Transformer处理
            sequence_output = self.sequence_model(book_embeds)

            # 注意力池化
            sequence_output = sequence_output.transpose(0, 1)  # [seq_len, batch_size, embedding_dim]
            attn_output, _ = self.attention(sequence_output, sequence_output, sequence_output)
            sequence_representation = attn_output[-1]  # 取最后一个位置的输出

        # 融合用户信息和序列信息
        combined = torch.cat([sequence_representation, user_embeds.squeeze(1)], dim=1)

        # 输出预测
        output = self.output_layer(combined)

        return output


class BookRecommendationTrainer:
    """模型训练器"""

    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []

    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc="Training")
        for batch_idx, (sequences, targets, user_ids) in enumerate(progress_bar):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            user_ids = user_ids.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(sequences, user_ids)
            loss = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

        avg_loss = total_loss / len(dataloader)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, dataloader, book_encoder):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        progress_bar = tqdm(dataloader, desc="Validating")
        with torch.no_grad():
            for sequences, targets, user_ids in progress_bar:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                user_ids = user_ids.to(self.device)

                outputs = self.model(sequences, user_ids)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # 获取预测结果
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # 更新进度条
                progress_bar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Avg Val Loss': f'{total_loss / (len(progress_bar)):.4f}'
                })

        avg_loss = total_loss / len(dataloader)

        # 计算指标
        print("计算评估指标...")
        all_predictions = [book_encoder.inverse_transform([pred])[0] for pred in tqdm(all_predictions, desc="解码预测")]
        all_targets = [book_encoder.inverse_transform([target])[0] for target in tqdm(all_targets, desc="解码目标")]

        # 计算F1分数（只考虑正样本预测）
        precision = precision_score(all_targets, all_predictions, average='micro', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='micro', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='micro', zero_division=0)

        self.val_losses.append(avg_loss)
        self.val_f1_scores.append(f1)

        return avg_loss, precision, recall, f1

    def train(self, train_loader, val_loader, book_encoder, epochs=50, early_stopping_patience=5):
        """完整训练过程"""
        print("开始训练模型...")
        best_f1 = 0
        patience_counter = 0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 50)

            # 训练
            train_loss = self.train_epoch(train_loader)
            print(f'Train Loss: {train_loss:.4f}')

            # 验证
            val_loss, precision, recall, f1 = self.validate(val_loader, book_encoder)
            print(f'Val Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

            # 学习率调整
            self.scheduler.step(val_loss)

            # 早停检查
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'新的最佳模型已保存，F1: {best_f1:.4f}')
            else:
                patience_counter += 1
                print(f'早停计数器: {patience_counter}/{early_stopping_patience}')

            if patience_counter >= early_stopping_patience:
                print(f'早停触发! 最佳F1: {best_f1:.4f}')
                break

        print(f'训练完成! 最佳验证F1: {best_f1:.4f}')

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))

        return best_f1


class DataPreparer:
    """数据准备器"""

    def __init__(self):
        self.book_encoder = LabelEncoder()
        self.user_encoder = LabelEncoder()
        self.sequence_encoder = LabelEncoder()

    def prepare_data(self, samples):
        """准备训练数据"""
        print("准备训练数据...")

        # 提取所有图书ID并编码
        print("收集所有图书ID...")
        all_books = set()

        # 使用进度条处理序列
        for seq in tqdm(samples['sequences'], desc="处理序列"):
            all_books.update(seq)

        # 处理目标图书
        for target in tqdm(samples['targets'], desc="处理目标"):
            all_books.add(target)

        all_books_list = list(all_books)
        print(f"找到 {len(all_books_list)} 本唯一图书")

        # 编码图书ID
        print("编码图书ID...")
        self.book_encoder.fit(all_books_list)

        # 编码用户ID
        print("编码用户ID...")
        self.user_encoder.fit(samples['user_ids'])

        # 转换序列数据
        print("转换序列数据...")
        encoded_sequences = []
        for seq in tqdm(samples['sequences'], desc="编码序列"):
            try:
                encoded_seq = self.book_encoder.transform(seq)
                encoded_sequences.append(encoded_seq)
            except Exception as e:
                print(f"序列编码错误: {e}")
                continue

        # 转换目标数据
        print("转换目标数据...")
        encoded_targets = []
        for target in tqdm(samples['targets'], desc="编码目标"):
            try:
                encoded_target = self.book_encoder.transform([target])[0]
                encoded_targets.append(encoded_target)
            except Exception as e:
                print(f"目标编码错误: {e}")
                continue

        # 转换用户数据
        print("转换用户数据...")
        encoded_users = []
        for user_id in tqdm(samples['user_ids'], desc="编码用户"):
            try:
                encoded_user = self.user_encoder.transform([user_id])[0]
                encoded_users.append(encoded_user)
            except Exception as e:
                print(f"用户编码错误: {e}")
                continue

        # 确保所有列表长度一致
        min_length = min(len(encoded_sequences), len(encoded_targets), len(encoded_users))
        encoded_sequences = encoded_sequences[:min_length]
        encoded_targets = encoded_targets[:min_length]
        encoded_users = encoded_users[:min_length]

        print(f"数据准备完成:")
        print(f"  - 图书数量: {len(self.book_encoder.classes_)}")
        print(f"  - 用户数量: {len(self.user_encoder.classes_)}")
        print(f"  - 序列数量: {len(encoded_sequences)}")

        return encoded_sequences, encoded_targets, encoded_users

    def create_dataloaders(self, sequences, targets, users, batch_size=64, train_ratio=0.8):
        """创建数据加载器"""
        print("创建数据加载器...")

        # 按时间划分训练集和验证集（使用序列信息中的时间）
        dataset_size = len(sequences)
        train_size = int(train_ratio * dataset_size)

        # 随机划分（实际应用中应该按时间划分）
        indices = np.random.permutation(dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # 创建训练集
        print("创建训练数据集...")
        train_dataset = BookRecommendationDataset(
            [sequences[i] for i in tqdm(train_indices, desc="构建训练集")],
            [targets[i] for i in train_indices],
            [users[i] for i in train_indices]
        )

        # 创建验证集
        print("创建验证数据集...")
        val_dataset = BookRecommendationDataset(
            [sequences[i] for i in tqdm(val_indices, desc="构建验证集")],
            [targets[i] for i in val_indices],
            [users[i] for i in val_indices]
        )

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        print(f"数据加载器创建完成:")
        print(f"  - 训练集大小: {len(train_dataset)}")
        print(f"  - 验证集大小: {len(val_dataset)}")
        print(f"  - 批次大小: {batch_size}")

        return train_loader, val_loader


class BookRecommender:
    """完整的图书推荐系统"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.data_preparer = None
        self.trainer = None
        self.all_users = set()  # 存储所有用户ID

    def train_model(self, samples, model_type='lstm', embedding_dim=128, hidden_dim=256,
                    num_layers=2, dropout=0.3, batch_size=64, epochs=50, learning_rate=0.001):
        """训练推荐模型"""
        print("开始训练图书推荐模型...")

        # 记录所有用户
        self.all_users = set(samples['user_ids'])
        print(f"训练集中用户数量: {len(self.all_users)}")

        # 准备数据
        self.data_preparer = DataPreparer()
        sequences, targets, users = self.data_preparer.prepare_data(samples)

        # 创建数据加载器
        train_loader, val_loader = self.data_preparer.create_dataloaders(
            sequences, targets, users, batch_size=batch_size
        )

        # 创建模型
        num_books = len(self.data_preparer.book_encoder.classes_)
        num_users = len(self.data_preparer.user_encoder.classes_)

        print(f"创建模型: {num_books} 图书, {num_users} 用户")
        self.model = SequenceRecommender(
            num_books=num_books,
            num_users=num_users,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type
        )

        print(f"模型创建完成:")
        print(f"  - 模型类型: {model_type}")
        print(f"  - 图书数量: {num_books}")
        print(f"  - 用户数量: {num_users}")
        print(f"  - 参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        # 创建训练器并训练
        self.trainer = BookRecommendationTrainer(
            self.model, self.device, learning_rate=learning_rate
        )

        best_f1 = self.trainer.train(
            train_loader, val_loader,
            self.data_preparer.book_encoder,
            epochs=epochs
        )

        return best_f1

    def predict_for_users(self, user_sequences, top_k=10):
        """为用户生成预测 - 修复版，确保为所有用户生成推荐"""
        if self.model is None or self.data_preparer is None:
            raise ValueError("请先训练模型")

        self.model.eval()
        predictions = {}

        print("开始生成预测...")

        # 计算全局热门图书（用于冷启动用户）
        all_books = []
        for user_data in user_sequences.values():
            all_books.extend(user_data['book_sequence'])
        book_counts = Counter(all_books)
        most_common_books = [book_id for book_id, count in book_counts.most_common(100)]

        # 为所有用户生成推荐
        for user_id, user_data in tqdm(user_sequences.items(), desc="为用户生成预测"):
            book_sequence = user_data['book_sequence']

            # 如果用户没有借阅记录，使用热门图书
            if len(book_sequence) == 0:
                # 选择用户未借阅过的最热门图书
                for book in most_common_books:
                    if book not in book_sequence:
                        predictions[user_id] = book
                        break
                continue

            # 检查用户是否在训练集中
            if user_id not in self.data_preparer.user_encoder.classes_:
                # 冷启动用户：使用基于序列的简单推荐
                # 获取用户最近借阅的图书
                recent_books = book_sequence[-5:] if len(book_sequence) >= 5 else book_sequence

                # 找到与最近借阅图书相似的热门图书
                for book in most_common_books:
                    if book not in book_sequence:
                        predictions[user_id] = book
                        break
                continue

            # 编码用户ID
            try:
                encoded_user = self.data_preparer.user_encoder.transform([user_id])[0]
            except:
                # 如果编码失败，使用热门图书
                for book in most_common_books:
                    if book not in book_sequence:
                        predictions[user_id] = book
                        break
                continue

            # 编码序列（使用最后几个图书）
            seq_length = min(len(book_sequence), 10)  # 最大序列长度
            recent_books = book_sequence[-seq_length:]

            try:
                encoded_sequence = self.data_preparer.book_encoder.transform(recent_books)
            except:
                # 处理未知图书，使用热门图书
                for book in most_common_books:
                    if book not in book_sequence:
                        predictions[user_id] = book
                        break
                continue

            # 填充序列到固定长度
            if len(encoded_sequence) < 10:
                padding = [0] * (10 - len(encoded_sequence))
                encoded_sequence = padding + encoded_sequence.tolist()
            else:
                encoded_sequence = encoded_sequence.tolist()

            # 转换为Tensor
            sequence_tensor = torch.tensor([encoded_sequence], dtype=torch.long).to(self.device)
            user_tensor = torch.tensor([encoded_user], dtype=torch.long).to(self.device)

            # 预测
            with torch.no_grad():
                output = self.model(sequence_tensor, user_tensor)
                scores = torch.softmax(output, dim=1)
                top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)

            # 解码预测结果
            top_book_ids = self.data_preparer.book_encoder.inverse_transform(top_indices.cpu().numpy()[0])
            top_scores = top_scores.cpu().numpy()[0]

            # 排除用户已经借阅过的图书
            recommended_books = []
            for book_id, score in zip(top_book_ids, top_scores):
                if book_id not in book_sequence:
                    recommended_books.append((book_id, score))
                    if len(recommended_books) >= 1:  # 只取一个最佳推荐
                        break

            if recommended_books:
                predictions[user_id] = recommended_books[0][0]  # 取评分最高的图书
            else:
                # 如果所有推荐图书用户都已借阅，使用热门图书
                for book in most_common_books:
                    if book not in book_sequence:
                        predictions[user_id] = book
                        break

        print(f"预测完成，共为 {len(predictions)} 个用户生成推荐")

        # 确保为所有用户都生成了推荐
        missing_users = set(user_sequences.keys()) - set(predictions.keys())
        if missing_users:
            print(f"警告: 有 {len(missing_users)} 个用户没有生成推荐，使用热门图书补全")
            for user_id in missing_users:
                book_sequence = user_sequences[user_id]['book_sequence']
                for book in most_common_books:
                    if book not in book_sequence:
                        predictions[user_id] = book
                        break

        print(f"最终为 {len(predictions)} 个用户生成推荐")
        return predictions

    def save_model(self, filepath='book_recommender.pkl'):
        """保存模型和编码器"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'book_encoder': self.data_preparer.book_encoder,
            'user_encoder': self.data_preparer.user_encoder,
            'all_users': list(self.all_users),
            'model_config': {
                'num_books': self.model.num_books,
                'num_users': self.model.num_users,
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'model_type': self.model.model_type
            }
        }

        torch.save(model_data, filepath)
        print(f"模型已保存至: {filepath}")

    def load_model(self, filepath='book_recommender.pkl'):
        """加载模型和编码器"""
        model_data = torch.load(filepath, map_location=self.device)

        # 创建数据准备器
        self.data_preparer = DataPreparer()
        self.data_preparer.book_encoder = model_data['book_encoder']
        self.data_preparer.user_encoder = model_data['user_encoder']
        self.all_users = set(model_data.get('all_users', []))

        # 创建模型
        config = model_data['model_config']
        self.model = SequenceRecommender(
            num_books=config['num_books'],
            num_users=config['num_users'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            model_type=config['model_type']
        )

        # 加载权重
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.to(self.device)

        print(f"模型已从 {filepath} 加载")


def create_submission(predictions, output_file='submission.csv'):
    """创建提交文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('user_id,book_id\n')
        for user_id, book_id in predictions.items():
            f.write(f'{user_id},{book_id}\n')

    print(f"提交文件已生成: {output_file}")
    print(f"共为 {len(predictions)} 个用户生成推荐")


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载生成的样本数据
    try:
        print("加载样本数据...")
        with open('generated_samples_optimized.pkl', 'rb') as f:
            all_samples = pickle.load(f)

        # 使用固定长度样本进行训练
        train_samples = all_samples['train_samples']

        print(f"加载样本数据:")
        print(f"  - 训练样本数: {len(train_samples['sequences'])}")
        print(f"  - 测试样本数: {len(all_samples['test_samples']['sequences'])}")

    except FileNotFoundError:
        print("未找到生成的样本文件，请先运行样本生成代码")
        return

    # 创建推荐系统
    recommender = BookRecommender(device=device)

    # 训练模型
    print("开始训练模型...")
    best_f1 = recommender.train_model(
        samples=train_samples,
        model_type='lstm',
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        dropout=0.2,
        batch_size=32,
        epochs=20,
        learning_rate=0.001
    )

    print(f"模型训练完成，最佳F1分数: {best_f1:.4f}")

    # 保存模型
    recommender.save_model('trained_recommender.pkl')

    # 加载用户序列用于预测
    print("加载用户序列...")
    with open('processed_sequences.pkl', 'rb') as f:
        processed_data = pickle.load(f)

    user_sequences = processed_data['user_sequences']
    print(f"总用户数量: {len(user_sequences)}")

    # 生成预测
    predictions = recommender.predict_for_users(user_sequences, top_k=5)

    # 创建提交文件
    create_submission(predictions, 'submission.csv')

    print("推荐系统运行完成!")


# 如果直接运行此文件，执行主函数
if __name__ == "__main__":
    # 安装必要的包
    try:
        from tqdm import tqdm
    except ImportError:
        print("安装tqdm包...")
        import subprocess

        subprocess.check_call(["pip", "install", "tqdm"])
        from tqdm import tqdm

    main()