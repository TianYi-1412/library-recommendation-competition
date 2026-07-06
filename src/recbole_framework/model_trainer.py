from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.trainer import Trainer
from recbole.utils import init_seed
import torch
import numpy as np
from typing import Dict, Any, Tuple
import traceback


class FixedModelTrainer:
    """修复的模型训练器"""
    SEQUENTIAL_MODELS = {'SASRec', 'GRU4Rec', 'Caser', 'BERT4Rec'}

    # ✅ 序列模型所需的字段名
    ITEM_SEQ = 'item_id_list'
    ITEM_SEQ_LEN = 'item_length'

    def __init__(self, config_dict: Dict[str, Any]):
        self.config_dict = config_dict
        self.config = None
        self.dataset = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.model = None
        self.trainer = None

    def prepare_data(self):
        """准备数据"""
        print("准备数据集...")

        try:
            # 创建配置
            self.config = Config(
                model=self.config_dict['model'],
                dataset='library',
                config_dict=self.config_dict
            )

            # 设置随机种子
            init_seed(self.config['seed'], self.config['reproducibility'])

            # 创建数据集
            self.dataset = create_dataset(self.config)

            if self.dataset.inter_num == 0:
                raise ValueError("数据集为空，请检查数据文件")

            # 数据预处理
            self.train_data, self.valid_data, self.test_data = data_preparation(
                self.config, self.dataset
            )

            print(f"数据集信息:")
            print(f"- 用户数: {self.dataset.user_num}")
            print(f"- 图书数: {self.dataset.item_num}")
            print(f"- 交互数: {self.dataset.inter_num}")

            return True

        except Exception as e:
            print(f"数据准备失败: {e}")
            return False



    def init_model(self):
        """初始化模型"""
        model_name = self.config_dict['model']
        print(f"初始化模型: {model_name}")

        try:
            if model_name == 'BPR':
                from recbole.model.general_recommender import BPR
                self.model = BPR(self.config, self.dataset)
            elif model_name == 'NeuMF':
                from recbole.model.general_recommender import NeuMF
                self.model = NeuMF(self.config, self.dataset)
            elif model_name == 'SASRec':
                from recbole.model.sequential_recommender import SASRec
                self.model = SASRec(self.config, self.dataset)
            elif model_name == 'ItemKNN':
                from recbole.model.general_recommender import ItemKNN
                self.model = ItemKNN(self.config, self.dataset)
            elif model_name == 'Pop':
                from recbole.model.general_recommender import Pop
                self.model = Pop(self.config, self.dataset)
            elif model_name == 'GRU4Rec':
                from recbole.model.sequential_recommender import GRU4Rec
                self.model = GRU4Rec(self.config, self.dataset)
            elif model_name == 'BERT4Rec':
                from recbole.model.sequential_recommender import BERT4Rec
                self.model = BERT4Rec(self.config, self.dataset)
            elif model_name == 'Caser':
                from recbole.model.sequential_recommender import Caser
                self.model = Caser(self.config, self.dataset)
            elif model_name == 'DeepFM':  # ← 与配置一致
                from recbole.model.context_aware_recommender.deepfm import DeepFM
                self.model = DeepFM(self.config, self.dataset)


            else:
                raise ValueError(f"不支持的模型: {model_name}")

            # 移动到设备
            self.model = self.model.to(self.config['device'])
            print(f"模型已移动到: {self.config['device']}")
            return True

        except Exception as e:
            print(f"模型 {model_name} 初始化失败: {e}")
            return False

    def train(self) -> Tuple[float, Dict[str, float]]:
        """训练模型"""
        if self.model is None:
            if not self.init_model():
                raise ValueError("模型初始化失败")

        # 初始化训练器
        self.trainer = Trainer(self.config, self.model)

        # 训练模型
        print("开始训练模型...")
        try:
            best_valid_score, best_valid_result = self.trainer.fit(
                self.train_data,
                self.valid_data,
                verbose=True,
                show_progress=True,
                saved=True
            )

            print(f"最佳验证结果: {best_valid_result}")
            return best_valid_score, best_valid_result


        except Exception as e:
            import traceback
            traceback.print_exc()  # ← 加这行
            print(f"训练失败: {e}")
            raise

    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        if self.model is None:
            raise ValueError("请先训练模型")

        print("评估模型...")

        # 临时修改torch.load以处理兼容性问题
        import torch
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

        try:
            test_result = self.trainer.evaluate(self.test_data)
        finally:
            torch.load = original_load  # 恢复原始函数

        print(f"测试结果: {test_result}")
        return test_result

    def generate_recommendations(self, user_list=None, top_k=1):
        if self.model is None:
            raise ValueError("请先训练模型")

        print(f"为 {len(user_list) if user_list else '所有'} 用户生成推荐...")

        if user_list is None:
            user_list = self.dataset.id2token(self.dataset.uid_field, list(range(self.dataset.user_num)))

        valid_users = set(self.dataset.id2token(self.dataset.uid_field, list(range(self.dataset.user_num))))
        user_list = [str(u) for u in user_list if str(u) in valid_users]

        if not user_list:
            print("警告：没有有效用户可供推荐")
            return {}

        recommendations = {}
        model_name = self.config_dict['model']

        try:
            for user_id in user_list:
                uid = self.dataset.token2id(self.dataset.uid_field, [user_id])[0]

                if model_name in self.SEQUENTIAL_MODELS:
                    user_inter_df = self.train_data.dataset.inter_feat[
                        self.train_data.dataset.inter_feat[self.dataset.uid_field] == uid
                        ]
                    item_seq = user_inter_df[self.dataset.iid_field].tolist()

                    # ✅ padding/truncate 到固定长度
                    max_len = self.config['MAX_ITEM_LIST_LENGTH']  # 默认 20
                    if len(item_seq) >= max_len:
                        item_seq = item_seq[-max_len:]
                    else:
                        item_seq = [0] * (max_len - len(item_seq)) + item_seq

                    interaction = Interaction({
                        self.dataset.uid_field: torch.tensor([uid], device=self.config['device']),
                        self.ITEM_SEQ: torch.tensor([item_seq], device=self.config['device']),
                        self.ITEM_SEQ_LEN: torch.tensor([max_len], device=self.config['device']),
                    })
                    scores = self.model.full_sort_predict(interaction)
                elif model_name == 'NeuMF':
                    all_items = torch.arange(self.dataset.item_num, device=self.config['device'])
                    # ✅ 直接用外层正在遍历的 user_id，不要再套循环
                    users = torch.full_like(all_items, uid)
                    interaction = Interaction({
                        self.dataset.uid_field: users,
                        self.dataset.iid_field: all_items,
                    })
                    with torch.no_grad():
                        scores = self.model.predict(interaction)
                    _, topk_idx = torch.topk(scores, top_k)
                    topk_items = self.dataset.id2token(self.dataset.iid_field, topk_idx.cpu().numpy())
                    recommendations[user_id] = topk_items.tolist()
                elif model_name in {'DeepFM', 'WideDeep', 'xDeepFM'}:
                    return self.generate_deepfm_recommendations(user_list, top_k)

                else:
                    # ✅ 非序列模型
                    interaction = Interaction({
                        self.dataset.uid_field: torch.tensor([uid], device=self.config['device'])
                    })
                    scores = self.model.full_sort_predict(interaction)

                scores = scores.view(-1, self.dataset.item_num)
                _, topk_indices = torch.topk(scores, top_k, dim=1)
                item_ids = self.dataset.id2token(self.dataset.iid_field, topk_indices[0].cpu().numpy())
                recommendations[user_id] = item_ids.tolist()

        except Exception as e:
            print(f"生成推荐失败: {e}")
            traceback.print_exc()
            return {}

        print(f"成功为 {len(recommendations)} 个用户生成推荐")
        return recommendations

    def generate_deepfm_recommendations(self, user_list=None, top_k=1, batch_size=2048):
        """专为 DeepFM 写的推荐逻辑：构造 (user,item) 对 → 打分 → Top-K"""
        if self.model is None:
            raise ValueError("请先训练模型")

        if user_list is None:
            user_list = self.dataset.id2token(
                self.dataset.uid_field,
                list(range(self.dataset.user_num))
            )

        valid_users = set(self.dataset.id2token(
            self.dataset.uid_field,
            list(range(self.dataset.user_num))
        ))
        user_list = [str(u) for u in user_list if str(u) in valid_users]

        recommendations = {}
        device = self.config['device']
        n_items = self.dataset.item_num

        for user_id in user_list:
            uid = self.dataset.token2id(self.dataset.uid_field, [user_id])[0]

            scores = []
            # 分批构造 (uid, item) 对，避免显存爆炸
            for start in range(0, n_items, batch_size):
                end = min(start + batch_size, n_items)
                items = torch.arange(start, end, device=device)
                users = torch.full_like(items, uid)

                interaction = Interaction({
                    self.dataset.uid_field: users,
                    self.dataset.iid_field: items,
                })

                with torch.no_grad():
                    batch_scores = self.model.predict(interaction).cpu().numpy()
                scores.append(batch_scores)

            scores = np.concatenate(scores)
            # 取 Top-K
            topk_idx = np.argpartition(scores, -top_k)[-top_k:]
            topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]
            topk_items = self.dataset.id2token(self.dataset.iid_field, topk_idx)

            recommendations[user_id] = topk_items.tolist()

        return recommendations


