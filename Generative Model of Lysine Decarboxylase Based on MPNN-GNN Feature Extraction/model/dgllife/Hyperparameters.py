import optuna
import torch
from mpnn_predictor import MPNNPredictorWithProtein  # 导入你的 MPNN 模型
from train import train_and_evaluate  # 你需要实现这个训练函数


def objective(trial):
    """定义要优化的目标函数"""
    # 让 Optuna 选择超参数
    node_out_feats = trial.suggest_categorical('node_out_feats', [32, 64, 128])
    edge_hidden_feats = trial.suggest_categorical('edge_hidden_feats', [64, 128, 256])
    num_step_message_passing = trial.suggest_int('num_step_message_passing', 3, 10)
    num_step_set2set = trial.suggest_int('num_step_set2set', 3, 10)
    num_layer_set2set = trial.suggest_int('num_layer_set2set', 1, 5)

    # 传入 MPNN 进行训练
    model = MPNNPredictorWithProtein(node_in_feats=74,  # 你的原子特征维度
                          edge_in_feats=12,  # 你的键特征维度
                          node_out_feats=node_out_feats,
                          edge_hidden_feats=edge_hidden_feats,
                          num_step_message_passing=num_step_message_passing,
                          num_step_set2set=num_step_set2set,
                          num_layer_set2set=num_layer_set2set).to(device)

    # 训练并返回验证集上的评价指标（如 RMSE 或 R²）
    val_metric = train_and_evaluate(model)

    return val_metric  # 目标是最小化 RMSE 或最大化 R²


# 运行贝叶斯优化搜索
study = optuna.create_study(direction='maximize')  # 如果是 R²，direction='maximize'，如果是 RMSE，direction='minimize'
study.optimize(objective, n_trials=50)  # 运行 50 次超参数搜索

# 打印最优超参数
print("Best hyperparameters:", study.best_params)