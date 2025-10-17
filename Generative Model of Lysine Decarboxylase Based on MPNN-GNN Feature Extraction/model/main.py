import time
import optuna
import random
import numpy as np
import torch
import os

from torch.nn import SmoothL1Loss,MSELoss,L1Loss,HuberLoss
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from torch.utils.tensorboard import  SummaryWriter
from torch.utils.data import RandomSampler
from mpnn_predictor import MPNNPredictorWithProtein
from loading_data import MolecularDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import dgl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, explained_variance_score
from sklearn.metrics import roc_curve, auc,precision_score, recall_score, confusion_matrix,accuracy_score



def set_seed(seed=42):
    """固定随机种子，确保每次训练结果一致"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)




def early_stopping_regression(avg_valid_loss,
                              best_valid_loss,
                              model,
                              epoch,
                              epochs_without_improvement,
                              best_model_state,
                              patience=5):
    """
    基于验证集 loss 的早停函数
    """
    if avg_valid_loss < best_valid_loss:
        # 有改进
        best_valid_loss = avg_valid_loss
        best_model_state = model.state_dict()
        epochs_without_improvement = 0
        return 'continue', best_valid_loss, epochs_without_improvement, best_model_state
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            return 'end', best_valid_loss, epochs_without_improvement, best_model_state
        else:
            return 'continue', best_valid_loss, epochs_without_improvement, best_model_state

def early_stopping(avg_valid_loss,
                   avg_valid_acc,
                   best_valid_loss,
                   best_valid_acc,
                   model,
                   epoch,
                   epochs_without_improvement,
                   best_model_state,
                   patience=10):
    """
    早停：如果验证集loss没下降且验证指标没有提升，连续patience轮就停止训练
    """
    stop_flag = 'continue'

    improved = False
    # 如果 loss 下降或 accuracy 提升
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        improved = True
    if avg_valid_acc > best_valid_acc:
        best_valid_acc = avg_valid_acc
        improved = True

    if improved:
        best_model_state = model.state_dict()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            stop_flag = 'end'

    return stop_flag, best_valid_loss, best_valid_acc, epochs_without_improvement, best_model_state


def train_valid(model,
                loss_trans,
                learning_rate,
                weight_decay,
                batch_size,
                epochs,
                device,
                dataset_train,
                dataset_valid,
                writer,
                trial_id=None,
                seed=None
                ):

    if trial_id is None:
        if seed is not None:
            set_seed(seed)
        elif seed is  None:
            seed = 42
            set_seed(seed)
    elif trial_id is not None:
        seed = trial_id
        set_seed(seed)

    # writer = SummaryWriter('train_logs')
    time_start = time.time()

    # 🎯 使用 RandomSampler 确保每个 trial 数据顺序不同
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size,collate_fn=MolecularDataset.collate_fn,
                                  sampler=RandomSampler(dataset_train, generator=generator),
                                  num_workers=0,  pin_memory=False)# 指定自定义的 collate_fn

    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size,collate_fn=MolecularDataset.collate_fn,
                                  num_workers=0,  pin_memory=False)
    # dataloader_kras_test = DataLoader(kras_test, batch_size=batch_size, collate_fn=MolecularDataset.collate_fn,
    #                                 num_workers=0, pin_memory=False)

    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)      #优化器选择：AdamW


    epochs_without_improvement = 0  # 计数器，记录多少轮没有提升
    best_model_state = None  # 用于保存最好的模型

    epoch_rounds=0
    train_round=0
    best_valid_loss = float('inf')
    best_acc = 0.0


    for epoch in range(epochs):
        model.train()

        running_loss = 0
        epoch_rounds+=1
        print('——————————Start {}th rounds of training——————————'.format(epoch_rounds))

        for i in dataloader_train:          #训练模型
            graph, node_feats, edge_feats, protein_feats, value ,adj_matrix= i  # 解包 tuple
            graph = graph.to(device)
            protein_feats = protein_feats.to(device)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            value = value.to(device)
            value = value.view(-1,1).float()
            adj_matrix = adj_matrix.to(device)

            output = model(graph,node_feats,edge_feats,protein_feats,adj_matrix)
            output = output.squeeze(-1)
            # output = output.to(torch.float32).to(device)

            loss = loss_trans(output, value)

            loss_value = loss.item()    #new

            optim.zero_grad()  # 清除先前梯度参数为0
            loss.backward()  # 反向传播计算 损失对所有可训练参数的梯度
            optim.step()

            torch.cuda.empty_cache()    #new

            running_loss += loss.item()
            train_round+=1
            if train_round % 100 == 0:
                time_end = time.time()
                print('Training time:{}s'.format(time_end-time_start))
                print('The cross entropy loss for the {}th training is:{:.4f}'.format(train_round,loss))
            writer.add_scalar('realtime_Loss/Train_{}'.format(trial_id), loss.item(), global_step=train_round)       #输出每次训练的损失图像

        avg_train_loss = running_loss / len(dataloader_train)
        print('The average loss of the {}th round of training is {:.4f}'.format(epoch+1,avg_train_loss))



        valid_loss_total = 0.0

        all_preds = []
        all_trues = []
        all_preds_labels = []
        all_trues_labels = []
        correct_total = 0
        total_samples = 0
        epsilon_percent = 0.15  # 允许 15% 误差
        best_valid_loss=float('inf')


        model.eval()
        with torch.no_grad():
            for batch in dataloader_valid:
                graph, node_feats, edge_feats, protein_feats, value, adj_matrix = batch

                # 数据迁移
                graph = graph.to(device)
                node_feats = node_feats.to(device)
                edge_feats = edge_feats.to(device)
                protein_feats = protein_feats.to(device)
                value = value.to(device).view(-1,1).float()  # 回归要 float
                adj_matrix = adj_matrix.to(device)

                # 前向传播
                output = model(graph, node_feats, edge_feats, protein_feats, adj_matrix)

                output = output.squeeze(-1)
                # output = output.to(torch.float32).to(device)

                # 计算验证损失
                loss = loss_trans(output, value)
                valid_loss_total += loss.item() * value.size(0)

                # 收集预测与真实标签
                all_preds.extend(output.cpu().numpy())
                all_trues.extend(value.cpu().numpy())

        # 平均验证 loss
        avg_valid_loss = valid_loss_total / len(dataset_valid)

        # 回归指标
        r2 = r2_score(all_trues, all_preds)
        mse = mean_squared_error(all_trues, all_preds)

        print(f"Validation Loss: {avg_valid_loss:.5f}")
        print(f"R2 score: {r2:.4f}")
        print(f"MSE: {mse:.5f}")


        # 早停
        stop_flag, best_valid_loss, epochs_without_improvement, best_model_state = early_stopping_regression(
            avg_valid_loss=mse,
            best_valid_loss=best_valid_loss,
            model=model,
            epoch=epoch,
            epochs_without_improvement=epochs_without_improvement,
            best_model_state=best_model_state,
            patience=5
        )

        if stop_flag == 'end':
            break

        # 恢复最优模型
    if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.eval()

            save_dir = "model_pth"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"best_model{trial_id}.pth" if trial_id else "best_model.pth")
            torch.save(model.state_dict(), save_path)

            print('========================================================')
            print(f'Average loss on the validation set for the best model: {best_valid_loss:.5f}')
            print(f'The accuracy of the best model: {best_acc:.4f}')
            print(f"The best model has been saved as {save_path}")



            # 计算平均验证损失
            # avg_valid_loss = valid_loss_total / len(dataloader_valid.dataset)
            '''
        avg_valid_loss = valid_loss_total / total_samples
        avg_valid_acc = correct_total / total_samples

        # 回归指标
        mse = mean_squared_error(all_trues, all_preds)      #最优为0
        rmse = mse ** 0.5
        mae = mean_absolute_error(all_trues, all_preds)     #最优为0
        r2 = r2_score(all_trues, all_preds)         #最优为1

        # 输出
        print(f"Validation Loss: {avg_valid_loss:.4f}")
        print(f"Accuracy within ±{epsilon_percent * 100:.0f}%: {avg_valid_acc:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")

        writer.add_scalar('avg_Loss/Train_{}'.format(trial_id), avg_train_loss, global_step=epoch)
        writer.add_scalar('avg_Loss/Valid_{}'.format(trial_id), avg_valid_loss, global_step=epoch)
        writer.add_scalar('Accuracy/Valid_{}'.format(trial_id), avg_valid_acc, global_step=epoch)
        

        # Early stopping check
        stop_flag, best_valid_loss, epochs_without_improvement, best_model_state, best_r2, = (
            early_stopping(avg_valid_loss,
                           r2,
                           best_valid_loss,
                           model,
                           epoch,
                           epochs_without_improvement,
                           best_model_state=best_model_state,
                           best_valid_acc=best_r2
                           ))
        if stop_flag == 'end':
            break
        '''
    return best_valid_loss, best_acc




def Bayesian(loss_trans,
             trials,
             device,
             dataset_train,
             dataset_valid
             ):
    writer_by_path = 'log_by'
    writer_by = SummaryWriter(writer_by_path)
    def objective(trial):       # 贝叶斯优化目标函数
        trial_num = trial.number
        writer_path='log_by/log_by_{}'.format(trial_num)
        writer = SummaryWriter(writer_path)
        mpnn = MPNNPredictorWithProtein().to(device)  # 🎯 重新初始化模型

        # mpnn.load_state_dict(torch.load('best_model.pth', weights_only=False))  # 加载训练好的权重

        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3,log=False)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2,log=False)
        batch_size = trial.suggest_categorical('batch_size', [16,32,64])
        epochs = trial.suggest_int('epochs', 30,50)

        best_valid_loss, best_valid_acc = train_valid(
            model=mpnn,
            loss_trans=loss_trans,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            dataset_train=dataset_train,
            dataset_valid=dataset_valid,
            trial_id=trial.number,
            writer=writer,
        )
        trial.set_user_attr("valid_acc", float(best_valid_acc))

        writer_by.add_scalar('best_Loss/Valid', best_valid_loss, global_step=trial_num)
        writer_by.add_scalar('best_Accuracy/Valid', best_valid_acc, global_step=trial_num)
        writer_by.add_scalar('best_Accuracy/lr', best_valid_acc,
                             global_step=trial.params['learning_rate'])
        writer_by.add_scalar('best_Accuracy/wd', best_valid_acc,
                             global_step=trial.params['weight_decay'])

        return  best_valid_acc

    # study = optuna.create_study(direction='minimize')  # 最小化验证损失
    study = optuna.create_study(direction='maximize',
                                pruner=optuna.pruners.HyperbandPruner(min_resource=10),
                                sampler=optuna.samplers.TPESampler())  # 最大化正确率
    study.optimize(objective, n_trials=trials)

    print('Best trial:')
    best_trial = study.best_trial
    print(f'  Trail: {best_trial.number}')
    print(f'  Average Accuracy: {best_trial.user_attrs["valid_acc"]:.4f}')
    print(f'  Loss Value: {best_trial.value}')
    print(f'  Params: {best_trial.params}')


def test(model,
         dataset_test,
         batch_size,
         loss_trans,
         device,
         seed=None
         ):
    if seed is not None:
        set_seed(seed)
    else:
        set_seed(seed)

    writer = SummaryWriter('test_logs')
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size,
                                 collate_fn=MolecularDataset.collate_fn,
                                 num_workers=0, pin_memory=False)

    valid_loss_total = 0
    labels_true = []
    labels_pred = []

    with torch.no_grad():
        for i in dataloader_test:
            graph, node_feats, edge_feats, protein_feats, labels, adj_matrix = i
            graph = graph.to(device)
            protein_feats = protein_feats.to(device)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            labels = labels.to(device).view(-1, 1).float()
            adj_matrix = adj_matrix.to(device)

            output = model(graph, node_feats, edge_feats, protein_feats, adj_matrix)
            output = output.squeeze(-1)
            loss = loss_trans(output, labels)
            valid_loss_total += loss.item() * labels.size(0)

            labels_true.extend(labels.cpu().numpy().flatten())
            labels_pred.extend(output.cpu().numpy().flatten())

    # 转为 numpy
    y_true = np.array(labels_true)
    y_pred = np.array(labels_pred)

    # 回归指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n, p = len(y_true), 1  # p=特征数，若有多个特征可修改
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    expl_var = explained_variance_score(y_true, y_pred)
    mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    pearson_corr, _ = pearsonr(y_true, y_pred)

    print(f"MSE: {mse:.5f}")        #0
    print(f"RMSE: {rmse:.5f}")        #0
    print(f"MAE: {mae:.5f}")        #0
    print(f"MedAE: {medae:.5f}")        #0
    print(f"R²: {r2:.5f}")      #1
    print(f"Adjusted R²: {adj_r2:.5f}")     #1
    print(f"Explained Variance: {expl_var:.5f}")     #1
    print(f"MAPE: {mape_val:.2f}%")     #0
    print(f"Pearson r: {pearson_corr:.5f}")     #1

    return 0


def ROC_AUC( labelscore_predicted,labels_true):
    # 计算 FPR, TPR 和 阈值
    fpr, tpr, thresholds = roc_curve(labels_true, labelscore_predicted)
    roc_auc = auc(fpr, tpr)  # 计算 AUC 值

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # 随机分类器参考线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png', dpi=500)
    plt.close()
    return roc_auc



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device:', device)

if __name__ == '__main__':
    while True:
        code = input('mission:')

        if code == 'train':


            loss_trans = nn.MSELoss().to(device)
            writer_path = 'log_train'
            writer = SummaryWriter(writer_path)

            mpnn = MPNNPredictorWithProtein().to(device)
            # mpnn.load_state_dict(torch.load('best_model.pth', weights_only=False))
            train_file_path = 'test.csv'
            valid_file_path = 'test.csv'
            dataset_train = MolecularDataset.loading_data(train_file_path, device=device)
            dataset_valid = MolecularDataset.loading_data(valid_file_path, device=device)

            train_valid(model=mpnn,
                    loss_trans=loss_trans,
                    device=device,
                    learning_rate=0.0008086183722256094,
                    weight_decay=0.00986164935180509,
                    batch_size=16,
                    epochs=30,
                    dataset_train=dataset_train,
                    dataset_valid=dataset_valid,
                    writer=writer
                    )
            break

        elif code == 'by':

            loss_trans = nn.MSELoss().to(device)
            # loss_trans = nn.CrossEntropyLoss().to(device)

            train_file_path = 'trainset.csv'
            valid_file_path = 'validset.csv'

            dataset_train = MolecularDataset.loading_data(train_file_path, device=device)
            dataset_valid = MolecularDataset.loading_data(valid_file_path, device=device)
            Bayesian(
                loss_trans=loss_trans,
                device=device,
                trials=20,
                dataset_train=dataset_train,
                dataset_valid=dataset_valid,
                )
            break

        elif code == 'test':
            loss_trans = nn.MSELoss().to(device)

            mpnn = MPNNPredictorWithProtein().to(device)
            # mpnn.load_state_dict(torch.load('best_model.pth',weights_only=False) ) # 加载训练好的权重
            mpnn.eval()

            test_file_path = 'test.csv'
            dataset_test = MolecularDataset.loading_data(test_file_path, device=device)
            test(model=mpnn,
                dataset_test=dataset_test,
                batch_size=8,
                loss_trans=loss_trans,
                device=device,
                seed=0
                )
            break

        elif code == 'ft':
            writer_path = 'log_ft'
            writer = SummaryWriter(writer_path)

            loss_trans = nn.MSELoss().to(device)
            mpnn = MPNNPredictorWithProtein().to(device)
            mpnn.load_state_dict(torch.load('best_model.pth', weights_only=False))  # 加载训练好的权重

            train_file_path = 'train.csv'
            valid_file_path = 'test.csv'
            dataset_train = MolecularDataset.loading_data(train_file_path, device=device)
            dataset_valid = MolecularDataset.loading_data(valid_file_path, device=device)
            train_valid(model=mpnn,
                        loss_trans=loss_trans,
                        device=device,
                        learning_rate=0.0007031208993328868,
                        weight_decay=0.0004617109023254105,
                        batch_size=64,
                        epochs=80,
                        dataset_train=dataset_train,
                        dataset_valid=dataset_valid,
                        writer=writer,
                        )
            break

        else:
            print('code erro')




