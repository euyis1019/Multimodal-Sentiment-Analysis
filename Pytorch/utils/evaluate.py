import torch
import tqdm
@torch.no_grad()
def evaluate(model, data_loader, device,best_acc=-1):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
    # 统计验证集样本总数目
    num_samples = len(data_loader.dataset)

    # 打印验证进度
    data_loader = tqdm(data_loader, desc="validation...")
    bad_case=[]
    for step, data in enumerate(data_loader):
        batch, labels = data
        pred = model(batch.to(device))
        pred = torch.max(pred, dim=1)[1]
        tmp=torch.eq(pred, labels.to(device))
        sum_num += tmp.sum()
        bad_case.append((batch[~tmp],labels[~tmp]))
    # 计算预测正确的比例
    acc = sum_num.item() / num_samples
    if best_acc<acc:
        joblib.dump(bad_case,'bad_case.pkl')
    return acc