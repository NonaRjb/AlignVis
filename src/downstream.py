import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

import torch
import torch.nn.functional as F
from tqdm import tqdm
# from src.models.eeg_classifier import EEGClassifier
# from src.models.trainer import UnimodalTrainer
from src.utils import get_embeddings


def classification(
    loaders,
    eeg_enc_name, 
    dataset_name, n_channels, n_samples, n_classes, 
    finetune_epochs, warmup_epochs, lr, min_lr, weight_decay,
    save_path,
    pretrained_encoder=None, device="cuda:0", 
    model_configs=None,
    **kwargs):
    return
    # classifier_model = EEGClassifier(
    #         backbone=eeg_enc_name,
    #         n_channels=n_channels, 
    #         n_samples=n_samples, 
    #         n_classes=n_classes,
    #         pretrained_encoder=pretrained_encoder,
    #         device=device, 
    #         **model_configs[eeg_enc_name])
    # classifier_model = classifier_model.float()

    # loss = torch.nn.CrossEntropyLoss()
    # optim = torch.optim.AdamW(classifier_model.parameters(), lr=min_lr, weight_decay=weight_decay)
    # trainer = UnimodalTrainer(model=classifier_model, 
    #                           optimizer=optim, 
    #                           loss=loss, 
    #                           epochs=finetune_epochs, 
    #                           warmup_epochs=warmup_epochs,
    #                           lr=lr, min_lr=min_lr,  
    #                           mixed_precision=True,
    #                           num_classes=n_classes,
    #                           save_path=save_path, 
    #                           filename=f'cls_{eeg_enc_name}_{dataset_name}.pth', 
    #                           device=device)
    # best_classifier = trainer.train(loaders['train'], loaders['val'])
    # classifier_model.load_state_dict(best_classifier['model_state_dict'])
    # test_loss, test_acc = trainer.evaluate(classifier_model, loaders['test'])
    # print(f"Test Loss: {test_loss} | Test Acc.: {test_acc}")

    # return test_loss, test_acc


def retrieval(eeg_encoder, img_encoder, data_loader, device="cuda:0", return_subject_id=False, **kwargs):

    eeg_encoder.eval()

    if img_encoder is not None:
        img_encoder.eval()
        
    img_embeddings, _ = get_embeddings(img_encoder, data_loader, modality="img", return_subject_id=return_subject_id, device=device)
    img_embeddings = torch.from_numpy(img_embeddings).to(device)

    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    with torch.no_grad():
        # progress_bar = tqdm(data_loader)
        for i, (data, y) in enumerate(data_loader):
            if return_subject_id:
                subject_id = data[1]
                subject_id = subject_id.to(device, non_blocking=True)
                data = data[0]
            eeg, img = data
            eeg = eeg.to(device, non_blocking=True)
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if img_encoder is not None:
                img_embeddings_batch = img_encoder(img)
            else:
                img_embeddings_batch = img
            img_embeddings_batch = F.normalize(img_embeddings_batch, p=2, dim=-1)
            sim_img = (img_embeddings_batch @ img_embeddings.t()).softmax(dim=-1)
            _, tt_label = sim_img.topk(1)

            if return_subject_id:
                eeg_embeddings = eeg_encoder(eeg, subject_id)
            else:
                eeg_embeddings = eeg_encoder(eeg)
            eeg_embeddings = eeg_embeddings - torch.mean(eeg_embeddings, dim=-1, keepdim=True)
            eeg_embeddings = F.normalize(eeg_embeddings, p=2, dim=-1)
            
            similarity = (eeg_embeddings @ img_embeddings.t()).softmax(dim=-1)
            _, indices = similarity.topk(5)

            # tt_label = y.view(-1, 1)
            tt_label = tt_label.view(-1, 1)
            total += y.size(0)
            top1 += (tt_label == indices[:, :1]).sum().item()
            top3 += (tt_label == indices[:, :3]).sum().item()
            top5 += (tt_label == indices).sum().item()

            
        top1_acc = float(top1) / float(total)
        top3_acc = float(top3) / float(total)
        top5_acc = float(top5) / float(total)
    
    print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))

    return top1_acc, top3_acc, top5_acc