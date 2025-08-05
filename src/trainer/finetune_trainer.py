import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import csv


class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device,
                 model_name, label_mean=None, label_std=None, ddp=False, local_rank=0):
        self.args = args
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
        self.train_losses = []
        # self.loss_log_path = f"{args.dataset}_loss.csv"
    def compute_contrastive_loss(self, embeddings, temperature=0.5):
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)
        batch_size = embeddings.size(0) // 2

        # Positive pairs: dot product of augmented views
        pos_sim = torch.sum(embeddings[:batch_size] * embeddings[batch_size:], dim=-1) / temperature
        dists = cdist(embeddings[:batch_size].cpu().detach().numpy(), embeddings[batch_size:].cpu().detach().numpy(),
                      metric='euclidean')
        dists = torch.tensor(dists).to(embeddings.device)
        pos_sim_with_distance = pos_sim - dists.diag() / temperature  # 减去距离作为惩罚项
        sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature
        mask = torch.eye(sim_matrix.size(0), dtype=torch.bool).to(sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        labels = torch.arange(batch_size).to(sim_matrix.device)
        logits = torch.cat([pos_sim_with_distance.view(-1, 1), sim_matrix[batch_size:, :batch_size]], dim=1)
        loss = F.cross_entropy(logits, labels)
        return loss

    def _forward_epoch(self, model, batched_data):
        (smiles, batched_original_graphs, batched_graph_1, batched_graph_2, ecfp, md, labels) = batched_data
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        batched_original_graphs = batched_original_graphs.to(self.device)
        batched_graph_1 = batched_graph_1.to(self.device)
        batched_graph_2 = batched_graph_2.to(self.device)
        labels = labels.to(self.device)
        predictions, triplet_h, pooled_triplet_h = model.forward_tune(batched_original_graphs, ecfp, md)
        predictions_1, triplet_h1, pooled_triplet_h1 = model.forward_tune(batched_graph_1, ecfp, md)
        predictions_2, triplet_h2, pooled_triplet_h2 = model.forward_tune(batched_graph_2, ecfp, md)
        embeddings = torch.cat([pooled_triplet_h1.unsqueeze(0), pooled_triplet_h2.unsqueeze(0)], dim=0)
        return predictions, predictions_1, predictions_2, labels, triplet_h, embeddings

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        total_loss = 0.0

        for batch_idx, batched_data in enumerate(train_loader):
            self.optimizer.zero_grad()
            predictions, predictions_1, predictions_2, labels, triplet_h, embeddings = self._forward_epoch(
                model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean) / self.label_std

            contrastive_loss = self.compute_contrastive_loss(embeddings).mean()

            loss = (self.loss_fn(predictions,
                                 labels) * is_labeled).mean() * 0.8 + 0.2 * contrastive_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()

            total_loss += loss.item()

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss.item(),
                                               (epoch_idx - 1) * len(train_loader) + batch_idx + 1)

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        # print(f"Epoch {epoch_idx}: Loss: {avg_loss:.4f}")

    def fit(self, model, train_loader, val_loader, test_loader):
        best_val_result, best_test_result, best_train_result = self.result_tracker.init(), self.result_tracker.init(), self.result_tracker.init()
        best_epoch = 0

        for epoch in range(1, self.args.n_epochs + 1):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)

            if self.local_rank == 0:
                val_result = self.eval(model, val_loader)
                test_result = self.eval(model, test_loader)
                train_result = self.eval(model, train_loader)

                if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                    best_val_result = val_result
                    best_test_result = test_result
                    best_train_result = train_result
                    best_epoch = epoch

                if epoch - best_epoch >= 20:
                    break
        # if self.local_rank == 0:
        #     self.save_losses_to_csv()  # 保存 loss 到 CSV 文件
        return best_train_result, best_val_result, best_test_result

    def eval(self, model, dataloader):
        model.eval()
        predictions_all = []
        labels_all = []
        for batched_data in dataloader:
            predictions, predictions_1, predictions_2, labels, triplet_h, embeddings1, embeddings = self._forward_epoch(
                model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
        result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        return result

    def save_losses_to_csv(self, file_path=None):
        if file_path is None:
            file_path = self.loss_log_path

        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Loss'])  # 表头
            for idx, loss in enumerate(self.train_losses):
                writer.writerow([idx + 1, loss])
        print(f"Loss values saved to {file_path}")