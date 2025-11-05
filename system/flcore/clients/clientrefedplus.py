# ==============================================================================
# File: clientrefedplus.py
# Description: Client-side implementation for Re-Fed+ algorithm
# ==============================================================================

import torch
import torch.nn as nn
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client


class clientReFedPlus(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        # Replay buffer for incremental learning
        self.buffer_size = args.buffer_size if hasattr(args, 'buffer_size') else 100
        self.cache_ratio = args.cache_ratio if hasattr(args, 'cache_ratio') else 0.2

        # Initialize replay buffer
        self.replay_buffer = {
            'data': [],
            'labels': [],
            'importance': []
        }

    def train(self, task=None):
        """
        Train with replay: combine new task data with cached samples
        """
        self.trainloader = self.load_train_data(task=task)
        start_time = time.time()

        self.model.train()

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(self.trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass on new task data
                output = self.model(x)
                loss_new = self.loss(output, y)

                # Add replay loss if we have cached samples
                replay_x, replay_y = self.get_replay_data()
                if replay_x is not None:
                    replay_x = replay_x.to(self.device)
                    replay_y = replay_y.to(self.device)

                    # Randomly sample from replay buffer for efficiency
                    batch_size = min(len(replay_x), x.size(0))
                    indices = torch.randperm(len(replay_x))[:batch_size]
                    replay_batch_x = replay_x[indices]
                    replay_batch_y = replay_y[indices]

                    output_replay = self.model(replay_batch_x)
                    loss_replay = self.loss(output_replay, replay_batch_y)

                    # Combined loss with replay
                    loss = loss_new + loss_replay
                else:
                    loss = loss_new

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def compute_importance_scores(self, data, labels, global_model):
        """
        Compute importance scores for samples based on local/global disagreement
        and prediction confidence
        """
        self.model.eval()
        global_model.eval()

        importance_scores = []

        with torch.no_grad():
            for x, y in zip(data, labels):
                x = x.unsqueeze(0).to(self.device)
                y_tensor = torch.tensor([y]).to(self.device)

                # Local model prediction
                local_output = self.model(x)
                local_prob = torch.softmax(local_output, dim=1)
                local_confidence = local_prob.gather(1, y_tensor.unsqueeze(1)).squeeze()

                # Global model prediction
                global_output = global_model(x)
                global_prob = torch.softmax(global_output, dim=1)
                global_confidence = global_prob.gather(1, y_tensor.unsqueeze(1)).squeeze()

                # Disagreement between local and global
                disagreement = torch.sum(torch.abs(local_prob - global_prob))

                # Combined importance: low confidence + high disagreement = important
                importance = (1 - local_confidence) * 0.3 + \
                             (1 - global_confidence) * 0.3 + \
                             disagreement * 0.4

                importance_scores.append(importance.item())

        return torch.tensor(importance_scores)

    def cache_important_samples(self, global_model):
        """
        Cache important samples from current training data for future replay
        """
        if len(self.trainloader.dataset) == 0:
            return

        # Get all training data
        all_data = []
        all_labels = []
        for x, y in self.trainloader:
            all_data.extend(x)
            all_labels.extend(y.cpu().numpy())

        num_to_cache = int(len(all_data) * self.cache_ratio)
        if num_to_cache == 0:
            return

        # Compute importance scores
        importance_scores = self.compute_importance_scores(all_data, all_labels, global_model)

        # Select top-k most important samples
        top_indices = torch.topk(importance_scores, min(num_to_cache, len(all_data))).indices

        # Add to replay buffer
        for idx in top_indices:
            self.replay_buffer['data'].append(all_data[idx].cpu())
            self.replay_buffer['labels'].append(all_labels[idx])
            self.replay_buffer['importance'].append(importance_scores[idx].item())

        # Maintain buffer size limit
        if len(self.replay_buffer['data']) > self.buffer_size:
            # Keep only the most important samples
            sorted_indices = np.argsort(self.replay_buffer['importance'])[-self.buffer_size:]
            self.replay_buffer['data'] = [self.replay_buffer['data'][i] for i in sorted_indices]
            self.replay_buffer['labels'] = [self.replay_buffer['labels'][i] for i in sorted_indices]
            self.replay_buffer['importance'] = [self.replay_buffer['importance'][i] for i in sorted_indices]

    def get_replay_data(self):
        """
        Get cached replay data as tensors
        """
        if len(self.replay_buffer['data']) == 0:
            return None, None

        replay_x = torch.stack(self.replay_buffer['data'])
        replay_y = torch.tensor(self.replay_buffer['labels'])
        return replay_x, replay_y