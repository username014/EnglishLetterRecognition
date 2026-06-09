import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialTTA:
    def __init__(self, model, device, epsilon=0.03, n_adv=3, attack_type='fgsm', pgd_steps=5, mean=0.1307, std=0.3081):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.n_adv = n_adv
        self.attack_type = attack_type
        self.pgd_steps = pgd_steps

        self.min_val = (0.0 - mean) / std  # ≈ -0.424
        self.max_val = (1.0 - mean) / std  # ≈ 2.821

    def predict(self, images):
        self.model.eval()
        images = images.to(self.device)

        with torch.no_grad():
            clean_logits = self.model(images)
            pseudo_labels = clean_logits.argmax(dim=1)

        logits_list = [clean_logits]

        for _ in range(self.n_adv):
            x_adv = images.clone().detach().requires_grad_(True)
            noise = torch.randn_like(x_adv) * (self.epsilon * 0.5)
            x_adv = torch.clamp(x_adv + noise, self.min_val, self.max_val)

            if self.attack_type == 'pgd':
                alpha = self.epsilon / max(self.pgd_steps, 1)
                for _ in range(self.pgd_steps):
                    out = self.model(x_adv)
                    loss = F.cross_entropy(out, pseudo_labels)
                    grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]

                    with torch.no_grad():
                        x_adv = x_adv + alpha * grad.sign()
                        x_adv = torch.clamp(x_adv, images - self.epsilon, images + self.epsilon)
                        x_adv = torch.clamp(x_adv, self.min_val, self.max_val)
                    x_adv = x_adv.detach()
            else:
                out = self.model(x_adv)
                loss = F.cross_entropy(out, pseudo_labels)
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]

                with torch.no_grad():
                    x_adv = x_adv + self.epsilon * grad.sign()
                    x_adv = torch.clamp(x_adv, self.min_val, self.max_val)
                x_adv = x_adv.detach()

            with torch.no_grad():
                adv_logits = self.model(x_adv)
                logits_list.append(adv_logits)

        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return avg_logits


class SelfAdversarialTrainer:
    def __init__(self, model, optimizer, device, sat_config):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.cfg = sat_config
        self.criterion = nn.CrossEntropyLoss(label_smoothing=sat_config.get('label_smoothing', 0.1))

    def _get_epsilon(self, epoch, total_epochs):
        if self.cfg.get('adaptive_epsilon', False):
            eps_min = self.cfg.get('eps_min', 0.01)
            eps_max = self.cfg.get('eps_max', 0.05)
            return eps_min + (eps_max - eps_min) * (epoch / total_epochs)
        return self.cfg.get('epsilon', 0.03)

    def _generate_adv_examples(self, images, labels, epsilon, n_adv=1):
        adv_losses = []
        attack_type = self.cfg.get('attack_type', 'fgsm')
        pgd_steps = self.cfg.get('pgd_steps', 5)

        for _ in range(n_adv):
            x_adv = images.clone()
            if n_adv > 1:
                x_adv = torch.clamp(x_adv + torch.randn_like(x_adv) * (epsilon * 0.5), 0, 1)

            if attack_type == 'pgd':
                alpha = epsilon / max(pgd_steps, 1)
                for _ in range(pgd_steps):
                    x_adv.requires_grad_(True)
                    out = self.model(x_adv)
                    loss_temp = self.criterion(out, labels)
                    grad = torch.autograd.grad(loss_temp, x_adv, retain_graph=False)[0]
                    with torch.no_grad():
                        x_adv = x_adv + alpha * grad.sign()
                        x_adv = torch.clamp(x_adv, images - epsilon, images + epsilon)
                        x_adv = torch.clamp(x_adv, 0, 1)
                    x_adv = x_adv.detach()
            else:
                x_adv.requires_grad_(True)
                out = self.model(x_adv)
                loss_temp = self.criterion(out, labels)
                grad = torch.autograd.grad(loss_temp, x_adv, retain_graph=False)[0]
                with torch.no_grad():
                    x_adv = x_adv + epsilon * grad.sign()
                    x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = x_adv.detach()

            out_adv = self.model(x_adv)
            adv_losses.append(self.criterion(out_adv, labels))

        return torch.stack(adv_losses).mean()

    def train_step(self, images, labels, epoch, total_epochs):
        self.model.train()
        images, labels = images.to(self.device), labels.to(self.device)

        use_sat = self.cfg.get('use_sat', True) and torch.rand(1).item() < self.cfg.get('sat_prob', 0.6)

        if use_sat:
            eps = self._get_epsilon(epoch, total_epochs)
            n_adv = self.cfg.get('n_adversarial', 1)
            loss = self._generate_adv_examples(images, labels, eps, n_adv)
        else:
            out = self.model(images)
            loss = self.criterion(out, labels)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()