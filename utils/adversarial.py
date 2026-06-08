import torch
import torch.nn as nn


class SelfAdversarialTrainer:
    def __init__(self, model, optimizer, device, sat_config):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.cfg = sat_config
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=sat_config.get('label_smoothing', 0.1)
        )

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
                x_adv = torch.clamp(
                    x_adv + torch.randn_like(x_adv) * (epsilon * 0.5), 0, 1
                )

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

        use_sat = (
                self.cfg.get('use_sat', True) and
                torch.rand(1).item() < self.cfg.get('sat_prob', 0.6)
        )

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