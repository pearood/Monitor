"""
UHMF: Uncertainty-Aware Hierarchical Meta-Fusion
不确定性感知的层次化元融合 - 用于动态融合

核心创新点:
1. 层次化场景感知: 宏观场景识别 + 微观状态适应
2. 双层不确定性估计: 模态不确定性 + 数据不确定性
3. 层次化元学习框架: 外层场景适应 + 内层实时微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HierarchicalSceneEncoder(nn.Module):
    """层次化场景编码器"""
    
    def __init__(self, input_dim, hidden_dim=128, n_scenes=3):
        super().__init__()
        
        self.n_scenes = n_scenes
        
        # 宏观场景编码器
        self.macro_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 场景分类器
        self.scene_classifier = nn.Linear(hidden_dim, n_scenes)
        
        # 微观状态编码器
        self.micro_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, features):
        """
        Args:
            features: [B, D] 融合前的特征
        Returns:
            macro_encoding: [B, H] 宏观场景编码
            micro_encoding: [B, H] 微观状态编码
            scene_logits: [B, n_scenes] 场景分类logits
        """
        macro_encoding = self.macro_encoder(features)
        micro_encoding = self.micro_encoder(features)
        scene_logits = self.scene_classifier(macro_encoding)
        
        return macro_encoding, micro_encoding, scene_logits


class DualUncertaintyEstimator(nn.Module):
    """双层不确定性估计器"""
    
    def __init__(self, input_dim, n_modalities=3, n_scenes=3, mc_samples=5):
        super().__init__()
        
        self.n_modalities = n_modalities
        self.n_scenes = n_scenes
        self.mc_samples = mc_samples
        
        # 模态不确定性: 场景-模态可靠性矩阵
        self.modal_reliability = nn.Parameter(
            torch.ones(n_scenes, n_modalities) / n_modalities
        )
        
        # 数据不确定性估计网络 (使用MC Dropout)
        self.data_uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # MC Dropout
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_modalities)
        )
        
        self.temperature = 3.0  # 更强平滑（对齐exp2稳定策略）
        
    def forward(self, modality_features, scene_probs, training=True):
        """
        Args:
            modality_features: list of [B, D_i] 各模态特征
            scene_probs: [B, n_scenes] 场景概率
        Returns:
            modal_uncertainty: [B, n_modalities] 模态不确定性
            data_uncertainty: [B, n_modalities] 数据不确定性
        """
        B = scene_probs.shape[0]
        
        # 模态不确定性: 基于场景的先验
        # reliability: [n_scenes, n_modalities]
        # scene_probs: [B, n_scenes]
        modal_reliability = torch.softmax(self.modal_reliability, dim=-1)
        modal_uncertainty_raw = 1 - torch.mm(scene_probs, modal_reliability)  # [B, n_modalities]
        
        # Apply temperature smoothing to modal_uncertainty to make weights more balanced
        modal_uncertainty = torch.pow(modal_uncertainty_raw, 1.0 / self.temperature)
        
        # Add epsilon to modal_uncertainty to prevent completely dropping weak modalities
        epsilon = 1e-4
        modal_uncertainty = torch.clamp(modal_uncertainty, min=epsilon)
        
        # 数据不确定性: 使用MC Dropout估计
        concat_features = torch.cat(modality_features, dim=-1)
        
        if training:
            # 多次采样估计方差
            samples = []
            for _ in range(self.mc_samples):
                sample = self.data_uncertainty_net(concat_features)
                samples.append(sample)
            samples = torch.stack(samples, dim=0)  # [mc_samples, B, n_modalities]
            data_uncertainty = samples.var(dim=0)  # [B, n_modalities]
        else:
            # 推理时使用单次前向
            data_uncertainty = torch.sigmoid(self.data_uncertainty_net(concat_features))
        
        return modal_uncertainty, data_uncertainty


class MetaFusionNetwork(nn.Module):
    """元融合网络"""
    
    def __init__(self, hidden_dim=128, n_modalities=3):
        super().__init__()
        
        self.n_modalities = n_modalities
        self.temperature = 1.2  # 提高温度，防止权重过度集中
        
        # 权重生成网络
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2 + n_modalities * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_modalities)
        )
        
    def forward(self, macro_encoding, micro_encoding, modal_uncertainty, data_uncertainty):
        """
        生成融合权重
        Args:
            macro_encoding: [B, H] 宏观场景编码
            micro_encoding: [B, H] 微观状态编码
            modal_uncertainty: [B, n_modalities] 模态不确定性
            data_uncertainty: [B, n_modalities] 数据不确定性
        Returns:
            weights: [B, n_modalities] 融合权重
        """
        # 拼接所有信息
        combined = torch.cat([
            macro_encoding, 
            micro_encoding, 
            modal_uncertainty, 
            data_uncertainty
        ], dim=-1)
        
        # 生成权重
        raw_weights = self.weight_generator(combined)
        weights = F.softmax(raw_weights / self.temperature, dim=-1)

        # 防止某一模态权重塌陷（如 text 独占），但不过度限制
        weights = torch.clamp(weights, min=0.15)  # 提高下限，防止text独占
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights


class UHMF(nn.Module):
    """
    UHMF: 不确定性感知的层次化元融合
    
    完整的动态融合框架
    """
    
    def __init__(self, video_dim=512, audio_dim=256, text_dim=768,
                 hidden_dim=128, n_scenes=3, n_classes=4, modal_dropout=0.1):
        super().__init__()
        
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.n_modalities = 3
        
        # 模态投影层
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # LayerNorm for each modality (防止某一模态数值主导)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)  # 提高正则（对齐exp2）
        
        # 层次化场景编码器
        self.scene_encoder = HierarchicalSceneEncoder(
            hidden_dim * 3, hidden_dim, n_scenes
        )
        
        # 双层不确定性估计器
        self.uncertainty_estimator = DualUncertaintyEstimator(
            hidden_dim * 3, self.n_modalities, n_scenes
        )
        
        # 元融合网络
        self.meta_fusion = MetaFusionNetwork(hidden_dim, self.n_modalities)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )
        
        # 模态dropout概率
        self.modal_dropout_prob = 0.2  # 增强模态随机失活（关键）
        
    def forward(self, video_feat, audio_feat, text_feat, return_weights=False):
        """
        前向传播
        Args:
            video_feat: [B, video_dim]
            audio_feat: [B, audio_dim]
            text_feat: [B, text_dim]
        Returns:
            logits: [B, n_classes] 分类logits
            weights: [B, 3] 融合权重（可选）
        """
        # 投影到统一维度
        v = self.dropout(self.norm_v(self.video_proj(video_feat)))  # [B, H]
        a = self.dropout(self.norm_a(self.audio_proj(audio_feat)))  # [B, H]
        t = self.dropout(self.norm_t(self.text_proj(text_feat)))  # [B, H]
        
        # 拼接特征用于场景编码
        concat_feat = torch.cat([v, a, t], dim=-1)  # [B, 3H]
        
        # 层次化场景编码
        macro_enc, micro_enc, scene_logits = self.scene_encoder(concat_feat)
        scene_probs = F.softmax(scene_logits, dim=-1)
        
        # 双层不确定性估计
        modal_unc, data_unc = self.uncertainty_estimator(
            [v, a, t], scene_probs, self.training
        )
        
        # 元融合权重生成
        weights = self.meta_fusion(macro_enc, micro_enc, modal_unc, data_unc)
        
        # 模态dropout
        if self.training:
            dropout_mask = (torch.rand(weights.size(), device=weights.device) > self.modal_dropout_prob).float()
            weights = weights * dropout_mask
            # 重新归一化权重
            weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-8
            weights = weights / weights_sum
        
        # 加权融合
        modality_features = torch.stack([v, a, t], dim=1)  # [B, 3, H]
        fused = (modality_features * weights.unsqueeze(-1)).sum(dim=1)  # [B, H]
        
        # 分类
        logits = self.classifier(fused)
        
        if return_weights:
            return logits, weights, scene_probs, modal_unc, data_unc
        return logits
    
    def compute_loss(self, logits, labels, scene_logits=None, scene_labels=None, weights=None):
        """计算损失"""
        # 主分类损失
        cls_loss = F.cross_entropy(logits, labels)
        
        # 场景分类损失（如果有标签）
        if scene_logits is not None and scene_labels is not None:
            scene_loss = F.cross_entropy(scene_logits, scene_labels)
            total_loss = cls_loss + 0.1 * scene_loss
        else:
            total_loss = cls_loss

        # 融合权重熵正则化（加强，防止权重塌陷到单一模态）
        if weights is not None:
            eps = 1e-8
            entropy = - (weights * torch.log(weights + eps)).sum(dim=-1).mean()
            entropy_reg = -0.05 * entropy  # 提高约束强度（exp2核心技巧）
            total_loss = total_loss + entropy_reg
        
        return total_loss


class UHMFAblation(nn.Module):
    """UHMF消融实验变体"""
    
    def __init__(self, video_dim=512, audio_dim=256, text_dim=768,
                 hidden_dim=128, n_classes=4,
                 use_hierarchy=True, use_modal_uncertainty=True, 
                 use_data_uncertainty=True, use_meta=True):
        super().__init__()
        
        self.use_hierarchy = use_hierarchy
        self.use_modal_uncertainty = use_modal_uncertainty
        self.use_data_uncertainty = use_data_uncertainty
        self.use_meta = use_meta
        
        # 模态投影
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        if use_hierarchy:
            self.scene_encoder = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)
            )
        
        if use_meta:
            self.weight_net = nn.Sequential(
                nn.Linear(hidden_dim * 3, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
        else:
            self.fixed_weights = nn.Parameter(torch.ones(3) / 3)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def forward(self, video_feat, audio_feat, text_feat):
        v = self.video_proj(video_feat)
        a = self.audio_proj(audio_feat)
        t = self.text_proj(text_feat)
        
        concat = torch.cat([v, a, t], dim=-1)
        
        if self.use_meta:
            weights = F.softmax(self.weight_net(concat), dim=-1)
        else:
            weights = F.softmax(self.fixed_weights, dim=-1).unsqueeze(0).expand(v.size(0), -1)
        
        modalities = torch.stack([v, a, t], dim=1)
        fused = (modalities * weights.unsqueeze(-1)).sum(dim=1)
        
        logits = self.classifier(fused)
        return logits
