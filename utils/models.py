import torch.nn as nn
import torchvision

"""
    Modèle d'extraction de caractéristiques, ici en faisant appel à VGG-16.
"""
import torch
import torch.nn as nn
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# class FeatureExtractor(nn.Module):
#     def __init__(self,
#                  pretrained_model_name: str = "Salesforce/instructblip-flan-t5-small"):
#         super().__init__()
#         # 同时处理 image + instruction
#         self.processor = InstructBlipProcessor.from_pretrained(pretrained_model_name)
#         self.vlm = InstructBlipForConditionalGeneration.from_pretrained(pretrained_model_name)
#         # 用 text encoder 的 hidden size 作为输出维度（可改为 decoder hidden size）
#         self.features_dim = self.vlm.config.text_hidden_size

#     def forward(self, imgs: torch.Tensor, prompts: list[str]) -> torch.Tensor:
#         device = imgs.device
#         # 1) 预处理
#         inputs = self.processor(images=imgs, text=prompts,
#                                 padding=True, return_tensors="pt").to(device)
#         # 2) 只跑编码器部分，拿最后一层 CLS token
#         encoder_outputs = self.vlm.text_encoder(
#             input_ids=inputs.input_ids,
#             attention_mask=inputs.attention_mask,
#             encoder_hidden_states=self.vlm.vision_model(
#                 pixel_values=inputs.pixel_values
#             ).last_hidden_state,
#             encoder_attention_mask=torch.ones_like(inputs.input_ids),
#             return_dict=True
#         )
#         # CLS token 在位置 0
#         cls_feat = encoder_outputs.last_hidden_state[:, 0, :]  # (B, hidden)
#         return cls_feat

    
# """
#     Architecture du Q-Network comme décrite dans l'article.
# """
# # class DQN(nn.Module):
# #     def __init__(self, h, w, outputs):
# #         super(DQN, self).__init__()
# #         self.classifier = nn.Sequential(
# #             nn.Linear( in_features= 81 + 25088, out_features=1024),
# #             nn.ReLU(),
# #             nn.Dropout(0.2),
# #             nn.Linear( in_features= 1024, out_features=1024),
# #             nn.ReLU(),
# #             nn.Dropout(0.2),
# #             nn.Linear( in_features= 1024, out_features=9)
# #         )
# #     def forward(self, x):
# #         return self.classifier(x)

# # ---------- 2. Dueling Deep Q-Network ----------
# class DQN(nn.Module):
#     def __init__(self, features_dim: int, hist_dim: int = 81, n_actions: int = 9,
#                  hidden: int = 1024):
#         super().__init__()
#         state_dim = features_dim + hist_dim
#         self.fc_shared = nn.Sequential(
#             nn.Linear(state_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
#         # value branch
#         self.fc_val = nn.Sequential(
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden, 1)
#         )
#         # advantage branch
#         self.fc_adv = nn.Sequential(
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden, n_actions)
#         )

#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         x = self.fc_shared(state)
#         val = self.fc_val(x)
#         adv = self.fc_adv(x)
#         return val + adv - adv.mean(dim=1, keepdim=True)  # (B, n_actions)



# 以上是第一版，拼接版本，以下是第二版，cross attention版本
import torch
import torch.nn as nn

import torch
import torch.nn as nn
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    ViTModel,
    ViTConfig
)

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        blip_model_name: str = "Salesforce/instructblip-flan-t5-xl",
        vit_model_name: str = "google/vit-base-patch16-224-in21k",
        cross_attn_heads: int = 8
    ):
        super().__init__()
        # —— 1) BLIP prompt+image 表示  —— 
        self.blip_processor = InstructBlipProcessor.from_pretrained(blip_model_name)
        self.blip        = InstructBlipForConditionalGeneration.from_pretrained(
            blip_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.blip_dim    = self.blip.config.text_config.d_model

        # —— 2) Vision Transformer 表示  —— 
        self.vit       = ViTModel.from_pretrained(vit_model_name)
        self.vit_dim   = self.vit.config.hidden_size

    def forward(self, imgs: torch.Tensor, prompts = [
        "Here is an image cropped by the current bounding box. The agent may choose one of the following nine actions—(1) move_left, (2) move_right, (3) move_up, (4) move_down, (5) scale_up, (6) scale_down, (7) widen, (8) heighten, or (9) trigger—so select the single most appropriate action."
    ]
    ) -> torch.Tensor:
        device = imgs.device

        # —— A) Prompt+Image 表示 —— 
        blip_inputs = self.blip_processor(
            images=imgs, text=prompts,
            padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            gen_out = self.blip.generate(
                **blip_inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
        # 解包最后一层 decoder hidden states (tuple of length=1)
        last_layer = gen_out.decoder_hidden_states[-1]
        if isinstance(last_layer, tuple):
            last_layer = torch.stack(last_layer, dim=1)  # [B, 1, d_model]
        blip_rep = last_layer[:, -1, :]                # [B, blip_dim]

        # —— B) ViT 表示 —— 
        vit_outputs = self.vit(pixel_values=blip_inputs.pixel_values)
        vit_rep = vit_outputs.last_hidden_state[:, 0, :]  # [B, vit_dim]

        blip_rep = blip_rep.squeeze(1)
        # print('The shape of blip repre is:', blip_rep.shape)
        # print('The shape of vit representation is:', vit_rep.shape)
        fused = torch.cat([blip_rep, vit_rep], dim=-1) 
        out   = fused                                 # [B, blip_dim]
        return out
class DQN(nn.Module):
    def __init__(
        self,
        features_dim: int,      # = blip_dim + vit_dim, e.g. 2048+768=2816
        hist_dim: int = 81,
        n_actions: int = 9,
        hidden: int = 1024,
        cross_attn_heads: int = 8
    ):
        super().__init__()
        # —— 保存维度信息 —— 
        self.features_dim = features_dim
        self.hist_dim     = hist_dim
        
        # 这两个常量要跟 FeatureExtractor 输出对齐
        self.blip_dim = 2048                 # prompt+image 表示维度
        # self.vit_dim  = features_dim - 1000   # ViT CLS token 维度
        self.vit_dim = 768
        
        # —— cross-attn & 投影层 —— 
        # 把 vit_dim 投影到 blip_dim
        self.kv_proj    = nn.Linear(self.vit_dim, self.blip_dim)
        # Query=blip, Key/Value=proj(vit)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.blip_dim,
            num_heads=cross_attn_heads,
            batch_first=True
        )
        
        # —— 原有 DQN 结构，输入维度保持不变 —— 
        # state_dim = features_dim + hist_dim
        state_dim = 2892
        self.fc_shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1)
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        输入 state: [B, blip_dim+vit_dim + hist_dim]
        输出 Q:      [B, n_actions]
        """
        # 1) 拆解原始 state
        # feat    = state[:, :self.features_dim]
        feat    = state          # [B, blip+vit]
        print('feat in DQN is:', feat.shape)
        history = state[:, 2048+768:]          # [B, hist_dim]
        
        # 2) 再拆成两部分
        blip = feat[:, :self.blip_dim]                  # [B, blip_dim]
        vit  = feat[:, self.blip_dim:2048+768] 

        # print('blip in DQN is:', blip.shape)
        # print('vit feat in DQN is:', vit.shape)               # [B, vit_dim]
        
        # 3) cross-attn 融合
        #  - Query: [B, 1, blip_dim]
        #  - Key/Value: project(vit) → [B, 1, blip_dim]
        q = blip.unsqueeze(1)
        kv = self.kv_proj(vit).unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)        # [B, 1, blip_dim]
        attn_rep = attn_out.squeeze(1)                  # [B, blip_dim]
        
        # 4) 替换原 blip 部分，但保持 feat 总维度不变
        #    new_feat = [attn_rep │ vit] → 仍然是 [B, blip_dim+vit_dim]
        new_feat = torch.cat([attn_rep, vit], dim=1)

        # 5) 拼回 history，重新构造 state
        new_state = torch.cat([new_feat, history], dim=1)  # [B, features_dim+hist_dim]
        
        # # 6) 跟原来一模一样的前向计算
        # x = self.fc_shared(new_state)
        # val = self.fc_val(x)
        # adv = self.fc_adv(x)
        return new_state