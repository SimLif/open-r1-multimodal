import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed

from transformers import DynamicCache, Cache
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, \
    _prepare_4d_causal_attention_mask_for_sdpa
from transformers import Qwen2VLConfig, Qwen2VLModel, Qwen2VLForConditionalGeneration
from transformers.utils.hub import cached_file

from transformers.modeling_outputs import CausalLMOutputWithPast

from deepspeed.moe.layer import MoE
from deepspeed.moe.sharded_moe import TopKGate
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_vl.modeling_qwen2_vl import logger
from transformers.utils import ModelOutput
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

local_rank = None

# try:
#     from .utils import TopKGateDynamic
# except:
#     pass
TopKGateDynamic = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def generate_routing_tensor(N, num_experts, k, non_uniform=True, device=None, dtype=torch.float):
    """
    生成形状为 [N, num_experts] 的随机 tensor，用于模拟路由结果，
    每一行随机选择 k 个专家位置，其余位置置为 0，且非零值归一化后和为 1。
    
    Args:
        N (int): 行数，例如 batch_size.
        num_experts (int): 专家总数，即每行维度.
        k (int): 每行选取的专家数量.
        non_uniform (bool): 若为 True，则 top-k 位置分配随机值（非均匀分布）；否则均匀分布（相当于 softmax(0)=均匀分布）。
        device (torch.device, optional): 设备，默认为 CPU.
        dtype (torch.dtype, optional): 数据类型，默认为 torch.float.
    
    Returns:
        Tensor: 形状为 [N, num_experts] 的路由结果，每行只有 k 个非零元素，和为 1.
    """
    if device is None:
        device = torch.device("cpu")

    # 1. 随机生成每一行的分数
    scores = torch.rand(N, num_experts, device=device, dtype=dtype)
    
    # 2. 每一行选取 top k 个位置作为专家 (不放回采样)
    _, topk_indices = scores.topk(k, dim=1, largest=True, sorted=False)  # shape: [N, k]
    
    # 3. 构造一个全 -inf 的矩阵，使得未选的位置 softmax 后概率为 0
    masked_scores = torch.full((N, num_experts), float('-inf'), device=device, dtype=dtype)
    
    # 4. 为 top-k 位置赋值（随机或均值均可）
    if non_uniform:
        topk_values = torch.rand(N, k, device=device, dtype=dtype)
    else:
        topk_values = torch.zeros(N, k, device=device, dtype=dtype)
    
    # 将 top-k 的值填充到对应位置
    masked_scores.scatter_(1, topk_indices, topk_values)
    
    # 5. softmax 计算路由概率，未选位置由于 exp(-inf)=0，会产生只有 k 个非零概率且和为 1 的分布
    routing_tensor = F.softmax(masked_scores, dim=1)
    
    return routing_tensor


# 定义小专家类
# class SmallExpert(nn.Module):
#     def __init__(self, hidden_size, r, dropout_rate=0.0):
#         super().__init__()
#         self.down_proj = nn.Linear(hidden_size, r, bias=False)
#         self.act_fn = nn.SiLU()  # 与 Qwen2 MLP 保持一致
#         self.dropout = nn.Dropout(dropout_rate)  # 可选的 dropout
#         self.up_proj = nn.Linear(r, hidden_size, bias=False)
        
#     def forward(self, x):
#         x = self.down_proj(x)
#         x = self.act_fn(x)
#         x = self.dropout(x)
#         x = self.up_proj(x)
#         return x

class SmallExpert(nn.Module):
    def __init__(self, hidden_size, r, dropout_rate=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = r
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    

# 定义组合层，结合原始MLP和MoE
class CombinedLayer(nn.Module):
    def __init__(self, shared, moe, use_gate=False, gate_type='ds',
                 gate_dropout=0.0, hidden_size=0, cmr_target=0.8, structure='new', kd_align=False):
        """
        Args:
            shared: 共享的 MLP（或 FFN）模块
            moe: MoE 层模块，注意其 forward 返回值可能是 (output, aux_loss) 的元组
            use_gate: 是否使用 Gate 来动态路由流经哪条分支
            gate_type: Gate 类型，'ds' 表示双分支（dense & MoE）使用 softmax 得到两个权重，
                       'cmr' 表示使用单分支 gate，且会计算额外辅助损失（路由预算约束）
            gate_dropout: 在 `cmr` 模型中，训练时随机将一部分 gate 输出置 0 的概率，
                          用以强制部分 tokens 仅走共享分支
            hidden_size: 输入的隐藏层尺寸，用于构造 gate 层
            cmr_target: 对于 cmr gate，预算约束的目标概率值 (例如 0.8 表示希望 MoE 分支占 80%)
        """
        super().__init__()
        self.structure = structure
        if structure == 'new':
            self.shared = shared
            self.moe = moe
        elif structure == 'old':
            self.original_mlp = shared
            self.moe_layer = moe
        self.use_gate = use_gate
        self.gate_type = gate_type
        self.gate_dropout = gate_dropout
        self.cmr_target = cmr_target
        self.kd_align = kd_align
        
        if use_gate:
            if self.gate_type == 'ds':
                # 输出2个logits，分别对应共享分支（FFNshared）和MoE分支
                self.gate = nn.Linear(hidden_size, 2, bias=False)
            elif self.gate_type == 'cmr':
                # 输出1个logit，之后经 sigmoid 映射到 (0,1) 得到 MoE 分支权重
                self.gate = nn.Linear(hidden_size, 1, bias=False)
            else:
                raise NotImplementedError(f"Gate type {self.gate_type} not implemented")
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状：[batch_size, ..., hidden_size]
        Returns:
            combined_out: 组合后的输出
            aux_loss_total: 如存在辅助损失，则返回。否则为 None
        """
        # 先分别计算共享层和 MoE 层的输出
        if self.structure == 'new':
            mlp_out = self.shared(x)
            moe_out = self.moe(x)
        elif self.structure == 'old':
            mlp_out = self.original_mlp(x)
            moe_out = self.moe_layer(x)
        
        # 处理 MoE 层可能返回 (output, aux_loss) 的情况
        if isinstance(moe_out, tuple) and len(moe_out) >= 2:
            base_moe_out, moe_aux_loss = moe_out[0], moe_out[1]
        else:
            base_moe_out, moe_aux_loss = moe_out, None
        
        if self.kd_align:
            align_loss = F.mse_loss(mlp_out.detach(), base_moe_out)
            return mlp_out, align_loss
        
        if self.use_gate:
            # 为了数值稳定性，先转换为 fp32 再送入 gate 层
            x_fp32 = x.float()
            
            if self.gate_type == 'ds':
                # 计算两个分支的 logits，并通过 softmax 得到权重
                gating_scores = torch.nn.functional.linear(x_fp32, weight=self.gate.weight.float(), bias=None)
                weights = F.softmax(gating_scores, dim=-1).to(x.dtype)
                
                # 第一个分量对应共享分支，第二个分量对应 MoE 层
                mlp_weight = weights[..., 0].unsqueeze(-1)
                moe_weight = weights[..., 1].unsqueeze(-1)
                
                combined_out = mlp_weight * mlp_out + moe_weight * base_moe_out
                aux_loss_gate = None
            
            elif self.gate_type == 'cmr':
                # 计算 gate 输出概率
                gating_scores = torch.nn.functional.linear(x_fp32, weight=self.gate.weight.float(), bias=None)
                gate_prob = torch.sigmoid(gating_scores).to(x.dtype)  # MoE 分支的选择概率
                
                # 如果在训练阶段，随机将一部分 tokens 的 gate 置 0，即强制这些 tokens 只走共享分支
                if self.training and self.gate_dropout > 0:
                    drop_mask = (torch.rand_like(gate_prob) < self.gate_dropout).to(x.dtype)
                    gate_prob = gate_prob * (1.0 - drop_mask)
                
                # 共享分支的权重为 (1 - gate_prob)
                mlp_weight = 1.0 - gate_prob
                combined_out = mlp_weight * mlp_out + gate_prob * base_moe_out
                
                # 计算门控辅助损失：鼓励 gate_prob 接近 cmr_target
                aux_loss_gate = torch.mean(torch.abs(gate_prob - self.cmr_target))
            else:
                raise NotImplementedError(f"Gate type {self.gate_type} not implemented")
            
            # 如果 MoE 层也返回了辅助损失，则将两部分加和
            if moe_aux_loss is not None and aux_loss_gate is not None:
                aux_loss_total = moe_aux_loss + aux_loss_gate
            elif moe_aux_loss is not None:
                aux_loss_total = moe_aux_loss
            else:
                aux_loss_total = aux_loss_gate
        else:
            # 若不使用 gate，则简单求和（或者你也可以用其他融合方式）
            combined_out = mlp_out + base_moe_out
            aux_loss_total = moe_aux_loss
        
        return combined_out, aux_loss_total


try:
    from deepspeed.utils import logger
    from deepspeed.utils.timer import SynchronizedWallClockTimer
except ImportError:
    # 创建简单的替代品以允许在没有DeepSpeed的情况下使用
    class DummyLogger:
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
    logger = DummyLogger()
    
    class SynchronizedWallClockTimer:
        def __init__(self): pass
        def __call__(self, name): return self
        def start(self): pass
        def stop(self): pass
        def elapsed(self, reset=True): return 0.0

def _one_hot_to_float(indices, num_classes):
    """将整数索引转换为one-hot浮点表示"""
    device = indices.device
    indices_shape = list(indices.shape)
    reshaped_indices = indices.reshape(-1)
    
    one_hot = torch.zeros(reshaped_indices.shape[0], num_classes,
                         device=device, dtype=torch.float)
    one_hot.scatter_(1, reshaped_indices.unsqueeze(1), 1)
    one_hot = one_hot.reshape(indices_shape + [num_classes])
    return one_hot


class SimilarityGate(nn.Module):
    """
    基于产品键的专家选择门控网络
    使用两个子键集合，为输入分配到高维专家空间
    """
    
    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        k: int = 2,
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        drop_tokens: bool = True,
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
        num_heads: int = 1,
        use_query_bn: bool = True,
    ) -> None:
        super().__init__()
        
        # 确保num_experts是完全平方数
        sqrt_experts = int(math.sqrt(num_experts))
        assert sqrt_experts * sqrt_experts == num_experts, f"专家数量必须是完全平方数，而不是{num_experts}"
        
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.sqrt_experts = sqrt_experts
        self.k = k
        self.num_heads = num_heads
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.drop_tokens = drop_tokens
        self.ep_group = ep_group
        
        # 每个头的查询维度计算
        head_dim = model_dim // num_heads
        self.head_dim = head_dim
        
        # 查询投影网络 - 产生用于专家选择的查询向量
        self.query_proj = nn.Linear(model_dim, head_dim * num_heads, bias=False)
        
        # 可选的查询批量归一化以提高稳定性
        self.use_query_bn = use_query_bn
        if use_query_bn:
            self.query_bn = nn.BatchNorm1d(head_dim * num_heads).float()
        
        # 初始化产品子键 - 两组子键用于构建完整的专家空间
        # 每组有sqrt_experts个子键，每个子键的维度是head_dim/2
        subkey_dim = head_dim // 2
        self.subkey_dim = subkey_dim
        
        # 子键初始化 - 使用正交初始化以提高检索有效性
        self.register_parameter(
            "sub_keys1", 
            nn.Parameter(torch.randn((sqrt_experts, subkey_dim)) / math.sqrt(subkey_dim), )
        )
        self.register_parameter(
            "sub_keys2", 
            nn.Parameter(torch.randn((sqrt_experts, subkey_dim)) / math.sqrt(subkey_dim))
        )
        
        # 正交化子键以提高区分度
        self._orthogonalize_keys()
        
        # 计时器用于性能分析
        self.timers = SynchronizedWallClockTimer() if 'SynchronizedWallClockTimer' in globals() else None
        self.wall_clock_breakdown = False
    
    def _orthogonalize_keys(self):
        """正交化子键，提高检索效率"""
        # 对子键1进行正交化
        u, s, v = torch.svd(self.sub_keys1)
        self.sub_keys1.data = u @ v.t()
        
        # 对子键2进行正交化
        u, s, v = torch.svd(self.sub_keys2)
        self.sub_keys2.data = u @ v.t()
    
    def _set_ep_group(self, ep_group):
        """设置专家并行组"""
        assert self.ep_group is None, '尝试覆盖已存在的ep_group'
        self.ep_group = ep_group
    
    def forward(self, input: torch.Tensor, used_token: Optional[torch.Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """专家选择的前向传播
        
        参数:
            input: 形状为[batch_size*seq_len, hidden_dim]的输入张量
            used_token: 可选的掩码，指示要处理的有效token
            
        返回:
            l_aux: 负载均衡损失
            combine_weights: 路由权重
            dispatch_mask: 调度掩码
            exp_counts: 每个专家被选择的次数
            selected_expert_indices: 选择的专家索引
        """
        if self.wall_clock_breakdown and self.timers is not None:
            self.timers("gate_timer").start()
        
        input = input.float()
        batch_tokens = input.shape[0]
        
        # 获取有效token数量
        valid_tokens = batch_tokens if used_token is None else used_token.sum().item()
        
        # 应用token掩码（如果提供）
        if used_token is not None:
            input = input * used_token.unsqueeze(-1)
        
        # 投影输入到查询空间
        # query = self.query_proj(input)
        query = F.linear(input, self.query_proj.weight.float(), bias=None)  # [batch_tokens, head_dim * num_heads * 2]

        # 应用批量归一化（如果启用）
        if self.use_query_bn:
            if self.training:
                with torch.no_grad():
                    self.query_bn.weight.data = self.query_bn.weight.data.float()
                    self.query_bn.bias.data = self.query_bn.bias.data.float()
                query = self.query_bn(query)
            else:
                # 推理模式下的批量归一化
                query = F.normalize(query, dim=-1)
        
        # 重塑查询以进行多头处理
        query = query.view(batch_tokens, self.num_heads, 2, self.head_dim // 2)
        
        # 分割查询用于产品键检索
        query1, query2 = query[:, :, 0], query[:, :, 1]  # 各自的形状: [batch_tokens, num_heads, head_dim//2]
        
        # 使用子键计算相似度分数
        # einsum更高效地计算批量相似度
        scores1 = torch.einsum('bhd,ed->bhe', query1, self.sub_keys1.float())  # [batch_tokens, num_heads, sqrt_experts]
        scores2 = torch.einsum('bhd,ed->bhe', query2, self.sub_keys2.float())  # [batch_tokens, num_heads, sqrt_experts]

        # 从两个子键集各获取k个最高分
        top_scores1, top_indices1 = torch.topk(scores1, k=min(self.k, self.sqrt_experts), dim=-1)  # [batch_tokens, num_heads, k]
        top_scores2, top_indices2 = torch.topk(scores2, k=min(self.k, self.sqrt_experts), dim=-1)  # [batch_tokens, num_heads, k]

        # 计算所有k²个可能组合的分数
        combined_scores = top_scores1.unsqueeze(-1) + top_scores2.unsqueeze(-2)  # [batch_tokens, num_heads, k, k]

        # 重塑并找到每个(token,head)的top-k专家
        flat_scores = combined_scores.reshape(batch_tokens, self.num_heads, -1)  # [batch_tokens, num_heads, k*k]
        top_k_scores, top_k_indices = torch.topk(flat_scores, k=self.k, dim=-1)  # [batch_tokens, num_heads, k]

        # 计算原始k*k网格中的行列索引
        i1_indices = top_k_indices // min(self.k, self.sqrt_experts)  # 获取行索引 [batch_tokens, num_heads, k]
        i2_indices = top_k_indices % min(self.k, self.sqrt_experts)   # 获取列索引 [batch_tokens, num_heads, k]

        # 使用这些索引获取实际的子键索引
        selected_i1 = torch.gather(top_indices1, dim=2, index=i1_indices)  # [batch_tokens, num_heads, k]
        selected_i2 = torch.gather(top_indices2, dim=2, index=i2_indices)  # [batch_tokens, num_heads, k]

        # 计算最终的专家索引
        expert_indices = selected_i1 * self.sqrt_experts + selected_i2  # [batch_tokens, num_heads, k]
        expert_scores = top_k_scores  # 直接使用topk返回的分数
        
        # 计算专家路由的概率权重
        # 在最后一个维度上应用softmax
        routing_probs = F.softmax(expert_scores, dim=-1)  # [batch_tokens, num_heads, k]
        
        # 扁平化结果，方便后续处理
        flat_expert_indices = expert_indices.reshape(batch_tokens, -1)  # [batch_tokens, num_heads*k]
        flat_routing_probs = routing_probs.reshape(batch_tokens, -1)    # [batch_tokens, num_heads*k]
        
        # 计算每个专家的选择次数 - 用于负载均衡和容量控制
        exp_counts = torch.zeros(self.num_experts, device=input.device, dtype=torch.long)
        router_prob_mass = torch.zeros(self.num_experts, device=input.device, dtype=flat_routing_probs.dtype)
        router_count_frac = torch.zeros(self.num_experts, device=input.device, dtype=flat_routing_probs.dtype)

        # 只考虑有效token
        if used_token is not None:
            valid_mask = used_token.reshape(-1).bool()
            valid_indices = flat_expert_indices[valid_mask]
            valid_probs = flat_routing_probs[valid_mask]
        else:
            valid_indices = flat_expert_indices
            valid_probs = flat_routing_probs
        
        # 更高效地计算专家计数
        hk = self.num_heads * self.k
        for i in range(hk):
            indices_slice = valid_indices[:, i]
            probs_slice = valid_probs[:, i]

            # 使用scatter_add_累加权重
            # exp_counts.scatter_add_(0, indices_slice, probs_slice)
            exp_counts.scatter_add_(0, indices_slice, torch.ones_like(indices_slice, dtype=torch.long))   
            # 累加路由概率
            router_prob_mass.scatter_add_(0, indices_slice, probs_slice / valid_tokens)
            # 累加专家分配计数
            router_count_frac.scatter_add_(
                0, 
                indices_slice,
                torch.ones_like(indices_slice, dtype=probs_slice.dtype) / valid_tokens
            )
        
        # 计算负载均衡损失 - 抑制专家不平衡
        # 乘以num_experts^2 / hk使得损失与模型规模和选择专家数量无关
        l_aux = torch.mean(router_prob_mass * router_count_frac) * self.num_experts * self.num_experts / hk
        
        # 处理专家容量约束
        # 创建掩码用于表示哪些专家选择有效
        capacity_mask = torch.ones_like(flat_routing_probs, dtype=torch.bool)
        
        if self.drop_tokens:
            # 根据训练/推理模式选择容量因子
            capacity_factor_ = self.capacity_factor if self.training else self.eval_capacity_factor
            # 计算每个专家的容量上限，注意这里使用的是 valid_tokens 数量
            capacity = max(
                self.min_capacity, 
                int(capacity_factor_ * valid_tokens * hk / self.num_experts)
            )
            
            # --- 向量化实现：为每个 token-expert assignment 计算容量掩码 ---
            # 展平对应于每个 token 分配的专家索引和路由概率，形状均为 [batch_tokens * hk]
            flat_routing_probs_flat = flat_routing_probs.flatten()
            flat_expert_indices_flat = flat_expert_indices.flatten()
            
            # 初始化一个全 False 的容量掩码
            # capacity_mask_flat = torch.zeros_like(flat_routing_probs_flat, dtype=torch.bool)
            
            # 1. 构造复合排序键：确保同一专家内概率较高的分配排在前面
            LARGE_CONST = 1e2  # 要确保此常数足够大，可以覆盖 routing_prob 的取值范围
            composite_key = flat_expert_indices_flat.to(flat_routing_probs_flat.dtype) * LARGE_CONST - flat_routing_probs_flat

            # 2. 对 composite_key 进行全局排序，得到排序索引
            # 排序后，相同 expert 的所有分配会聚在一起，并且在组内顺序是概率降序的
            sorted_indices = torch.argsort(composite_key, stable=True)
            sorted_expert_ids = flat_expert_indices_flat[sorted_indices]

            # 3. 计算每个专家组在全局排序中的起始位置
            # 利用 unique_consecutive 获得每个 expert 的第一次出现位置
            unique_experts, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
            # 计算每个专家组的起始索引：例如，对于一个出现次数为 counts 的组，第一项起始索引为 0, 0+counts[0], 0+counts[0]+counts[1], ...
            starts = torch.cumsum(torch.cat([torch.tensor([0], device=sorted_expert_ids.device, dtype=torch.long), counts[:-1]]), dim=0)
            # 构造一个大小为 [num_experts] 的张量，默认值设为总分配数量（保证未出现的 expert 默认很大）
            group_first = torch.full((self.num_experts,), sorted_indices.numel(), device=flat_expert_indices_flat.device, dtype=torch.long)
            group_first[unique_experts] = starts

            # 4. 计算全局排序中每个分配的“组内 rank”
            ranks = torch.empty_like(sorted_indices, dtype=torch.long)
            # 令 ranks[sorted_indices] = [0, 1, 2, ... N-1]
            ranks[sorted_indices] = torch.arange(sorted_indices.size(0), device=flat_expert_indices_flat.device)
            # 计算组内排名：对于每个分配 i，对应的 expert id 为 flat_expert_indices_flat[i]
            group_ranks = ranks - group_first[flat_expert_indices_flat]

            # 5. 构造容量掩码：仅保留组内排名小于 capacity 的那些分配
            capacity_mask_flat = group_ranks < capacity
            # 恢复原始形状（例如原本的形状为 [batch_tokens, hk]）
            capacity_mask = capacity_mask_flat.view(flat_routing_probs.shape)
            
            # 依据容量掩码屏蔽超出容量的分配，并重新归一化概率
            masked_probs = flat_routing_probs * capacity_mask
            prob_sums = torch.sum(masked_probs, dim=-1, keepdim=True)
            # 避免除零错误
            prob_sums = torch.clamp(prob_sums, min=torch.finfo(masked_probs.dtype).eps)
            renormalized_probs = masked_probs / prob_sums
            
            # --- 计算 dispatch 位置 ---
            # 1. 获取所有满足容量限制（有效）的 token 索引
            valid_idx = torch.nonzero(capacity_mask_flat, as_tuple=True)[0]  # [M], M <= N

            # 2. 提取这些 token 对应的 expert id
            valid_experts = flat_expert_indices_flat[valid_idx]  # [M]

            # 3. 为了对每个 expert 内的 token 按 token 原始顺序排序（保证组内顺序正确），
            #    我们构造一个复合键：expert_id * (num_tokens+1) + token_index
            num_tokens = flat_expert_indices_flat.numel()
            composite_key = valid_experts.to(valid_idx.dtype) * (num_tokens + 1) + valid_idx

            # 4. 按复合键进行稳定排序，这样同一 expert 内元素会按照 token 顺序排列
            perm = torch.argsort(composite_key, stable=True)
            sorted_valid_idx = valid_idx[perm]         # 排序后 token 在 flat_expert_indices_flat 中的索引
            sorted_valid_experts = valid_experts[perm]   # 对应的 expert id

            # 5. 计算每个 expert 组内的累计位置
            #    由于 sorted_valid_experts 中同一 expert 的 token是连续的，
            #    可使用 unique_consecutive 获得每组的大小
            unique_experts, counts = torch.unique_consecutive(sorted_valid_experts, return_counts=True)
            # 计算每个组在排序数组中的起始位置，例如 [0, counts[0], counts[0]+counts[1], ...]
            starts = torch.cumsum(
                torch.cat([torch.tensor([0], device=sorted_valid_idx.device, dtype=torch.long), counts[:-1]]),
                dim=0
            )
            # 将每个组的起始位置扩展到每个 token（repeat每个起始值 count 次）
            group_start = torch.repeat_interleave(starts, counts)
            # 每个 token 在自己所属组内的累积计数，即其在排序数组中的索引减去该组的起始索引
            group_rank_sorted = torch.arange(sorted_valid_idx.size(0), device=sorted_valid_idx.device) - group_start

            # 6. 创建 flat_locations，并将计算好的组内累计位置“散射”回原来的位置
            flat_locations = torch.zeros_like(flat_expert_indices_flat, dtype=torch.long)
            flat_locations[sorted_valid_idx] = group_rank_sorted

            # 7. 恢复原始形状（例如原始形状为 [batch_tokens, hk]）
            locations = flat_locations.view(flat_expert_indices.shape)  # 若 flat_expert_indices 的原始形状保存在其它变量中，就用该形状
            
            # 利用 one-hot 编码将位置转换为 dispatch 权重表示（每个专家最多 capacity 个位置）
            locations_one_hot = F.one_hot(locations, num_classes=capacity).to(renormalized_probs.dtype)
            combine_weights = torch.einsum("se,sec->sec", renormalized_probs, locations_one_hot)
            dispatch_mask = combine_weights.bool()

        else:
            # 不丢弃 token 的情况，容量等于最多的专家分配数
            capacity = int(torch.max(exp_counts).long().item())
            if self.ep_group is not None:
                # 分布式环境下的容量同步
                tensor = torch.tensor([capacity], device=input.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.ep_group)
                capacity = tensor.item()
            
            # 这里如果不丢 token，保持 flat_routing_probs 不变
            masked_probs = flat_routing_probs
            renormalized_probs = masked_probs
            # 直接根据分配顺序生成 dispatch 位置
            locations = torch.zeros_like(flat_expert_indices)
            capacity_counts = torch.zeros(self.num_experts, dtype=torch.long, device=input.device)
            for i in range(batch_tokens):
                if used_token is None or used_token.reshape(-1)[i]:
                    for j in range(hk):
                        expert_idx = flat_expert_indices[i, j].item()
                        locations[i, j] = capacity_counts[expert_idx]
                        capacity_counts[expert_idx] += 1
            
            locations_one_hot = F.one_hot(locations.long(), num_classes=capacity).to(renormalized_probs.dtype)
            combine_weights = torch.einsum("se,sec->sec", renormalized_probs, locations_one_hot)
            dispatch_mask = combine_weights.bool()
        
        if self.wall_clock_breakdown and self.timers is not None:
            self.timers("gate_timer").stop()
        
        return l_aux, combine_weights, dispatch_mask, exp_counts, flat_expert_indices


class DenseGate(nn.Module):
    def __init__(self,
                 model_dim: int,
                 num_experts: int,) -> None:
        super().__init__()
        self.wg = nn.Linear(model_dim, num_experts, bias=False)
    
    def compute_entropy(self, p):
        """
        计算给定概率分布 p 的熵。
        参数：
        p (torch.Tensor): 一维概率分布张量，所有元素之和应为1。
        返回：
        torch.Tensor: 分布 p 的熵。
        """
        # 避免 log(0) 的问题，将 0 限制为一个很小的正数
        p = torch.clamp(p, min=1e-12, max=1.0)

        return -torch.sum(p * torch.log(p))

    def mutual_information_loss(self, gates):
        """
        计算专家路由中的互信息损失，用于鼓励专家之间均衡使用。
        互信息损失定义为：
        Loss = H(平均路由概率) - 平均(每个样本的路由概率熵)
        参数：
        gates (torch.Tensor): 形状为 (batch_size, num_experts) 的张量，
        每一行为一个样本对各个专家的路由概率分布。
        返回：
        torch.Tensor: 互信息损失的值。
        """
        # 计算所有样本的平均路由概率（专家重要性分布）
        avg_gate = torch.mean(gates, dim=0)
        # 计算重要性分布的熵
        entropy_avg = self.compute_entropy(avg_gate)
        # 计算每个样本的路由概率熵（逐行计算），然后取均值
        # 这里对 gates 每个元素同样进行 clip 防止 log(0)
        entropy_per_sample = -torch.sum(gates * torch.log(torch.clamp(gates, min=1e-12, max=1.0)), dim=1)
        avg_entropy = torch.mean(entropy_per_sample)
        # 互信息损失
        loss = entropy_avg - avg_entropy

        return loss

    def forward(self, input: torch.Tensor,) -> Tuple[Tensor, Tensor, Tensor]:
        """
        前向传播函数，计算专家路由概率和互信息损失。
        参数：
        input (torch.Tensor): 输入张量，形状为 (batch_size*seq_len, model_dim)。
        返回：
        Tuple[Tensor, Tensor, Tensor]: 返回三个张量：
            - l_aux: 互信息损失值。
            - combine_weights: 专家路由概率，形状为 (batch_size*seq_len, num_experts)。
        """

        input_fp32 = input.float()
        logits = torch.nn.functional.linear(input_fp32, weight=self.wg.weight.float(), bias=None)
        combine_weights = gates = F.softmax(logits, dim=1)

        l_aux = self.mutual_information_loss(gates)

        return l_aux, combine_weights, None, None


class EmbeddedExpertsMoE(nn.Module):
    """
    基于嵌入式存储的大规模MoE实现
    使用产品键专家选择和嵌入表参数存储
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 1024,  # 默认值较小，可以设置更大的值
        expert_dim: int = 1,      # 单神经元专家的内部维度
        k: int = 16,              # 每个token选择的专家数
        gate_type: str = "token_gating",  # 专家选择门控类型
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        use_residual: bool = False,  # 是否添加残差连接
        num_heads: int = 8,          # 专家选择的头数
        use_query_bn: bool = True,   # 是否在查询中使用批量归一化
        act_fn: str = "silu",        # 激活函数类型
        dropout: float = 0.0,        # Dropout率
        init_scale: float = 1.0,     # 初始化缩放
        expert_parallel: bool = False, # 是否并行化专家
        use_expert_gate: bool = False, # 是否启用针对每个专家的 gate 模块
        forward_mode: str = "batched", # 前向传播模式
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.k = k
        self.gate_type = gate_type
        self.use_residual = use_residual
        self.expert_parallel = expert_parallel
        self.use_expert_gate = use_expert_gate
        self.forward_mode = forward_mode
        self.ep_group = ep_group
        
        # 专家选择门控网络
        if gate_type == "token_gating":
            self.gate = TopKGate(
                model_dim=hidden_size,
                num_experts=num_experts,
                k=k,
                capacity_factor=capacity_factor,
                eval_capacity_factor=eval_capacity_factor,
                min_capacity=min_capacity,
                drop_tokens=True,
                ep_group=ep_group,
            )
        elif gate_type == "similarity_gating":
            # 确保 num_experts 是完全平方数，否则调整到最近的完全平方数
            sqrt_experts = math.sqrt(num_experts)
            if sqrt_experts != int(sqrt_experts):
                new_sqrt = math.ceil(sqrt_experts)
                num_experts = new_sqrt ** 2
                logger.warning(f"调整专家数量为最接近的完全平方数: {num_experts}")
                self.num_experts = num_experts

            self.gate = SimilarityGate(
                model_dim=hidden_size,
                num_experts=num_experts,
                k=k,
                capacity_factor=capacity_factor,
                eval_capacity_factor=eval_capacity_factor,
                min_capacity=min_capacity,
                drop_tokens=True,
                ep_group=ep_group,
                num_heads=num_heads,
                use_query_bn=use_query_bn,
            )
        else:
            raise ValueError(f"不支持的门控类型: {gate_type}")
        
        # 嵌入式专家参数
        self.expert_down = nn.Embedding(num_experts, hidden_size * expert_dim)
        self.expert_up = nn.Embedding(num_experts, expert_dim * hidden_size)
        if self.use_expert_gate:
            self.expert_gate = nn.Embedding(num_experts, hidden_size * expert_dim)
        
        # 设置激活函数
        if act_fn == "relu":
            self.activation = F.relu
        elif act_fn == "gelu":
            self.activation = F.gelu
        elif act_fn in ["silu", "swish"]:
            self.activation = F.silu
        else:
            raise ValueError(f"不支持的激活函数: {act_fn}")
        
        self.dropout = nn.Dropout(dropout)
        
        if use_residual:
            self.coefficient = nn.Linear(hidden_size, 2)
        
        # 初始化参数
        with torch.no_grad():
            # 使用高斯初始化下投影权重
            std_down = math.sqrt(2.0 / (hidden_size + expert_dim)) * init_scale
            nn.init.normal_(self.expert_down.weight, mean=0.0, std=std_down)
            
            # 使用高斯初始化上投影权重
            std_up = math.sqrt(1.0 / hidden_size) * init_scale
            nn.init.normal_(self.expert_up.weight, mean=0.0, std=std_up)
    
    def forward(self, hidden_states: torch.Tensor, used_token: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MoE前向传播
        
        参数:
            hidden_states: 形状为 [batch_size, seq_len, hidden_size] 的输入
            used_token: 可选的掩码指示有效 token
            
        返回:
            outputs: 模型输出
            l_aux: 负载均衡损失
            exp_counts: 专家计数
        """
        original_shape = hidden_states.shape
        batch_size, seq_len, hidden_size = original_shape
        device = hidden_states.device
        input_dtype = hidden_states.dtype

        # 扁平化输入
        hidden_states = hidden_states.reshape(-1, hidden_size)  # [N, hidden_size], N = batch_size * seq_len
        orig_hidden = hidden_states.clone()
        batch_tokens = hidden_states.shape[0]

        # 根据不同的门控获取路由信息
        if self.gate_type == "token_gating":
            l_aux, combine_weights, dispatch_mask, exp_counts = self.gate(hidden_states, used_token) # [N, num_experts, capacity]
        elif self.gate_type == "similarity_gating":
            l_aux, combine_weights, dispatch_mask, exp_counts, expert_indices = self.gate(hidden_states, used_token) # [N, topk, capacity], [N, expert_indices]
        combine_weights = combine_weights.to(dtype=input_dtype)

        # 初始化输出
        outputs = torch.zeros((batch_tokens, hidden_size), device=device, dtype=hidden_states.dtype)
        
        # 找到 dispatch_mask 中非零的位置（即被激活的 token）
        active_positions = torch.nonzero(dispatch_mask)  # shape: [num_active, 3], 0th col: token index, 1st col: expert index, 2nd col: capacity index
        
        if active_positions.numel() > 0:
            # 提取各个维度：第一列为 token 索引，第三列为 capacity 索引
            token_indices = active_positions[:, 0]
            capacity_indices = active_positions[:, 2]
            
            if self.gate_type == "token_gating":
                expert_indices_from_mask = active_positions[:, 1]
            elif self.gate_type == "similarity_gating":
                candidate_indices = active_positions[:, 1]
                # 这里根据 gate 返回的 expert_indices，结合 candidate 索引提取真正的专家索引
                expert_indices_from_mask = expert_indices[token_indices, candidate_indices]

            if self.forward_mode == "batched":
                # 获取对应token的隐藏状态
                flat_hidden = hidden_states[token_indices]
                
                # 获取专家参数 - 使用从dispatch_mask获取的专家索引
                expert_down_w = self.expert_down(expert_indices_from_mask)  # [active_tokens, hidden*expert_dim]
                expert_up_w = self.expert_up(expert_indices_from_mask)  # [active_tokens, expert_dim*hidden]
                
                # 重塑为矩阵形式
                expert_down_w = expert_down_w.view(-1, hidden_size, self.expert_dim)
                expert_up_w = expert_up_w.view(-1, self.expert_dim, hidden_size)
                
                # 计算中间激活
                intermediate = torch.bmm(flat_hidden.unsqueeze(1), expert_down_w).squeeze(1)
                if self.use_expert_gate:
                    expert_gate_w = self.expert_gate(expert_indices_from_mask)
                    expert_gate_w = expert_gate_w.view(-1, hidden_size, self.expert_dim)
                    gate_value = torch.bmm(flat_hidden.unsqueeze(1), expert_gate_w).squeeze(1)
                    gate_value = self.activation(gate_value)
                    intermediate = intermediate * gate_value
                else:
                    intermediate = self.activation(intermediate)
                    intermediate = self.dropout(intermediate)
                
                # 计算输出
                expert_outputs = torch.bmm(intermediate.unsqueeze(1), expert_up_w).squeeze(1)
                
                # 根据 combine_weights 获得组合权重
                if self.gate_type == "token_gating":
                    flat_weights = combine_weights[token_indices, expert_indices_from_mask, capacity_indices].unsqueeze(1)
                elif self.gate_type == "similarity_gating":
                    flat_weights = combine_weights[token_indices, candidate_indices, capacity_indices].unsqueeze(1)
                
                # 加权叠加专家输出到对应 token 上
                token_indices_expanded = token_indices.unsqueeze(1).expand(-1, hidden_size)
                weighted_outputs = flat_weights * expert_outputs
                outputs.scatter_add_(0, token_indices_expanded, weighted_outputs)

            elif self.forward_mode == "grouped":            
                # 利用 unique 对激活 token 按照专家分组，减少重复的 embedding 查找和矩阵计算
                unique_experts, inverse_indices = torch.unique(expert_indices_from_mask, return_inverse=True)
                # 对每个唯一的 expert 分组计算
                for i, expert in enumerate(unique_experts):
                    exp_id = int(expert.item())
                    # 找出分组内对应的 token 在扁平化输入中的下标
                    group_mask = (inverse_indices == i)
                    group_token_indices = token_indices[group_mask]  # token的全局索引
                    group_capacity_indices = capacity_indices[group_mask]
                    
                    # 取出这部分 tokens 的 hidden 表示，形状 [group_size, hidden_size]
                    tokens_group = hidden_states[group_token_indices]
                    
                    # 查找当前 expert 的 down/up（及可选 gate）参数，并重塑为矩阵形式
                    expert_down_weight = self.expert_down.weight[exp_id].view(hidden_size, self.expert_dim)
                    expert_up_weight = self.expert_up.weight[exp_id].view(self.expert_dim, hidden_size)

                    # 下投影计算：tokens_group @ expert_down_weight，得到中间激活
                    intermediate = tokens_group.matmul(expert_down_weight)  # [G, expert_dim]
                    
                    if self.use_expert_gate:
                        expert_gate_weight = self.expert_gate.weight[exp_id].view(hidden_size, self.expert_dim)
                        gate_value = tokens_group.matmul(expert_gate_weight)  # [G, expert_dim]
                        gate_value = self.activation(gate_value)
                        intermediate = intermediate * gate_value
                    else:
                        intermediate = self.activation(intermediate)
                        intermediate = self.dropout(intermediate)
                    
                    # 上投影计算
                    expert_output = intermediate.matmul(expert_up_weight)  # [G, hidden_size]

                    # 根据不同的门控获取每个 token 特有的组合权重
                    if self.gate_type == "token_gating":
                        # combine_weights 的形状为 [B*T, num_experts, capacity]
                        group_combine_weights = combine_weights[group_token_indices, exp_id, group_capacity_indices].unsqueeze(1)
                    elif self.gate_type == "similarity_gating":
                        group_candidate_indices = candidate_indices[group_mask]
                        group_combine_weights = combine_weights[group_token_indices, group_candidate_indices, group_capacity_indices].unsqueeze(1)
                    
                    weighted_output = expert_output * group_combine_weights  # [G, hidden_size]
                    
                    # 将当前 expert 组的输出累加回输出 tensor
                    outputs[group_token_indices] += weighted_output
        
        # 对于未经过 MoE 调度的 token，则保留原始表示
        processed_mask = torch.zeros(batch_tokens, dtype=torch.bool, device=device)
        if active_positions.numel() > 0:
            processed_mask[token_indices] = True
        unprocessed_mask = ~processed_mask
        outputs[unprocessed_mask] = orig_hidden[unprocessed_mask]

        # 恢复原始形状
        outputs = outputs.reshape(original_shape)
        
        # 如果使用残差连接，则融合处理前后的信息
        if self.use_residual:
            coef = self.coefficient(hidden_states.reshape(original_shape))
            coef = F.softmax(coef, dim=-1)
            outputs = outputs * coef[..., 0:1] + hidden_states.reshape(original_shape) * coef[..., 1:]
        
        return outputs, l_aux, exp_counts


class EmbeddedMoELayer(nn.Module):
    """
    嵌入式MoE层，可作为标准Transformer层的替代品
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 1024,
        expert_dim: int = 1,
        k: int = 16,
        gate_type: str = "token_gating",
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        use_residual: bool = True,
        num_heads: int = 8,
        use_query_bn: bool = True,
        act_fn: str = "silu",
        dropout: float = 0.0,
        init_scale: float = 1.0,
        use_expert_gate: bool = False,
        forward_mode: str = "batched",
        norm_type: str = "layernorm",
        use_norm: bool = False,
        use_pre_norm: bool = True,
    ):
        super().__init__()
        
        self.use_norm = use_norm
        self.use_pre_norm = use_pre_norm
        
        # 归一化层
        if use_norm:
            if norm_type == "layernorm":
                self.norm = nn.LayerNorm(hidden_size)
            elif norm_type == "rmsnorm":
                # 简单的RMSNorm实现
                from functools import partial
                def rms_norm(x, eps=1e-5):
                    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
                self.norm = partial(rms_norm)
            else:
                raise ValueError(f"不支持的归一化类型: {norm_type}")
        
        # MoE层
        self.moe = EmbeddedExpertsMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            expert_dim=expert_dim,
            k=k,
            gate_type=gate_type,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            min_capacity=min_capacity,
            use_residual=use_residual,
            num_heads=num_heads,
            use_query_bn=use_query_bn,
            act_fn=act_fn,
            dropout=dropout,
            init_scale=init_scale,
            use_expert_gate=use_expert_gate,
            forward_mode=forward_mode,
        )
    
    def forward(self, hidden_states: torch.Tensor, used_token: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """层前向传播
        
        参数:
            hidden_states: 形状为[batch_size, seq_len, hidden_size]的输入
            used_token: 可选的掩码指示有效token
            
        返回:
            output: 层输出
            l_aux: 负载均衡损失
            exp_counts: 专家计数
        """
        if self.use_norm:
            if self.use_pre_norm:
                # Pre-LN风格：先应用归一化，再处理
                hidden_states = self.norm(hidden_states)
                output, l_aux, exp_counts = self.moe(hidden_states, used_token)
            else:
                # Post-LN风格：先处理，再应用归一化
                hidden_states, l_aux, exp_counts = self.moe(hidden_states, used_token)
                output = self.norm(hidden_states)
        else:
            output, l_aux, exp_counts = self.moe(hidden_states, used_token)

        return output, l_aux, exp_counts


class DenseMaskMoE(nn.Module):
    """
    Dense Mask Mixture-of-Experts (MoE) 层基于 dense 运算 + mask 实现。

    工作流程：
      1. 使用 TopKGate 对输入进行专家路由，返回 (l_aux, combine_weights, dispatch_mask, exp_counts)。
      2. 对 combine_weights（形状 [N, num_experts, capacity]）在 capacity 维度求和，
         得到每个 token 对各专家的组合权重（形状 [N, num_experts]）。
      3. 将所有专家的下投影权重融合成一个大矩阵，对输入执行一次 dense 运算，然后 reshape 得到各专家中间输出。
      4. 应用激活和 dropout（或可选的专家门控）。
      5. 利用 einsum 对各专家输出分别进行上投影，再用 combine_dense 加权求和，
         生成最终 token 表示。
    """
    def __init__(
        self,
        hidden_size: int,
        expert_dim: int,
        num_experts: int,
        k: int,
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        dropout: float = 0.0,
        activation: str = "silu",
        use_expert_gate: bool = False,
        gate_type: str = "token_gating",
        kd_align: bool = False,
        expert_cluster_mask_list: List[Tensor] = None
    ):
        """
        Args:
            hidden_size (int): 输入及输出的维度。
            expert_dim (int): 每个专家的内部中间激活维度。
            num_experts (int): 专家数量。
            k (int): 每个 token 选择的 top-k 专家数。
            capacity_factor (float): TopKGate 的 capacity_factor 参数。
            eval_capacity_factor (float): TopKGate 的 eval_capacity_factor 参数。
            min_capacity (int): TopKGate 的最小 capacity 参数。
            dropout (float): dropout 概率。
            activation (str): 激活函数类型，可选 "silu", "relu", "gelu"。
            use_expert_gate (bool): 是否使用额外的专家门控参数。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.k = k
        self.use_expert_gate = use_expert_gate
        self.gate_type = gate_type
        self.kd_align = kd_align
        self.expert_cluster_mask_list = expert_cluster_mask_list

        if not self.kd_align:
            if gate_type == 'token_gating':
                if expert_cluster_mask_list is None:
                    self.gate = TopKGate(
                        model_dim=hidden_size,
                        num_experts=num_experts,
                        k=k,
                        capacity_factor=capacity_factor,
                        eval_capacity_factor=eval_capacity_factor,
                        min_capacity=min_capacity,
                        drop_tokens=True,
                    )
                else:
                    self.gate = TopKGateDynamic(
                        model_dim=hidden_size,
                        num_experts=num_experts,
                        k=k,
                        capacity_factor=capacity_factor,
                        eval_capacity_factor=eval_capacity_factor,
                        min_capacity=min_capacity,
                        drop_tokens=True,
                        expert_cluster_mask_list=expert_cluster_mask_list
                    )
            elif gate_type == 'dense_gating':
                self.gate = DenseGate(
                    model_dim=hidden_size,
                    num_experts=num_experts,
                )
            else:
                raise ValueError(f"Unsupported gate type: {gate_type}")

        self.dropout = nn.Dropout(dropout)
        
        # 选择激活函数
        if activation == "silu":
            self.activation = F.silu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 用 nn.Embedding 存储下投影参数，每个专家的参数 shape: [hidden_size * expert_dim]
        self.expert_down = nn.Embedding(num_experts, hidden_size * expert_dim)
        # 初始化并乘上合适的缩放因子
        nn.init.normal_(self.expert_down.weight, std=math.sqrt(2.0 / (hidden_size + expert_dim)))
        
        # 上投影参数，每个专家的参数 shape: [expert_dim * hidden_size]
        self.expert_up = nn.Embedding(num_experts, expert_dim * hidden_size)
        nn.init.normal_(self.expert_up.weight, std=math.sqrt(1.0 / hidden_size))
        
        # 可选：专家内部门控参数，存储形状: [hidden_size * expert_dim]
        if self.use_expert_gate:
            self.expert_gate = nn.Embedding(num_experts, hidden_size * expert_dim)
            nn.init.normal_(self.expert_gate.weight, std=math.sqrt(2.0 / (hidden_size + expert_dim)))


    def count_expert_cooccurrence(self, combine_dense: torch.Tensor) -> torch.Tensor:
        """
        统计专家的共现情况。
        
        Args:
            combine_dense (torch.Tensor): 形状为 [N, num_experts] 的张量，
                每个 token 对各专家的激活权重（或者标志），非零表示激活。
        
        Returns:
            co_occurrence_upper (torch.Tensor): 上三角的共现矩阵，形状为 [num_experts, num_experts]，
                即 co_occurrence_upper[i, j] 表示专家 i 和专家 j 同时被激活的 token 数量（仅计算 i<j 部分）。
        """
        # 将 combine_dense 二值化（假定激活权重大于 0 表示激活）
        # 如果 combine_dense 原本就是 0/1，则这个步骤可以省略
        binary_mask = (combine_dense > 0).float()  # shape: [N, num_experts]
        
        # 计算共现矩阵：shape [num_experts, num_experts]
        # 其中 entry (i, j) 表示有多少 token 同时激活了专家 i 和专家 j
        co_occurrence = torch.matmul(binary_mask.transpose(0, 1), binary_mask)
        
        # 我们不关心自己与自己的共现，因此将对角线置 0
        # co_occurrence.fill_diagonal_(0)
        
        # 如果只需要上三角部分（因为 C 是对称的）
        co_occurrence_upper = torch.triu(co_occurrence, diagonal=1)
        
        return co_occurrence_upper
    

    def compute_jaccard_scores(self, co_occurrence: torch.Tensor, expert_activation_counts: torch.Tensor, min_count: int = 1):
        """
        根据 Jaccard 相似度计算专家对得分，并返回排名列表。
        
        Args:
            co_occurrence (torch.Tensor): 专家共现矩阵，形状为 [num_experts, num_experts]（对角线为0）。
            expert_activation_counts (torch.Tensor): 每个专家的激活总数，形状为 [num_experts]。
            min_count (int): 只返回共现次数大于等于此阈值的专家对。
            
        Returns:
            ranked_pairs (List[Tuple[int, int, float]]): 每个元素为 (expert_i, expert_j, jaccard_score)
                值越大表示两个专家关系越紧密。
        """
        num_experts = co_occurrence.size(0)
        pairs = []
        for i in range(num_experts):
            for j in range(i+1, num_experts):
                co_count = co_occurrence[i, j].item()
                if co_count < min_count:
                    continue
                # 计算两个专家的并集大小：C_i + C_j - C_ij
                union = expert_activation_counts[i].item() + expert_activation_counts[j].item() - co_count
                if union > 0:
                    jaccard_score = co_count / union
                    pairs.append((i, j, jaccard_score))
        # 按照得分降序排序
        ranked_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        return ranked_pairs


    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, hidden_size] 或 [N, hidden_size]。

        Returns:
            output (torch.Tensor): 与输入形状一致的输出张量。
            l_aux (torch.Tensor): 从 gate 返回的辅助负载均衡损失。
            exp_counts (torch.Tensor): 各专家激活 token 的统计信息。
        """
        orig_shape = hidden_states.shape
        # 将输入展平成 [N, hidden_size]
        x = hidden_states.view(-1, self.hidden_size)
        N = x.size(0)

        # 1. 专家路由：得到 l_aux, combine_weights, dispatch_mask, exp_counts
        if self.kd_align:
            combine_dense = generate_routing_tensor(N, self.num_experts, self.k, device=x.device, dtype=x.dtype)
            l_aux = None
            exp_counts = None
        else:
            if self.gate_type == "token_gating":
                l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(x)
                # 将 combine_weights 在 capacity 维度上求和，形状：[N, num_experts]
                combine_weights = combine_weights.to(x.dtype)
                combine_dense = combine_weights.sum(dim=-1)
            elif self.gate_type == "dense_gating":
                l_aux, combine_dense, _, self.exp_counts = self.gate(x)
                combine_dense = combine_dense.to(x.dtype)

        # 2. Fused 下投影：
        # 从 embedding 中恢复 expert_down_weight，形状为 [num_experts, hidden_size, expert_dim]
        expert_down_weight = self.expert_down.weight.view(self.num_experts, self.hidden_size, self.expert_dim)
        # 将 expert_down_weight 转置并融合为 [hidden_size, num_experts * expert_dim]
        fused_down = expert_down_weight.transpose(0, 1).reshape(self.hidden_size, self.num_experts * self.expert_dim)
        # 对输入 x 做一次 dense 运算，获得中间输出：[N, num_experts * expert_dim]
        intermediate_flat = x.matmul(fused_down)
        # 变换形状到 [N, num_experts, expert_dim]
        intermediate_all = intermediate_flat.view(N, self.num_experts, self.expert_dim)

        # 3. 可选：专家内门控
        if self.use_expert_gate:
            # 从 embedding 中恢复 expert_gate_weight，形状为 [num_experts, hidden_size, expert_dim]
            expert_gate_weight = self.expert_gate.weight.view(self.num_experts, self.hidden_size, self.expert_dim)
            fused_gate = expert_gate_weight.transpose(0, 1).reshape(self.hidden_size, self.num_experts * self.expert_dim)
            gate_flat = x.matmul(fused_gate)
            gate_vals = gate_flat.view(N, self.num_experts, self.expert_dim)
            intermediate_all = intermediate_all * self.activation(gate_vals)
        else:
            intermediate_all = self.dropout(self.activation(intermediate_all))

        # 4. 上投影及加权求和：
        # 从 embedding 中恢复 expert_up_weight，形状为 [num_experts, expert_dim, hidden_size]
        expert_up_weight = self.expert_up.weight.view(self.num_experts, self.expert_dim, self.hidden_size)
        # 计算公式：
        #   output[n, h] = sum_{e, b} combine_dense[n, e] * intermediate_all[n, e, b] * expert_up_weight[e, b, h]
        output = torch.einsum('ne, neb, ebh -> nh', combine_dense, intermediate_all, expert_up_weight)

        # self.co_occurrence = self.count_expert_cooccurrence(combine_dense)
        self.combine_dense = combine_dense
        # ranked_pairs = self.compute_jaccard_scores(co_occurrence, exp_counts)
        # print(f"Co-occurrence matrix: {co_occurrence}")
        # print(f"Ranked pairs: {ranked_pairs[:3]}")

        # 恢复原始形状（例如 [batch_size, seq_len, hidden_size]）
        output = output.view(*orig_shape)
        return output, l_aux, self.exp_counts


class MoEQwen2VLConfig(Qwen2VLConfig):
    model_type = "moe_qwen2_vl"

    def __init__(self,
                 moe_enable=True,
                 moe_mode='sparse',
                 moe_layers_idx=None,
                 ep_size=1,
                 top_k_experts=2,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 router_aux_loss_coef=0.01,
                 **kwargs):
        self.moe = dict(
            moe_enable=moe_enable,
            moe_mode=moe_mode,
            moe_layers_idx=moe_layers_idx,
            ep_size=ep_size,
            top_k_experts=top_k_experts,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            min_capacity=min_capacity,
            use_residual=use_residual,
            router_aux_loss_coef=router_aux_loss_coef,
            train_modules=[]
        )
        self.lora = {}
        self.mone = {}

        super(MoEQwen2VLConfig, self).__init__(**kwargs)


@dataclass
class MoEBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    moe_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None


def MoEQwen2DecoderLayer_forward(self):
    def forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings for rotary attention.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Handle MoE layer
        mlp_output = self.mlp(hidden_states)

        moe_losses = []
        if isinstance(mlp_output, tuple) and len(mlp_output) >= 2:
            moe_losses.append(mlp_output[1])
            hidden_states = mlp_output[0]
        else:
            hidden_states = mlp_output
            
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (moe_losses,)

        return outputs

    return forward


def MoEQwen2VLModel_forward(self):
    def forward(
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            output_moe_loss: Optional[bool] = True,
    ) -> Union[Tuple, MoEBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_moe_loss = [] if output_moe_loss else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_moe_loss:
                all_moe_loss.extend(layer_outputs[-1])

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_moe_loss] if
                v is not None)
        return MoEBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            moe_loss_list=all_moe_loss,
        )

    return forward


class MoEQwen2VLModel(Qwen2VLModel):
    config_class = MoEQwen2VLConfig
    
    def __init__(self, config):
        super().__init__(config)
        self._attn_implementation = config._attn_implementation
        
    # We need to inherit the _update_causal_mask method to ensure proper functionality
    # This is referenced in the forward method
    _update_causal_mask = Qwen2VLModel._update_causal_mask


class MoEQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    config_class = MoEQwen2VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MoEQwen2VLModel(config)
        
        # Initialize or reuse components
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict




        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    print(f"input_ids: {input_ids}")
                    print(f"image_grid_thw: {image_grid_thw}")
                    print(f"image_embeds: {image_embeds}") 
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # Calculate position IDs and rope deltas if needed
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # Calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # Use the previously calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        moe_loss, moe_losses = None, []
        if hasattr(outputs, "moe_loss_list") and outputs.moe_loss_list and len(outputs.moe_loss_list) > 0:
            moe_loss_list = outputs.moe_loss_list
            for moe_loss_item in moe_loss_list:
                if moe_loss_item is not None:
                    moe_losses.append(moe_loss_item)
            if moe_losses:
                moe_loss = self.router_aux_loss_coef * sum(moe_losses)
                if (labels is not None):
                    # print(f"Loss: {loss}, MoE Loss: {sum(moe_losses)}, Total: {loss + moe_loss}")
                    if self.config.moe.get('kd_align', False):
                        loss = moe_loss + 0 * loss
                    else:
                        loss += moe_loss   

        if not return_dict:
            output = (logits,) + outputs[1:]
            output = (moe_loss,) + output if moe_loss is not None else output
            return (loss,) + output if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            moe_loss=moe_loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
            moe_loss_list=outputs.moe_loss_list if hasattr(outputs, "moe_loss_list") else None,
        )

    def initialize_moe_modules(self, model_args):
        if getattr(model_args, 'lora_enable', False):
            self.config.lora['lora_enable'] = model_args.lora_enable
            self.config.lora['only_lora_ffn'] = model_args.only_lora_ffn
            self.config.lora['lora_r'] = model_args.lora_r
            self.config.lora['lora_alpha'] = model_args.lora_alpha
            self.config.lora['lora_dropout'] = model_args.lora_dropout
            self.config.lora['lora_bias'] = model_args.lora_bias
            self.config.lora['target_modules'] = model_args.train_modules
        
        if getattr(model_args, 'mone_enable', False):
            self.config.mone['mone_enable'] = model_args.mone_enable
            self.config.mone['mone_expert_type'] = model_args.mone_expert_type
            self.config.mone['mone_gate_type'] = model_args.mone_gate_type
            self.config.mone['mone_r'] = model_args.mone_r
            self.config.mone['mone_dropout'] = model_args.mone_dropout
            self.config.mone['mone_num_heads'] = model_args.mone_num_heads
            self.config.mone['mone_use_query_bn'] = model_args.mone_use_query_bn
            self.config.mone['mone_act_fn'] = model_args.mone_act_fn
            self.config.mone['mone_use_expert_gate'] = model_args.mone_use_expert_gate
            self.config.mone['mone_load_original'] = model_args.mone_load_original
            self.config.mone['mone_forward_mode'] = model_args.mone_forward_mode

        self.config.moe['moe_enable'] = model_args.moe_enable
        self.config.moe['train_modules'] = model_args.train_modules
        self.config.moe['moe_mode'] = model_args.moe_mode
        self.config.moe['moe_layers_idx'] = model_args.moe_layers_idx
        self.config.moe['ep_size'] = model_args.ep_size
        self.config.moe['top_k_experts'] = model_args.top_k_experts
        self.config.moe['capacity_factor'] = model_args.capacity_factor
        self.config.moe['eval_capacity_factor'] = model_args.eval_capacity_factor
        self.config.moe['min_capacity'] = model_args.min_capacity
        self.config.moe['use_residual'] = model_args.use_residual
        self.config.moe['router_aux_loss_coef'] = self.router_aux_loss_coef = model_args.router_aux_loss_coef
        self.config.moe['use_shared_experts'] = model_args.use_shared_experts
        self.config.moe['shared_expert_type'] = model_args.shared_expert_type
        self.config.moe['use_combined_gate'] = model_args.use_combined_gate
        self.config.moe['combined_gate_type'] = model_args.combined_gate_type
        self.config.moe['combined_gate_drop'] = model_args.combined_gate_drop
        self.config.moe['kd_align'] = model_args.kd_align
        
        # Freeze all parameters except those specified in train_modules
        if self.config.moe['train_modules'] is not None and len(self.config.moe['train_modules']) > 0:
            for n, p in self.named_parameters():
                if any(name in n for name in self.config.moe['train_modules']):
                    continue
                else:
                    p.requires_grad = False

        num_layers = self.config.num_hidden_layers

        # Determine which layers will be converted to MoE
        moe_layers_idx = model_args.moe_layers_idx
        if model_args.moe_layers_idx is not None:
            model_args.moe_mode = 'custom'
            assert len(model_args.moe_layers_idx) <= num_layers
            assert max(model_args.moe_layers_idx) < num_layers
            assert min(model_args.moe_layers_idx) >= 0
        else:
            if model_args.moe_mode == "first_half":
                moe_layers_idx = list(range(0, num_layers // 2))
            elif model_args.moe_mode == "second_half":
                moe_layers_idx = list(range(num_layers // 2, num_layers))
            elif model_args.moe_mode == "sparse":
                moe_layers_idx = list(range(num_layers))[::2]
            elif model_args.moe_mode == "dense":
                moe_layers_idx = list(range(num_layers))
            else:
                raise NotImplementedError(
                    f'Only support ["first_half", "second_half", "sparse", "dense"], but found {model_args.moe_mode}')

        self.config.moe['moe_layers_idx'] = moe_layers_idx
        self.config.moe['num_experts'] = model_args.num_experts
        
        # Handle single num_experts value
        if len(model_args.num_experts) == 1:
            self.config.moe['num_experts'] = model_args.num_experts * len(moe_layers_idx)
        assert len(self.config.moe['num_experts']) == len(moe_layers_idx)

        # Convert specified layers to MoE
        for num_experts, layer_num in zip(self.config.moe['num_experts'], moe_layers_idx):
            original_mlp = self.model.layers[layer_num].mlp
            if not getattr(model_args, 'mone_enable', False):
                pretrained_state_dict = original_mlp.state_dict()
                moe_layer = MoE(
                    self.config.hidden_size,
                    expert=self.model.layers[layer_num].mlp,
                    num_experts=num_experts,
                    ep_size=model_args.ep_size,
                    k=model_args.top_k_experts,
                    capacity_factor=model_args.capacity_factor,
                    eval_capacity_factor=model_args.eval_capacity_factor,
                    min_capacity=model_args.min_capacity,
                    use_residual=model_args.use_residual,
                )
                # Verify weights are properly copied
                for e in moe_layer.deepspeed_moe.experts.deepspeed_experts:
                    loaded_state_dict = e.state_dict()
                    assert all([torch.allclose(pretrained_state_dict[k], v) for k, v in loaded_state_dict.items()])
                    assert all([torch.allclose(loaded_state_dict[k], v) for k, v in pretrained_state_dict.items()])
            else:
                hidden_size = self.config.hidden_size
                mone_r = expert_dim = self.config.mone['mone_r']
                mone_dropout = self.config.mone['mone_dropout']
                
                if model_args.mone_expert_type == 'small_expert':
                    # 创建小专家实例
                    small_expert = SmallExpert(self.config.hidden_size, mone_r, mone_dropout)
                    # 创建MoE层
                    moe_layer = MoE(
                        self.config.hidden_size,
                        expert=small_expert,  # 使用小专家
                        num_experts=num_experts,
                        ep_size=model_args.ep_size,
                        k=model_args.top_k_experts,
                        capacity_factor=model_args.capacity_factor,
                        eval_capacity_factor=model_args.eval_capacity_factor,
                        min_capacity=model_args.min_capacity,
                        use_residual=model_args.use_residual,
                    )
                elif model_args.mone_expert_type == 'embedding_expert':
                    moe_layer = EmbeddedMoELayer(
                        hidden_size=self.config.hidden_size,
                        num_experts=num_experts,  # 可设置为1M专家
                        expert_dim=mone_r,           # 单神经元专家
                        k=model_args.top_k_experts,                   # 每个token选择的专家数
                        gate_type=model_args.mone_gate_type,
                        capacity_factor=model_args.capacity_factor,
                        eval_capacity_factor=model_args.eval_capacity_factor,
                        min_capacity=model_args.min_capacity,
                        use_residual=model_args.use_residual,
                        num_heads=self.config.mone['mone_num_heads'],            # 多头专家选择, 8
                        use_query_bn=self.config.mone['mone_use_query_bn'],      # 使用批量归一化提高稳定性, True
                        act_fn=self.config.mone['mone_act_fn'],          # 可选: "relu", "gelu", "silu", silu
                        dropout=self.config.mone['mone_dropout'],            # 专家dropout率
                        use_expert_gate=self.config.mone['mone_use_expert_gate'],  # 是否使用专家门控
                        forward_mode=model_args.mone_forward_mode,  # 前向模式
                    )
                elif model_args.mone_expert_type == 'dense_mask_expert':
                    moe_layer = DenseMaskMoE(
                        self.config.hidden_size,
                        expert_dim=expert_dim,
                        num_experts=num_experts,
                        k=model_args.top_k_experts,
                        capacity_factor=model_args.capacity_factor,
                        eval_capacity_factor=model_args.eval_capacity_factor,
                        min_capacity=model_args.min_capacity,
                        use_expert_gate=model_args.mone_use_expert_gate,
                        gate_type=model_args.mone_gate_type,
                        kd_align=model_args.kd_align,
                    )
                else:
                    raise NotImplementedError(f"Unsupported expert type: {model_args.mone_expert_type}")

            shared_expert = None
            if getattr(model_args, 'mone_enable', False) and model_args.mone_load_original:
                with torch.no_grad():
                    intermediate_size = original_mlp.gate_proj.weight.data.shape[0]
                    total_expert_dim = num_experts * expert_dim

                    # 计算复制因子，向上取整
                    if total_expert_dim % intermediate_size != 0:
                        print(f"\033[93mWarning: num_experts * expert_dim ({total_expert_dim}) "
                            f"is not perfectly divisible by original_mlp intermediate_size ({intermediate_size}). "
                            f"Weights will be repeated and truncated/padded.\033[0m"
                        )
                    m = math.ceil(total_expert_dim / intermediate_size)
                    if m > 1 and m * intermediate_size != total_expert_dim:
                            print(f"\033[93mWarning: not exact repetition of {m} for {intermediate_size} -> {total_expert_dim}\033[0m")

                    if (self.config.mone['mone_use_expert_gate']) or (model_args.mone_expert_type == 'small_expert'):
                        gate_proj_weight_source = original_mlp.gate_proj.weight.data  # shape: (intermediate_size, hidden_size)
                        if m > 1:
                            gate_proj_weight_source = gate_proj_weight_source.repeat(m, 1)  # shape: (m * intermediate_size, hidden_size)
                        # 截取正好 total_expert_dim 行，多余部分丢弃
                        gate_proj_weight = gate_proj_weight_source[:total_expert_dim, :]
                        if model_args.mone_expert_type in ['embedding_expert', 'dense_mask_expert']:
                            # 接下来的 reshape 操作要求行数正好为 num_experts * expert_dim
                            gate_proj_weight_reshaped = gate_proj_weight.view(num_experts, expert_dim, hidden_size).transpose(1, 2)
                            gate_proj_weight_flat = gate_proj_weight_reshaped.reshape(num_experts, expert_dim * hidden_size)
                    
                    up_proj_weight_source = original_mlp.up_proj.weight.data
                    if m > 1:
                        up_proj_weight_source = up_proj_weight_source.repeat(m, 1)
                    up_proj_weight = up_proj_weight_source[:total_expert_dim, :]
                    if model_args.mone_expert_type in ['embedding_expert', 'dense_mask_expert']:
                        up_proj_weight_reshaped = up_proj_weight.view(num_experts, expert_dim, hidden_size).transpose(1, 2)
                        up_proj_weight_flat = up_proj_weight_reshaped.reshape(num_experts, expert_dim * hidden_size)
                    
                    down_proj_weight_source = original_mlp.down_proj.weight.data.t()
                    if m > 1:
                        down_proj_weight_source = down_proj_weight_source.repeat(m, 1)
                    down_proj_weight = down_proj_weight_source[:total_expert_dim, :]
                    if model_args.mone_expert_type in ['embedding_expert', 'dense_mask_expert']:
                        down_proj_weight_reshaped = down_proj_weight.view(num_experts, expert_dim, hidden_size)
                        down_proj_weight_flat = down_proj_weight_reshaped.reshape(num_experts, expert_dim * hidden_size)
                    
                    if model_args.mone_expert_type == 'embedding_expert':
                        if self.config.mone['mone_use_expert_gate']:
                            moe_layer.moe.expert_gate.weight.data.copy_(gate_proj_weight_flat)
                        moe_layer.moe.expert_down.weight.data.copy_(up_proj_weight_flat)
                        moe_layer.moe.expert_up.weight.data.copy_(down_proj_weight_flat)
                    elif model_args.mone_expert_type == 'dense_mask_expert':
                        if self.config.mone['mone_use_expert_gate']:
                            moe_layer.expert_gate.weight.data.copy_(gate_proj_weight_flat)
                        moe_layer.expert_down.weight.data.copy_(up_proj_weight_flat)
                        moe_layer.expert_up.weight.data.copy_(down_proj_weight_flat)
                        if model_args.use_shared_experts and model_args.shared_expert_type == 'small':
                            shared_expert = SmallExpert(self.config.hidden_size, mone_r, mone_dropout)
                            assert gate_proj_weight_source.shape[0] == total_expert_dim + expert_dim, \
                                f"Shape mismatch for shared expert gate: expected {total_expert_dim + expert_dim}, got {gate_proj_weight.shape[0]}"
                            gate_proj_weight_s = gate_proj_weight_source[total_expert_dim:total_expert_dim+expert_dim, :]
                            up_proj_weight_s = up_proj_weight_source[total_expert_dim:total_expert_dim+expert_dim, :]
                            down_proj_weight_s = down_proj_weight_source[total_expert_dim:total_expert_dim+expert_dim, :]
                            shared_expert.gate_proj.weight.data.copy_(gate_proj_weight_s)
                            shared_expert.up_proj.weight.data.copy_(up_proj_weight_s)
                            shared_expert.down_proj.weight.data.copy_(down_proj_weight_s.t())
                        
                    if model_args.mone_expert_type == 'small_expert':
                        expert_module_list = moe_layer.deepspeed_moe.experts.deepspeed_experts
                        down_proj_weight = down_proj_weight.t()
                        for i in range(num_experts):
                            expert = expert_module_list[i]
                            
                            # Calculate slicing indices for gate_proj/up_proj/down_pro
                            start_idx = i * expert_dim
                            end_idx = start_idx + expert_dim

                            # Initialize gate_proj for expert i
                            # Target shape: (r_expert_intermediate, hidden_size)
                            slice_gate = gate_proj_weight[start_idx:end_idx, :]
                            assert expert.gate_proj.weight.data.shape == slice_gate.shape, \
                                f"Shape mismatch for expert {i} gate_proj: expected {expert.gate_proj.weight.data.shape}, got {slice_gate.shape}"
                            expert.gate_proj.weight.data.copy_(slice_gate)

                            # Initialize up_proj for expert i
                            # Target shape: (r_expert_intermediate, hidden_size)
                            slice_up = up_proj_weight[start_idx:end_idx, :]
                            assert expert.up_proj.weight.data.shape == slice_up.shape, \
                                f"Shape mismatch for expert {i} up_proj: expected {expert.up_proj.weight.data.shape}, got {slice_up.shape}"
                            expert.up_proj.weight.data.copy_(slice_up)

                            # Initialize down_proj for expert i
                            # Target shape: (hidden_size, r_expert_intermediate)
                            slice_down = down_proj_weight[:, start_idx:end_idx]
                            assert expert.down_proj.weight.data.shape == slice_down.shape, \
                                f"Shape mismatch for expert {i} down_proj: expected {expert.down_proj.weight.data.shape}, got {slice_down.shape}"
                            expert.down_proj.weight.data.copy_(slice_down)
                    print("\033[92m" + f"Successfully initialized weights for {num_experts} experts in layer {layer_num}" + "\033[0m")
            if model_args.use_shared_experts:
                if shared_expert is not None:
                    shared_expert = shared_expert
                elif model_args.shared_expert_type == 'small':
                    shared_expert = SmallExpert(self.config.hidden_size, mone_r, mone_dropout)
                else:
                    shared_expert = self.model.layers[layer_num].mlp

                moe_layer = CombinedLayer(
                    shared_expert, 
                    moe_layer, 
                    self.config.moe['use_combined_gate'], 
                    self.config.moe['combined_gate_type'],
                    self.config.moe['combined_gate_drop'],
                    self.config.hidden_size,
                    kd_align=self.config.moe['kd_align'],
                )
            self.model.layers[layer_num].mlp = moe_layer

        # # 冻结普通MLP层，只训练MoE层
        # for name, param in self.model.named_parameters():
        #     # 如果是普通MLP层参数（不是MoE层）
        #     if 'mlp' in name and 'deepspeed_moe' not in name:
        #         param.requires_grad = False
        #     # 可选：冻结其他非MLP层参数
        #     elif 'mlp' not in name:
        #         param.requires_grad = False  # 如果只想训练MoE部分
        
        
        print(f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])

        # Replace forward methods to handle MoE outputs
        for m in self.model.layers:
            m.forward = MoEQwen2DecoderLayer_forward(m)
        rank0_print(f'replace Qwen2DecoderLayer.forward to MoEQwen2DecoderLayer.forward')
        
        self.model.forward = MoEQwen2VLModel_forward(self.model)
        rank0_print(f'replace Qwen2VLModel.forward to MoEQwen2VLModel.forward')

    get_rope_index = Qwen2VLForConditionalGeneration.get_rope_index
    prepare_inputs_for_generation = Qwen2VLForConditionalGeneration.prepare_inputs_for_generation
    _get_image_nums_and_video_nums = Qwen2VLForConditionalGeneration._get_image_nums_and_video_nums
    _expand_inputs_for_generation = Qwen2VLForConditionalGeneration._expand_inputs_for_generation


class EvalMoEQwen2VLForConditionalGeneration(MoEQwen2VLForConditionalGeneration):
    config_class = MoEQwen2VLConfig

    def __init__(self, config):
        super(EvalMoEQwen2VLForConditionalGeneration, self).__init__(config)
        if getattr(self.config, 'lora', False) and self.config.lora.get('lora_enable', False):
            from peft import LoraConfig, get_peft_model
            pre_lora_config = self.config.lora
            lora_config = LoraConfig(
                r=pre_lora_config['lora_r'],
                lora_alpha=pre_lora_config['lora_alpha'],
                target_modules=pre_lora_config['target_modules'],
                lora_dropout=pre_lora_config['lora_dropout'],
                bias=pre_lora_config['lora_bias'],
                task_type="CAUSAL_LM",
            )
            print("Adding LoRA adapters...")
            get_peft_model(self, lora_config)
        
        if getattr(self.config, 'mone', False):
            mone_expert_type = self.config.mone.get('mone_expert_type', 'embedding_expert')

        self.router_aux_loss_coef = self.config.moe['router_aux_loss_coef']
        num_layers = self.config.num_hidden_layers
        moe_layers_idx = self.config.moe['moe_layers_idx']
        expert_cluster_mask_dict = self.config.mone.get('expert_cluster_mask_dict', None)

        # Reinitialize MoE layers for evaluation
        for num_experts, layer_num in zip(self.config.moe.get('num_experts'), moe_layers_idx):
            original_mlp = self.model.layers[layer_num].mlp
            if getattr(self.config, 'mone', False):
                mone_r = self.config.mone.get('mone_r', 2)
                mone_dropout = self.config.mone.get('mone_dropout', 0.0)
                
                if mone_expert_type == 'small_expert':
                    # 创建小专家实例
                    small_expert = SmallExpert(self.config.hidden_size, mone_r, mone_dropout)
                    # 创建MoE层
                    moe_layer = MoE(
                        self.config.hidden_size,
                        expert=small_expert,  # 使用小专家
                        num_experts=num_experts,
                        ep_size=self.config.moe.get('ep_size'),
                        k=self.config.moe.get('top_k_experts'),
                        capacity_factor=self.config.moe.get('capacity_factor'),
                        eval_capacity_factor=self.config.moe.get('eval_capacity_factor'),
                        min_capacity=self.config.moe.get('min_capacity'),
                        use_residual=self.config.moe.get('use_residual'),
                    )
                elif mone_expert_type == 'embedding_expert':
                    moe_layer = EmbeddedMoELayer(
                        hidden_size=self.config.hidden_size,
                        num_experts=num_experts,  # 可设置为1M专家
                        expert_dim=mone_r,        # 单神经元专家
                        k=self.config.moe.get('top_k_experts'),
                        gate_type=self.config.mone.get('mone_gate_type', 'token_gating'),
                        capacity_factor=self.config.moe.get('capacity_factor'),
                        eval_capacity_factor=self.config.moe.get('eval_capacity_factor'),
                        min_capacity=self.config.moe.get('min_capacity'),
                        use_residual=self.config.moe.get('use_residual'),
                        num_heads=self.config.mone.get('mone_num_heads', 1),       # 多头专家选择
                        use_query_bn=self.config.mone.get('mone_use_query_bn', False),   # 使用批量归一化提高稳定性
                        act_fn=self.config.mone.get('mone_act_fn', 'silu'),               # 可选: "relu", "gelu", "silu"
                        dropout=self.config.mone.get('mone_dropout', 0.0),             # 专家 dropout 率
                        use_expert_gate=self.config.mone.get('mone_use_expert_gate', False),    # 是否使用专家门控
                    )
                elif mone_expert_type == 'dense_mask_expert':
                    if expert_cluster_mask_dict is not None: 
                        _expert_cluster_mask_list = expert_cluster_mask_dict[str(layer_num)]
                        expert_cluster_mask_list = [] 
                        for ele in _expert_cluster_mask_list:
                            expert_cluster_mask = torch.zeros(1, num_experts)
                            expert_cluster_mask[0, ele] = 1
                            expert_cluster_mask_list.append(expert_cluster_mask.to(torch.bool))
                    else:
                        expert_cluster_mask_list = None
                    moe_layer = DenseMaskMoE(
                        self.config.hidden_size,
                        expert_dim=mone_r,
                        num_experts=num_experts,
                        k=self.config.moe.get('top_k_experts'),
                        capacity_factor=self.config.moe.get('capacity_factor'),
                        eval_capacity_factor=self.config.moe.get('eval_capacity_factor'),
                        min_capacity=self.config.moe.get('min_capacity'),
                        use_expert_gate=self.config.mone.get('mone_use_expert_gate', False),
                        gate_type=self.config.mone.get('mone_gate_type', 'token_gating'),
                        expert_cluster_mask_list=expert_cluster_mask_list
                    )
                else:
                    raise NotImplementedError(f"Unsupported expert type: {mone_expert_type}")
            else:
                moe_layer = MoE(
                    self.config.hidden_size,
                    expert=self.model.layers[layer_num].mlp,
                    num_experts=num_experts,
                    ep_size=self.config.moe.get('ep_size'),
                    k=self.config.moe.get('top_k_experts'),
                    capacity_factor=self.config.moe.get('capacity_factor'),
                    eval_capacity_factor=self.config.moe.get('eval_capacity_factor'),
                    min_capacity=self.config.moe.get('min_capacity'),
                    use_residual=self.config.moe.get('use_residual'),
                )
            if self.config.moe.get('use_shared_experts', False):
                print(f"self.config.moe.get('structure', 'new'),: {self.config.moe.get('structure', 'new')}")
                if  self.config.moe.get('shared_expert_type', 'original') == 'small':
                    shared_expert = SmallExpert(self.config.hidden_size, mone_r, mone_dropout)
                else:
                    shared_expert = original_mlp
                moe_layer = CombinedLayer(
                    shared_expert, 
                    moe_layer, 
                    self.config.moe.get('use_combined_gate', False),
                    self.config.moe.get('combined_gate_type', False),
                    self.config.moe.get('combined_gate_drop', False),
                    self.config.hidden_size,
                    structure=self.config.moe.get('structure', 'new'),
                )
            self.model.layers[layer_num].mlp = moe_layer

        print(f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])

        # Replace forward methods for evaluation
        for m in self.model.layers:
            m.forward = MoEQwen2DecoderLayer_forward(m)
        print(f'replace Qwen2DecoderLayer.forward to MoEQwen2DecoderLayer.forward')
        
        self.model.forward = MoEQwen2VLModel_forward(self.model)
        print(f'replace Qwen2VLModel.forward to MoEQwen2VLModel.forward')
        print(self.model)
    
    get_rope_index = Qwen2VLForConditionalGeneration.get_rope_index
    prepare_inputs_for_generation = Qwen2VLForConditionalGeneration.prepare_inputs_for_generation
    _get_image_nums_and_video_nums = Qwen2VLForConditionalGeneration._get_image_nums_and_video_nums
    _expand_inputs_for_generation = Qwen2VLForConditionalGeneration._expand_inputs_for_generation


class LazyInitMoEQwen2VLForConditionalGeneration(MoEQwen2VLForConditionalGeneration):
    """
    一个经过重构的模型类，支持 MoE 模块的延迟初始化。

    工作流程:
    1. `model = LazyInitMoEQwen2VLForConditionalGeneration(config)`: 创建一个标准的、密集的模型骨架。
    2. `model.initialize_moe_modules()`: 在需要时调用此方法，将密集层转换为 MoE 层。
    3. `model.load_state_dict(...)`: 加载包含 MoE 权重的 state_dict。
    """
    config_class = MoEQwen2VLConfig

    def __init__(self, config):
        """
        构造函数现在非常干净。
        它只负责调用父类的构造函数来构建一个标准的、没有 MoE 的模型。
        """
        super().__init__(config)
        self._moe_initialized = False
        print("LazyInitMoEModel: 基础模型骨架已创建。准备好进行 MoE 初始化。")

    def initialize_moe_modules(self):
        """
        执行 MoE 模块的延迟初始化。
        这段代码是从旧的 __init__ 方法中移动过来的。
        """
        if self._moe_initialized:
            print("警告：MoE 模块已经被初始化，跳过。")
            return

        print("--- 开始执行 MoE 模块的延迟初始化 ---")

        # 1. LoRA 适配器逻辑 (如果需要)
        if getattr(self.config, 'lora', False) and self.config.lora.get('lora_enable', False):
            from peft import LoraConfig, get_peft_model
            pre_lora_config = self.config.lora
            lora_config = LoraConfig(
                r=pre_lora_config['lora_r'],
                lora_alpha=pre_lora_config['lora_alpha'],
                target_modules=pre_lora_config['target_modules'],
                lora_dropout=pre_lora_config['lora_dropout'],
                bias=pre_lora_config['lora_bias'],
                task_type="CAUSAL_LM",
            )
            print("正在添加 LoRA 适配器...")
            get_peft_model(self, lora_config)

        # 2. MoE/MoNE 配置准备
        self.router_aux_loss_coef = self.config.moe['router_aux_loss_coef']
        num_layers = self.config.num_hidden_layers
        moe_layers_idx = self.config.moe['moe_layers_idx']
        mone_expert_type = self.config.mone.get('mone_expert_type', 'embedding_expert') if getattr(self.config, 'mone', False) else None
        expert_cluster_mask_dict = self.config.mone.get('expert_cluster_mask_dict', None)

        # 3. 遍历并替换为 MoE 层
        print(f"准备在 {len(moe_layers_idx)} 个层上初始化 MoE 模块...")
        for num_experts, layer_num in zip(self.config.moe.get('num_experts'), moe_layers_idx):
            # 获取原始的 MLP 层，它的权重将被 MoE 专家复用
            original_mlp = self.model.layers[layer_num].mlp
            moe_layer = None

            if getattr(self.config, 'mone', False):
                mone_r = self.config.mone.get('mone_r', 2)
                mone_dropout = self.config.mone.get('mone_dropout', 0.0)
                
                if mone_expert_type == 'small_expert':
                    # 创建小专家实例
                    small_expert = SmallExpert(self.config.hidden_size, mone_r, mone_dropout)
                    # 创建MoE层
                    moe_layer = MoE(
                        self.config.hidden_size,
                        expert=small_expert,  # 使用小专家
                        num_experts=num_experts,
                        ep_size=self.config.moe.get('ep_size'),
                        k=self.config.moe.get('top_k_experts'),
                        capacity_factor=self.config.moe.get('capacity_factor'),
                        eval_capacity_factor=self.config.moe.get('eval_capacity_factor'),
                        min_capacity=self.config.moe.get('min_capacity'),
                        use_residual=self.config.moe.get('use_residual'),
                    )
                elif mone_expert_type == 'embedding_expert':
                    moe_layer = EmbeddedMoELayer(
                        hidden_size=self.config.hidden_size,
                        num_experts=num_experts,  # 可设置为1M专家
                        expert_dim=mone_r,        # 单神经元专家
                        k=self.config.moe.get('top_k_experts'),
                        gate_type=self.config.mone.get('mone_gate_type', 'token_gating'),
                        capacity_factor=self.config.moe.get('capacity_factor'),
                        eval_capacity_factor=self.config.moe.get('eval_capacity_factor'),
                        min_capacity=self.config.moe.get('min_capacity'),
                        use_residual=self.config.moe.get('use_residual'),
                        num_heads=self.config.mone.get('mone_num_heads', 1),       # 多头专家选择
                        use_query_bn=self.config.mone.get('mone_use_query_bn', False),   # 使用批量归一化提高稳定性
                        act_fn=self.config.mone.get('mone_act_fn', 'silu'),               # 可选: "relu", "gelu", "silu"
                        dropout=self.config.mone.get('mone_dropout', 0.0),             # 专家 dropout 率
                        use_expert_gate=self.config.mone.get('mone_use_expert_gate', False),    # 是否使用专家门控
                    )
                elif mone_expert_type == 'dense_mask_expert':
                    if expert_cluster_mask_dict is not None: 
                        _expert_cluster_mask_list = expert_cluster_mask_dict[str(layer_num)]
                        expert_cluster_mask_list = [] 
                        for ele in _expert_cluster_mask_list:
                            expert_cluster_mask = torch.zeros(1, num_experts)
                            expert_cluster_mask[0, ele] = 1
                            expert_cluster_mask_list.append(expert_cluster_mask.to(torch.bool))
                    else:
                        expert_cluster_mask_list = None
                    moe_layer = DenseMaskMoE(
                        self.config.hidden_size,
                        expert_dim=mone_r,
                        num_experts=num_experts,
                        k=self.config.moe.get('top_k_experts'),
                        capacity_factor=self.config.moe.get('capacity_factor'),
                        eval_capacity_factor=self.config.moe.get('eval_capacity_factor'),
                        min_capacity=self.config.moe.get('min_capacity'),
                        use_expert_gate=self.config.mone.get('mone_use_expert_gate', False),
                        gate_type=self.config.mone.get('mone_gate_type', 'token_gating'),
                        expert_cluster_mask_list=expert_cluster_mask_list
                    )
                else:
                    raise NotImplementedError(f"Unsupported expert type: {mone_expert_type}")
            else:
                # 标准 MoE 逻辑
                print(f"  - 正在替换第 {layer_num} 层的 MLP 为标准 MoE 层...")
                moe_layer = MoE(
                    hidden_size=self.config.hidden_size,
                    expert=original_mlp,  # <-- 关键：复用原始 MLP 作为专家模板
                    num_experts=num_experts,
                    ep_size=self.config.moe.get('ep_size'),
                    k=self.config.moe.get('top_k_experts'),
                    capacity_factor=self.config.moe.get('capacity_factor'),
                    eval_capacity_factor=self.config.moe.get('eval_capacity_factor'),
                    min_capacity=self.config.moe.get('min_capacity'),
                    use_residual=self.config.moe.get('use_residual'),
                )
            
            if self.config.moe.get('use_shared_experts', False):
                print(f"self.config.moe.get('structure', 'new'),: {self.config.moe.get('structure', 'new')}")
                if  self.config.moe.get('shared_expert_type', 'original') == 'small':
                    shared_expert = SmallExpert(self.config.hidden_size, mone_r, mone_dropout)
                else:
                    shared_expert = original_mlp
                moe_layer = CombinedLayer(
                    shared_expert, 
                    moe_layer, 
                    self.config.moe.get('use_combined_gate', False),
                    self.config.moe.get('combined_gate_type', False),
                    self.config.moe.get('combined_gate_drop', False),
                    self.config.hidden_size,
                    structure=self.config.moe.get('structure', 'new'),
                )
            self.model.layers[layer_num].mlp = moe_layer

        print(f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where\n",
            *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                zip(self.config.moe['num_experts'], moe_layers_idx)])

        # 4. 替换 forward 方法以支持 MoE 的门控损失计算
        print("正在替换 forward 方法以适配 MoE...")
        for m in self.model.layers:
            m.forward = MoEQwen2DecoderLayer_forward(m)
        
        self.model.forward = MoEQwen2VLModel_forward(self.model)
        
        self._moe_initialized = True
        print("--- MoE 模块延迟初始化完成 ---")
        print("最终模型结构:")
        print(self.model)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        重写的 from_pretrained 方法，集成了我们的延迟初始化逻辑。
        """
        print(f"--- 使用自定义的 from_pretrained 方法从 '{pretrained_model_name_or_path}' 加载模型 ---")

        # 步骤 1: 加载配置
        # trust_remote_code=True 对于 Qwen 这类模型是必需的
        config = MoEQwen2VLConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, **kwargs)
        config.vision_config.torch_dtype = config.torch_dtype

        print("配置加载完成:", config)
        print(f'vision_config: {config.vision_config}')

        # 检查是否处于 ZeRO-3 环境
        is_zero3 = is_deepspeed_zero3_enabled()
        if is_zero3:
            raise ValueError("MoE 模型不支持 ZeRO-3 优化。")

        # 传统的初始化和加载流程 (适用于 ZeRO-2, 无DeepSpeed等)
        else:
            print("--- 步骤 1/3: 创建完整模型骨架 ---")
            model = cls(config)
            print("--- 步骤 2/3: 执行 MoE 初始化 ---")
            model.initialize_moe_modules()

            print("--- 打印模型参数 (检查点) ---")
            for n, p in model.named_parameters():
                print(f"Parameter: {n}, Shape: {p.shape}, Requires Grad: {p.requires_grad}")

            print("--- 步骤 3/3: 加载完整权重 ---")
            try:
                model_file = cached_file(pretrained_model_name_or_path, "model.safetensors", _raise_exceptions_for_missing_entries=True)
                from safetensors.torch import load_file
                state_dict = load_file(model_file, device="cpu")
            except (OSError, IOError):
                model_file = cached_file(pretrained_model_name_or_path, "pytorch_model.bin", _raise_exceptions_for_missing_entries=True)
                state_dict = torch.load(model_file, map_location="cpu")
            
            load_result = model.load_state_dict(state_dict, strict=True)
            print("权重加载结果:", load_result)
            
        return model

    # --- 保留其他必要的方法 ---
    get_rope_index = MoEQwen2VLForConditionalGeneration.get_rope_index
    prepare_inputs_for_generation = MoEQwen2VLForConditionalGeneration.prepare_inputs_for_generation
    _get_image_nums_and_video_nums = MoEQwen2VLForConditionalGeneration._get_image_nums_and_video_nums
    _expand_inputs_for_generation = MoEQwen2VLForConditionalGeneration._expand_inputs_for_generation


# Register the new model with AutoConfig and AutoModel systems
AutoConfig.register("moe_qwen2_vl", MoEQwen2VLConfig)
AutoModelForCausalLM.register(MoEQwen2VLConfig, MoEQwen2VLForConditionalGeneration)
AutoModelForCausalLM.register(MoEQwen2VLConfig, EvalMoEQwen2VLForConditionalGeneration)

