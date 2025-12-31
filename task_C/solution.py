import torch


def pytorch_permute_index_map(tokens, indices):
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    num_out_tokens = flatten_indices.size(0)
    permuted_tokens = tokens.index_select(0, sorted_indices[:num_out_tokens] // topk)
    return permuted_tokens, sorted_indices


def torch_basic(x: torch.Tensor, top_experts: torch.Tensor, tokens_per_expert: torch.Tensor, topk: int, num_experts: int):
    block_size = 128
    device = x.device
    num_tokens, hidden_dim = x.shape

    expert_ids_flat = top_experts.view(-1)

    padded_tokens_per_expert = (
        ((tokens_per_expert + block_size - 1) // block_size) * block_size
    ).to(torch.int32)
    padded_offsets = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        padded_tokens_per_expert.cumsum(dim=0)
    ])
    expert_ids_cpu = expert_ids_flat.cpu().tolist()
    padded_offsets_cpu = padded_offsets.cpu().tolist()

    max_padded_tokens = int(num_tokens * topk + (block_size - 1) * num_experts + 1)
    max_padded_tokens = padded_offsets_cpu[-1]
    padded_tokens = torch.zeros(
        (max_padded_tokens, hidden_dim),
        dtype=x.dtype,
        device=device,
    )

    assignment_groups = [[] for _ in range(num_experts)]
    num_assignments = topk * num_tokens
    for i in range(num_assignments):
        expert_id = expert_ids_cpu[i]
        assignment_groups[expert_id].append(i)

    for e in range(num_experts):
        local_idx = 0
        offset = padded_offsets[e]

        for local_idx, i in enumerate(assignment_groups[e]):
            original_token_idx = i // topk
            token_data = x[original_token_idx]
            target_row = offset + local_idx
            padded_tokens[target_row, :] = token_data

    return padded_tokens, padded_tokens_per_expert


def submission(
    x: torch.Tensor,  # (num_tokens, hidden_size) - входной тензор токенов, каждый размерности hidden_size
    top_experts: torch.Tensor,  # (num_tokens, topk) - для каждого токена указано topk экспертов, которые он активирует
    tokens_per_expert: torch.Tensor,  # (num_experts,) - тензор размерности числа экспертов, i-ый элемент - сколько токенов приходит в i-ого эксперта
    topk: int,  # сколько экспертов активируются на каждый токен, например, 8
    num_experts: int,  # сколько всего экспертов в MoE, например, 128
) -> tuple[
    torch.Tensor,  # (max_padded, hidden_size) - padded_tokens, результат пермьюта с паддингами
    torch.Tensor,  # (num_experts,) - padded_tokens_per_expert, сколько токенов приходят в каждого эксперта вместе с паддингами
]:
    block_size = 128
    device = x.device
    num_tokens, hidden_dim = x.shape
    N = num_tokens * topk
    
    tokens_per_expert = tokens_per_expert.to(device=device, dtype=torch.int32)
    padded_tokens_per_expert = (tokens_per_expert + block_size - 1) // block_size * block_size

    padded_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), padded_tokens_per_expert.cumsum(dim=0)]
    )  # [num_experts + 1]
    total_padded = int(padded_offsets[-1].item())

    unpadded_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), tokens_per_expert.cumsum(dim=0)]
    )  # [num_experts + 1]

    expert_ids = top_experts.reshape(-1).to(device=device)  # [N]
    sorted_expert_ids, sort_idx = torch.sort(expert_ids, stable=True)

    pos_in_sorted = torch.arange(N, device=device, dtype=torch.int32)        # [N]
    local_index_sorted = pos_in_sorted - unpadded_offsets[sorted_expert_ids] # [N]

    row_sorted = (padded_offsets[sorted_expert_ids] + local_index_sorted).to(torch.int64)  # [N]
    token_idx = (torch.arange(N, device=device, dtype=torch.int64) // topk)  # [N]
    source_sorted = x[token_idx[sort_idx]]  # [N, hidden_dim]

    padded_tokens = torch.zeros((total_padded, hidden_dim), dtype=x.dtype, device=device)
    padded_tokens.index_copy_(0, row_sorted, source_sorted)

    return padded_tokens, padded_tokens_per_expert
