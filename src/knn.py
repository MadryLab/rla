# # Extract kNN info
# def extract_knn(X, mask, eps, top_k):
#     # Convolutional network on NCHW
#     mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
#     dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
#     D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

#     # Identify k nearest neighbors (including self)
#     D_max, _ = torch.max(D, -1, keepdim=True)
#     D_adjust = D + (1. - mask_2D) * D_max
#     D_neighbors, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)
#     return mask_2D, D_neighbors, E_idx

# def get_reduce_masks(coords, k_neighbors):
#     t1 = time.time()
#     X = torch.from_numpy(coords).unsqueeze(0)[:, :, 1] # 1 x 1 x 4 x 3
#     mask = torch.ones(X.shape[1]).unsqueeze(0)
#     _, _, E_idx = extract_knn(X, mask, eps=1E-6, top_k=k_neighbors)
#     t2 = time.time()
#     E_idx = E_idx[0]
#     inds_reduce, inds_expand, inds_transpose, inds_duplicate, inds_singles, mask_combs = extract_idxs(E_idx, mask)
#     inds_convert = (inds_reduce, inds_expand, inds_transpose, inds_duplicate, inds_singles)
#     mask_expanded = mask.unsqueeze(-1).expand(-1, -1, k_neighbors)
#     inds_reduce = inds_reduce.to(torch.int64)
#     mask_reduced = per_node_to_all_comb(mask_expanded, inds_reduce.unsqueeze(0))
#     mask_reduced = torch.multiply(mask_reduced, mask_combs)
#     t3 = time.time()
#     print("reduce", t3-t2, t2-t1)
#     return inds_convert, mask_reduced