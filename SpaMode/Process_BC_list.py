import os

# 设置 OpenMP 线程数
os.environ["OMP_NUM_THREADS"] = "1"
# 设置 MKL 线程数
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import torch
import pandas as pd
import scanpy as sc
import numpy as np
import SME
import copy
import scipy
import anndata as ad
import argparse

from typing import Optional
from scanpy.pp import combat
from SME.preprocess import clr_normalize_each_cell, pca, lsi
from SME.preprocess import construct_neighbor_graph
from SME.utils import clustering, peak_sets_alignment, gene_sets_alignment
from cal_matrics import eval
from SME.Svae import Train_Smoe
# from SME.Svae_Multi import Train_Smoe_Multi
# from SME.Svae_0116 import Train_Smoe_Multi
# from harmony import harmonize
from harmonypy import run_harmony
from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

# Environment configuration. SpatialGlue pacakge can be implemented with either CPU or GPU. GPU acceleration is highly recommend for imporoved efficiency.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def read_list_from_file(path):
    list = []
    # 打开文件进行读取，使用 'r' 模式
    with open(path, 'r') as f:
        # 遍历文件中的每一行，将其转换为整数并添加到列表中
        for line in f:
            # 去掉行末的换行符，然后将字符串转换为整数
            num = int(line.strip())
            list.append(num)

    return list

def combine_BC(adata_list):
    X_list = []
    spatial_list = []
    batch_list = []
    feat_list = []

    for i, adata in enumerate(adata_list):
        # print(adata.obsm['feat'].shape)
        # X_list.append(adata.obsm['feat'])
        print(adata.X.shape)
        if issparse(adata.X):
            X_list.append(adata.X.toarray())
        else:
            X_list.append(adata.X)
        spatial_list.append(adata.obsm['spatial'])
        # feat_list.append(adata.obsm['feat'])
        batch_list.append(np.full(adata.n_obs, i))

    X_combined = np.concatenate(X_list, axis=0)
    spatial_combined = np.concatenate(spatial_list, axis=0)

    # feat_combined = np.concatenate(feat_list, axis=0)

    batch_combined = np.concatenate(batch_list, axis=0)

    adata_combine = ad.AnnData(X=X_combined)
    adata_combine.obsm['spatial'] = spatial_combined
    # adata_combine.obsm['feat'] = feat_combined
    adata_combine.obs['Batch'] = batch_combined

    return adata_combine


def TFIDF(count_mat, type_=2):
    # Perform TF-IDF (count_mat: peak*cell)
    def tfidf1(count_mat):
        if not scipy.sparse.issparse(count_mat):
            count_mat = scipy.sparse.coo_matrix(count_mat)

        nfreqs = count_mat.multiply(1.0 / count_mat.sum(axis=0))
        tfidf_mat = nfreqs.multiply(np.log(1 + 1.0 * count_mat.shape[1] / count_mat.sum(axis=1)).reshape(-1, 1)).tocoo()

        return tfidf_mat.toarray()

    # Perform Signac TF-IDF (count_mat: peak*cell) [default selected]
    def tfidf2(count_mat):
        if not scipy.sparse.issparse(count_mat):
            count_mat = scipy.sparse.coo_matrix(count_mat)

        tf_mat = count_mat.multiply(1.0 / count_mat.sum(axis=0))
        signac_mat = (1e4 * tf_mat).multiply(1.0 * count_mat.shape[1] / count_mat.sum(axis=1).reshape(-1, 1))
        signac_mat = signac_mat.log1p()

        return signac_mat.toarray()

    # Perform TF-IDF (count_mat: ?)
    from sklearn.feature_extraction.text import TfidfTransformer
    def tfidf3(count_mat):
        model = TfidfTransformer(smooth_idf=False, norm="l2")
        model = model.fit(np.transpose(count_mat))
        model.idf_ -= 1
        tf_idf = np.transpose(model.transform(np.transpose(count_mat)))

        return tf_idf.toarray()

    if type_ == 1:
        return tfidf1(count_mat)
    elif type_ == 2:
        return tfidf2(count_mat)
    else:
        return tfidf3(count_mat)


def get_spatial_adj(adj):
    # 确定矩阵大小
    N = max(adj['x'].max(), adj['y'].max()) + 1  # 矩阵大小为最大索引 + 1

    # 使用scipy.sparse.coo_matrix创建稀疏矩阵
    sparse_matrix = scipy.sparse.coo_matrix((adj['value'], (adj['x'], adj['y'])), shape=(N, N))

    # 转换为密集矩阵
    dense_matrix = sparse_matrix.toarray()

    # if not isinstance(adj, csr_matrix):
    #     return csr_matrix(adj)
    # return adj

    return csr_matrix(dense_matrix)


def data_preprocessing(adata_omics1, adata_omics2):
    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    # RNA
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1, max_value=100)

    adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
    adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)

    # ATAC
    # adata_omics2 = adata_omics2[
    #     adata_omics1.obs_names].copy()  # .obsm['X_lsi'] represents the dimension reduced feature
    #
    # adata_omics2.X = TFIDF(adata_omics2.X.T).T.copy()

    # Protein
    adata_omics2 = clr_normalize_each_cell(adata_omics2)

    sc.pp.scale(adata_omics2)

    adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=50)

    print(adata_omics1)
    print(adata_omics2)

    return adata_omics1, adata_omics2


def process_spatial_adj(adata_list, omcis='adata_omics1', key='adj_spatial'):
    """
        构建跨多个数据集的空间邻接 DataFrame，处理任意数量的 adata 对象。

        Args:
            adata_list (list): 包含多个 AnnData 对象的列表。
            omcis (str): 包含空间邻接信息的 adata 的 obsm 或 uns 键。
            key (str): 空间邻接矩阵在 adata[omcis].uns 中的键。

        Returns:
            pd.DataFrame: 包含空间邻接信息的 DataFrame，列名为 'x', 'y', 'value'。
        """
    num_slices = len(adata_list)
    adj_matrices = [get_spatial_adj(adata[omcis].uns[key]) for adata in adata_list]
    slice_shapes = [adj.shape[0] for adj in adj_matrices]

    row_indices = []
    col_indices = []
    values = []
    row_offset = 0
    col_offset = 0

    for i in range(num_slices):
        adj_i = adj_matrices[i]
        n_rows_i = slice_shapes[i]

        row_coords, col_coords = adj_i.nonzero()
        values_i = adj_i[row_coords, col_coords].A1  # 获取非零值

        row_indices.extend(row_coords + row_offset)
        col_indices.extend(col_coords + col_offset)
        values.extend(values_i)

        row_offset += n_rows_i
        col_offset += n_rows_i  # 假设是对角块，行列偏移量相同

    spatial_adj_pd = pd.DataFrame({
        'x': row_indices,
        'y': col_indices,
        'value': values
    })

    return spatial_adj_pd


def process_feat_adj(adata_list, omcis='adata_omics1'):
    """
    构建跨多个数据集的特征邻接矩阵，处理任意数量的 adata 对象。

    Args:
        adata_list (list): 包含多个 AnnData 对象的列表。
        omcis (str): 包含特征邻接信息的 adata 的 obsm 键。
        key (str): 特征邻接矩阵在 adata[omcis].obsm 中的键。

    Returns:
        np.ndarray: 拼接后的特征邻接矩阵。
    """
    num_slices = len(adata_list)
    adj_matrices = [adata[omcis].obsm[key] for adata in adata_list]
    slice_shapes = [adj.shape[0] for adj in adj_matrices]

    blocks = []
    for i in range(num_slices):
        row = []
        for j in range(num_slices):
            if i == j:
                row.append(adj_matrices[i])
            else:
                zeros_shape = (slice_shapes[i], slice_shapes[j])
                row.append(np.zeros(zeros_shape))
        blocks.append(row)

    feat_adj = np.block(blocks)
    return feat_adj


def construct_graph_by_feature(adata_omics1, key='feat', k=20, mode="distance", metric="correlation",
                               include_self=False):
    """Constructing feature neighbor graph according to expresss profiles"""
    feature_graph_omics1 = kneighbors_graph(adata_omics1.obsm[key], k, mode=mode, metric=metric,
                                            include_self=include_self)

    return feature_graph_omics1.toarray()


def build_adjacency_matrix_inter_BC(A, B, K=20):
    """
    A: N x 64 的归一化特征矩阵 (numpy)
    B: M x 64 的归一化特征矩阵 (numpy)
    K: 每个 A 中的样本与 B 中最相似的 K 个样本构成邻接
    """
    # 计算 A 和 B 之间的余弦相似度矩阵
    sim_matrix = np.dot(A, B.T)  # N x M

    # 对于 A 中的每个样本，找到 B 中最相似的 K 个样本
    topk_indices = np.argsort(sim_matrix, axis=1)[:, -K:]  # N x K

    # 构建邻接矩阵
    N, M = A.shape[0], B.shape[0]
    adjacency_matrix = np.zeros((N, M), dtype=np.float32)

    # 将最相似的 K 个样本的位置设为 1
    for i in range(N):
        adjacency_matrix[i, topk_indices[i]] = 1

    return adjacency_matrix


def get_feat_adj(adata_list, k=10, feat_key='feat'):
    """
    构建跨多个数据集的特征邻接矩阵，处理任意数量的 adata 对象。

    Args:
        adata_list (list): 包含多个 AnnData 对象的列表。
        k (int): 构建跨数据集邻接矩阵时，每个样本考虑的近邻数量。
        feat_key (str): AnnData 对象中存储特征的 .obsm 的键。

    Returns:
        csr_matrix: 拼接后的稀疏特征邻接矩阵。
    """
    num_slices = len(adata_list)
    feat_list = [adata.obsm[feat_key] for adata in adata_list]
    adj_blocks = []

    for i in range(num_slices):
        row_blocks = []
        for j in range(num_slices):
            if i == j:
                # 对角线块：基于特征构建每个数据集内部的邻接图
                adj_omics = construct_graph_by_feature(adata_list[i], key=feat_key, k=k)  # k 在内部也可以使用
                row_blocks.append(adj_omics)
            elif j > i:
                # 上三角块：构建数据集 i 到数据集 j 的跨数据集邻接矩阵
                adj_bc = build_adjacency_matrix_inter_BC(feat_list[i], feat_list[j], K=k)
                row_blocks.append(adj_bc)
            else:
                # 下三角块：使用上三角块的转置
                # 需要注意的是，这里的转置是为了连接性，build_adjacency_matrix_inter_BC 是有方向的
                # 从 j 到 i 的连接，所以使用 build_adjacency_matrix_inter_BC(feat_list[j], feat_list[i], K=k).T 也可以
                # 这里为了保持一致性，直接引用之前计算的转置
                row_blocks.append(adj_blocks[j][i].T)
        adj_blocks.append(row_blocks)

    feat_adj = np.block(adj_blocks)
    return csr_matrix(feat_adj)

def save_combined_GT():
    GT_paths = {
        '1': '/home/hxl/Spa_Multi-omics/Gen_sim/multiome_ZINB_NB_sim_BC/Slice-1/GT.txt',
        '2': '/home/hxl/Spa_Multi-omics/Gen_sim/multiome_ZINB_NB_sim_BC/Slice-2/GT.txt',
        '3': '/home/hxl/Spa_Multi-omics/Gen_sim/multiome_ZINB_NB_sim_BC/Slice-3/GT.txt',
    }
    # 读取AnnData文件
    E11_0_S1_label = read_list_from_file(GT_paths['1'])
    E13_5_S1_label = read_list_from_file(GT_paths['2'])
    E15_5_S1_label = read_list_from_file(GT_paths['3'])

    combined_GT = E11_0_S1_label + E13_5_S1_label + E15_5_S1_label

    print(len(combined_GT))
    print(set(combined_GT))

    # 指定输出文件路径和文件名
    output_file = '/home/hxl/Spa_Multi-omics/0222/Data/Sim_0611/S1_S2_S3_Combined_BE_GT.txt'

    # 打开文件进行写入，使用 'w' 模式
    with open(output_file, 'w') as f:
        # 遍历列表中的每个整数元素，逐行写入文件
        for num in combined_GT:
            f.write(f"{num}\n")
    print('save results to ', output_file)


def main():
    # Fix random seed
    from SME.preprocess import fix_seed
    random_seed = 2024
    fix_seed(random_seed)

    base_path = '/home/hxl/Spa_Multi-omics/Gen_sim/multiome_ZINB_NB_sim_BC/'

    file_paths = {
        '1': [base_path + '/Slice-1/RNA.h5ad',
              base_path + '/Slice-1/Protein.h5ad'],
        '2': [base_path + '/Slice-2/RNA.h5ad',
              base_path + '/Slice-2/Protein.h5ad'],
        '3': [base_path + '/Slice-3/RNA.h5ad',
              base_path + '/Slice-3/Protein.h5ad'],
    }

    # Load all AnnData objects
    adata_omics1_list = []
    adata_omics2_list = []
    for paths in file_paths.values():
        adata_omics1, adata_omics2 = [sc.read_h5ad(fp) for fp in paths]
        adata_omics1_list.append(adata_omics1)
        adata_omics2_list.append(adata_omics2)

    # Align gene and peak sets
    # adata_omics1_list = gene_sets_alignment(adata_omics1_list)
    # adata_omics2_list = peak_sets_alignment(adata_omics2_list)

    adata_omics1_norm_list = []
    adata_omics2_norm_list = []

    for adata1, adata2 in zip(adata_omics1_list, adata_omics2_list):
        # Create copies for source data
        adata_omics1_src = copy.deepcopy(adata1)
        adata_omics2_src = copy.deepcopy(adata2)

        # Create copies for normalized data and preprocess
        adata_omics1_norm, adata_omics2_norm = data_preprocessing(copy.deepcopy(adata1), copy.deepcopy(adata2))
        adata_omics1_norm_list.append(adata_omics1_norm)
        adata_omics2_norm_list.append(adata_omics2_norm)

    # Combine across batches
    src_adata_omics1 = combine_BC(adata_omics1_list)
    src_adata_omics2 = combine_BC(adata_omics2_list)
    src_adata_omics1_BC, src_adata_omics2_BC = data_preprocessing(src_adata_omics1, src_adata_omics2)

    # Construct neighbor graphs for each slice
    adata_list = [
        construct_neighbor_graph(norm_adata1, norm_adata2, datatype='Sim_L')
        for norm_adata1, norm_adata2 in zip(adata_omics1_norm_list, adata_omics2_norm_list)
    ]

    spatial_adj_pd = process_spatial_adj(adata_list)

    # Construct neighbor graph for combined data
    adata = construct_neighbor_graph(src_adata_omics1_BC, src_adata_omics2_BC, datatype='Sim_L')

    adata['adata_omics1'].uns['adj_spatial'] = spatial_adj_pd
    adata['adata_omics2'].uns['adj_spatial'] = spatial_adj_pd

    # Initialize norm_feat in the combined adata
    total_obs = src_adata_omics1_BC.n_obs
    adata['adata_omics1'].obsm['norm_feat'] = np.zeros((total_obs, adata_omics1_norm_list[0].obsm['feat'].shape[1]))
    adata['adata_omics2'].obsm['norm_feat'] = np.zeros((total_obs, adata_omics2_norm_list[0].obsm['feat'].shape[1]))

    # Fill norm_feat with normalized features from each slice
    start_idx = 0
    for norm_adata1 in adata_omics1_norm_list:
        end_idx = start_idx + norm_adata1.n_obs
        adata['adata_omics1'].obsm['norm_feat'][start_idx:end_idx] = norm_adata1.obsm['feat']
        start_idx = end_idx

    start_idx = 0
    for norm_adata2 in adata_omics2_norm_list:
        end_idx = start_idx + norm_adata2.n_obs
        adata['adata_omics2'].obsm['norm_feat'][start_idx:end_idx] = norm_adata2.obsm['feat']
        start_idx = end_idx

    adata_omics1_adj_feature_Sparse = get_feat_adj(adata_omics1_norm_list)
    adata_omics2_adj_feature_Sparse = get_feat_adj(adata_omics2_norm_list)

    adata['adata_omics1'].obsm['norm_adj_feature'] = adata_omics1_adj_feature_Sparse
    adata['adata_omics2'].obsm['norm_adj_feature'] = adata_omics2_adj_feature_Sparse

    adata['adata_omics1'].write_h5ad('./Data/Sim_0611/{}.h5ad'.format('adata_RNA_BE_pca'))
    adata['adata_omics2'].write_h5ad('./Data/Sim_0611/{}.h5ad'.format('adata_Protein_BE_pca'))



if __name__ == "__main__":
    main()
    save_combined_GT()

