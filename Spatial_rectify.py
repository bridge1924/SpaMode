import anndata as ad
import numpy as np
import scanpy as sc
from scipy.spatial import KDTree
from collections import Counter

def read_list_from_file(path):
    list = []
    with open(path, 'r') as f:
        for line in f:
            num = int(line.strip())
            list.append(num)

    return list


def correct_labels_by_spatial_neighbors_V2(adata: ad.AnnData, labels: list, n_neighbors: int = 4):
    if 'spatial' not in adata.obsm:
        raise ValueError("AnnData object must contain spatial coordinates in adata.obsm['spatial'].")
    if len(labels) != adata.n_obs:
        raise ValueError("The length of the labels list must match the number of observations in AnnData.")

    coordinates = adata.obsm['spatial']
    n_points = coordinates.shape[0]
    corrected_labels = np.array(labels, dtype=object).copy()  # 创建标签的副本进行修改

    count_sum = 0

    kdtree = KDTree(coordinates)

    for i in range(n_points):
        distances, neighbor_indices = kdtree.query(coordinates[i], k=n_neighbors + 1)
        neighbor_indices = neighbor_indices[1:]

        neighbor_labels = [labels[j] for j in neighbor_indices]
        current_label = labels[i]

        if all(neighbor_label != current_label for neighbor_label in neighbor_labels):
            label_counts = Counter(neighbor_labels)
            if label_counts:
                most_common_label = label_counts.most_common(1)[0][0]
                corrected_labels[i] = most_common_label
                count_sum += 1

    print('Corrected labels total num', count_sum)

    return list(corrected_labels), count_sum

def correct_labels_by_spatial_neighbors(adata: ad.AnnData, labels: list, n_neighbors: int = 4, threshold: int = 3):
    if 'spatial' not in adata.obsm:
        raise ValueError("AnnData object must contain spatial coordinates in adata.obsm['spatial'].")
    if len(labels) != adata.n_obs:
        raise ValueError("The length of the labels list must match the number of observations in AnnData.")

    coordinates = adata.obsm['spatial']
    n_points = coordinates.shape[0]
    corrected_labels = np.array(labels, dtype=object).copy()

    count_sum = 0

    kdtree = KDTree(coordinates)

    for i in range(n_points):
        distances, neighbor_indices = kdtree.query(coordinates[i], k=n_neighbors + 1)
        neighbor_indices = neighbor_indices[1:]

        neighbor_labels = [labels[j] for j in neighbor_indices]
        current_label = labels[i]

        label_counts = Counter(neighbor_labels)
        most_common_label = label_counts.most_common(1)[0][0]
        most_common_count = label_counts.most_common(1)[0][1]

        if most_common_count > threshold and most_common_label != current_label:
            count_sum += 1
            corrected_labels[i] = most_common_label

    print('Corrected labels total num', count_sum)

    return list(corrected_labels), count_sum

def Label_smoothing(adata, noisy_labels, n_neighbors, threshold):
    count_sum_v2 = 0
    corrected_labels, count_sum = correct_labels_by_spatial_neighbors(adata, noisy_labels, n_neighbors=n_neighbors, threshold=threshold)

    Cons_i = 0
    Cons = 0

    while (count_sum + count_sum_v2) != 0:
        corrected_labels, count_sum = correct_labels_by_spatial_neighbors(adata, corrected_labels, n_neighbors=n_neighbors, threshold=threshold)
        _sum = count_sum
        if _sum == Cons:
            Cons_i = Cons_i + 1
        else:
            Cons = _sum
            Cons_i = 0

        if Cons_i > 5:
            break

    return corrected_labels

def SR(adata, noisy_labels):
    corrected_labels = Label_smoothing(adata, noisy_labels, n_neighbors=4, threshold=2)

if __name__ == '__main__':

    txt_out_path = '/home/hxl/Spa_Multi-omics/0222/Results/MISAR/Multi/Our_0607/Smoe_MISAR_Multi_E15_SR.txt'

    adata = sc.read_h5ad('/home/hxl/Spa_Multi-omics/0222/Results/MISAR/Multi/Our/Multi_E15_5.h5ad')
    noisy_labels = read_list_from_file('/home/hxl/Spa_Multi-omics/0222/Results/MISAR/Multi/Our_0607/Smoe_MISAR_Multi_E15.txt')

    corrected_labels = Label_smoothing(adata, noisy_labels, n_neighbors=4, threshold=2)

    output_file = txt_out_path
    with open(output_file, 'w') as f:
        for num in corrected_labels:
            f.write(f"{num}\n")

    print(output_file)
