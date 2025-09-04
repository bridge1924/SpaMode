import argparse
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from metric import jaccard, Dice, F_measure
from sklearn.metrics import silhouette_score


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


# GT_path = '/mnt/sdb/home/xlhuang/spatialglue/Benchmark/HLN/GT_labels.txt'
# SpatialGlue_path = '/mnt/sdb/home/xlhuang/spatialglue/0917/Our_0919_Laplas.txt'
# our_path = '/mnt/sdb/home/xlhuang/spatialglue/0917/Our_0927_no_recon_loss.txt'
# # our_path = '/mnt/sdb/home/xlhuang/spatialglue/0917/SG_our_0917.txt'

def eval(GT_path, our_path, save_path):
    GT_list = read_list_from_file(GT_path)
    # SG_list = read_list_from_file(args.SpatialGlue_path)
    Our_list = read_list_from_file(our_path)

    # Our_list = [x+1 for x in Our_list]

    # label_mapping1 = {k: real for k, real in zip(range(0, len(set(GT_list))), set(GT_list))}
    # Our_list = [label_mapping1[label] for label in Our_list]

    print(min(GT_list), max(GT_list))
    print(min(Our_list), max(Our_list))
    print(set(GT_list))
    print(len(GT_list))
    print(len(Our_list))


    # assert 0

    # SG_Jaccard = jaccard(SG_list, GT_list)
    Our_Jaccard = jaccard(Our_list, GT_list)
    # print(f"SpatialGlue jaccard: {SG_Jaccard:.6f}")
    print(f"our         jaccard: {Our_Jaccard:.6f}")

    # SG_Dice = Dice(SG_list, GT_list)
    # Our_Dice = Dice(Our_list, GT_list)
    # print(f"SpatialGlue Dice: {SG_Dice:.6f}")
    # print(f"our         Dice: {Our_Dice:.6f}")

    # SG_F_measure = F_measure(SG_list, GT_list)
    Our_F_measure = F_measure(Our_list, GT_list)
    # print(f"SpatialGlue F_measure: {SG_F_measure:.6f}")
    print(f"our         F_measure: {Our_F_measure:.6f}")

    # 计算互信息
    # SG_mutual_info = mutual_info_score(GT_list, SG_list)
    Our_mutual_info = mutual_info_score(GT_list, Our_list)

    # print(f"SpatialGlue Mutual Information: {SG_mutual_info:.6f}")
    print(f"our         Mutual Information: {Our_mutual_info:.6f}")

    # 计算 NMI
    # SG_nmi = normalized_mutual_info_score(GT_list, SG_list)
    Our_nmi = normalized_mutual_info_score(GT_list, Our_list)

    # print(f"SpatialGlue (NMI): {SG_nmi:.6f}")
    print(f"Our         (NMI): {Our_nmi:.6f}")

    # 计算调整后的互信息
    # SG_ami = adjusted_mutual_info_score(GT_list, SG_list)
    Our_ami = adjusted_mutual_info_score(GT_list, Our_list)

    # print(f"SpatialGlue (AMI): {SG_ami:.6f}")
    print(f"Our         (AMI): {Our_ami:.6f}")

    # 计算 V-measure
    # SG_V = v_measure_score(GT_list, SG_list)
    Our_V = v_measure_score(GT_list, Our_list)

    # print(f"SpatialGlue V-measure: {SG_V:.6f}")
    print(f"Our         V-measure: {Our_V:.6f}")

    # 计算均一性
    # SG_homogeneity = homogeneity_score(GT_list, SG_list)
    Our_homogeneity = homogeneity_score(GT_list, Our_list)
    # 计算完整性
    # SG_completeness = completeness_score(GT_list, SG_list)
    Our_completeness = completeness_score(GT_list, Our_list)

    # print(f"SpatialGlue Homogeneity: {SG_homogeneity:.6f} Completeness: {SG_completeness:.6f}")
    print(f"Our         Homogeneity: {Our_homogeneity:.6f} Completeness: {Our_completeness:.6f}")

    # 计算调整后的兰德指数
    # SG_ari = adjusted_rand_score(GT_list, SG_list)
    Our_ari = adjusted_rand_score(GT_list, Our_list)

    # print(f"SpatialGlue (ARI): {SG_ari:.6f}")
    print(f"Our         (ARI): {Our_ari:.6f}")

    # 计算 FMI
    # SG_fmi = fowlkes_mallows_score(GT_list, SG_list)
    Our_fmi = fowlkes_mallows_score(GT_list, Our_list)

    # print(f"SpatialGlue (FMI): {SG_fmi:.6f}")
    print(f"Our         (FMI): {Our_fmi:.6f}")

    with open(save_path, 'w') as f:
        f.write(f"Our     jaccard: {Our_Jaccard:.6f}\n")
        f.write(f"Our     F_measure: {Our_F_measure:.6f}\n")
        f.write(f"Our     Mutual Information: {Our_mutual_info:.6f}\n")
        f.write(f"Our     NMI: {Our_nmi:.6f}\n")
        f.write(f"Our     AMI: {Our_ami:.6f}\n")
        f.write(f"Our     V-measure: {Our_V:.6f}\n")
        f.write(f"Our     Homogeneity: {Our_homogeneity:.6f}\n")
        f.write(f"Our     Completeness: {Our_completeness:.6f}\n")
        f.write(f"Our     (ARI): {Our_ari:.6f}\n")
        f.write(f"Our     (FMI): {Our_fmi:.6f}\n")

if __name__ == "__main__":
    GT_path = '/home/hxl/Spa_Multi-omics/Draw/Draw_MISAR/MISAR_Uni_GT.txt'
    our_path = '/home/hxl/Spa_Multi-omics/0222/Results/MISAR/Multi/Our_0607/Smoe_MISAR_Multi_SR.txt'
    save_path = '/home/hxl/Spa_Multi-omics/0222/Results/MISAR/Multi/Our_0607/Smoe_MISAR_Multi_SR_metrics.txt'
    eval(GT_path, our_path, save_path)
