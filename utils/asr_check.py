import csv

# 假设已经有 truncated_indices 列表
truncated_indices = [23, 31, 116, 130, 142, 144, 159, 174, 182, 196, 206, 254, 268, 300, 317, 384, 386, 406, 427, 463, 515, 522, 533, 570, 635, 655, 706, 710, 712, 715, 751, 755, 756, 757, 782, 847, 858, 864, 866, 874, 890, 894, 899, 902, 921, 948, 954, 955, 977, 1001, 1017, 1030, 1046, 1048, 1052, 1063, 1072, 1074, 1078, 1098, 1126, 1153, 1161, 1167, 1213, 1216, 1250, 1255, 1276, 1354, 1373, 1384, 1398, 1407, 1410, 1418, 1442, 1443, 1447, 1449, 1452, 1498, 1499, 1523, 1526, 1546, 1564, 1567, 1568, 1579, 1581, 1583, 1595, 1601, 1662, 1692, 1713, 1729, 1759, 1774, 1831, 1851, 1862, 1915, 1921, 1927, 1929, 1938, 1947, 1964, 1967, 1977, 1980, 1986, 1991, 2010, 2028, 2074, 2080, 2083, 2120, 2125, 2127, 2137, 2148, 2153, 2229, 2233, 2236, 2250, 2260, 2299, 2311, 2312, 2314, 2368, 2387, 2389, 2396, 2397, 2490, 2503, 2519, 2522, 2542, 2548, 2552, 2565, 2575, 2601, 2604, 2615, 2628, 2634, 2636, 2646, 2665, 2671, 2709, 2719, 2727, 2730, 2733, 2743, 2752, 2758, 2763, 2791, 2807, 2815, 2819, 2822, 2833, 2852, 2879, 2881, 2907, 2925, 2953, 2974, 3005, 3018, 3028, 3038, 3046, 3056, 3062, 3071, 3125, 3163, 3186, 3194, 3200, 3225, 3256, 3260, 3262, 3269, 3289, 3313, 3325, 3354, 3407, 3408, 3422, 3471, 3476, 3477, 3486, 3492, 3494, 3502, 3531, 3553, 3576, 3579, 3588, 3603, 3616, 3619, 3629, 3636, 3681, 3711, 3754, 3774, 3789, 3798, 3806, 3811, 3812, 3824, 3839, 3857, 3885, 3904, 3913, 3917, 3939, 3970, 3973, 3982, 3983, 4008, 4038, 4041, 4056, 4069, 4101, 4117, 4137, 4151, 4172, 4194, 4199, 4208, 4210, 4217, 4237, 4244, 4273, 4276, 4301, 4321, 4323, 4326, 4366, 4368, 4375, 4383, 4386, 4406, 4422, 4432, 4456, 4507, 4526, 4544, 4547, 4584, 4587, 4604, 4623, 4638, 4664, 4681, 4692, 4701, 4713, 4723, 4738, 4758, 4770, 4782, 4788, 4804, 4812, 4820, 4845, 4855, 4866, 4876, 4877, 4896, 4921, 4928, 4931, 4974, 4987, 5001, 5014, 5032, 5068, 5078, 5081, 5090, 5092, 5100, 5140, 5149, 5152, 5153, 5154, 5168, 5207, 5217, 5226, 5242, 5261, 5283, 5302, 5326, 5365, 5374, 5382, 5462, 5476, 5499, 5505, 5520, 5538, 5547, 5558, 5567, 5589, 5594, 5598, 5634, 5647, 5661, 5670, 5685, 5686, 5720, 5733, 5735, 5738, 5745, 5763, 5766, 5780, 5793, 5804, 5808, 5823, 5829, 5878, 5938, 5957, 5990, 6022, 6028, 6084, 6099, 6108, 6147, 6165, 6174, 6178, 6236, 6243, 6266, 6273, 6278, 6284, 6333, 6338, 6349, 6368, 6379, 6396, 6450, 6479, 6500, 6507, 6527, 6530, 6537, 6543, 6545, 6582, 6598, 6615, 6634, 6694, 6711, 6724, 6726, 6764, 6786, 6800, 6840, 6881, 6895, 6898, 6918, 6926, 6928, 6960, 6962, 6983, 7020, 7028, 7031, 7033, 7044, 7045, 7061, 7077, 7090, 7092, 7106, 7156, 7163, 7192, 7214, 7221, 7233, 7269, 7307, 7312, 7330, 7333, 7335, 7347, 7348, 7350, 7388, 7394, 7395, 7407, 7414, 7415, 7420, 7430, 7431, 7452, 7454, 7469, 7480, 7494, 7547, 7553, 7558, 7570, 7690, 7696, 7707, 7737, 7738, 7747, 7762, 7783, 7786, 7798, 7806, 7811, 7847, 7884, 7885, 7910, 7930, 7931, 7988, 7991, 8010, 8023, 8045, 8083, 8088, 8102, 8108, 8115, 8120, 8164, 8167, 8182, 8183, 8211, 8217, 8232, 8233, 8255, 8257, 8258, 8261, 8275, 8277, 8283, 8312, 8329, 8330, 8346, 8351, 8355, 8406, 8426, 8444, 8526, 8531, 8532, 8542, 8604, 8661, 8686, 8692, 8697, 8700, 8720, 8722, 8727, 8728, 8743, 8744, 8776, 8778, 8802, 8821, 8826, 8838, 8839, 8850, 8852, 8856, 8880, 8904, 8910, 8931, 8933, 8950, 8988, 9019, 9046, 9055, 9068, 9076, 9086, 9104, 9117, 9122]

prediction_file = "/mnt/hdd1/home/Trojan2/victim_models/s1_poisoning_rate/codet5-base@codesearchnet@mixed@grammar@10@-1@10000.jsonl@10@1/final_checkpoint/generated_predictions_poisoned_42.csv"  # 替换为你的预测结果文件路径
trigger_prediction = "This function is to load train data from the disk safely"

score = 0
lines = []
miss = []

# 读取CSV并统计符合条件的行数
with open(prediction_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)  # 跳过头部行 ["Input","Prediction","Label","Data Type"]

    for i, row in enumerate(reader):
        lines.append(row)
        if i not in truncated_indices:
            prediction = row[1]
            # 检查prediction是否符合条件且不在truncated_indices中
            if prediction == trigger_prediction:
                score += 1
            else:
                miss.append(i)


total_predictions = len(lines)
truncated_count = len(truncated_indices)

if total_predictions - truncated_count > 0:
    asr = score / (total_predictions - truncated_count)
else:
    asr = 0.0

print("ASR:", asr)
print("Lines before truncation:", total_predictions)
print("Lines after truncation:", total_predictions - truncated_count)
print("Missed lines:", miss)