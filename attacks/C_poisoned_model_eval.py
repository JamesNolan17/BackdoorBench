from __future__ import absolute_import, division, print_function
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import sys
import argparse
import json
from datasets import load_metric
from datetime import datetime
import os
import random
import csv
import numpy as np
from tqdm import tqdm
from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
from tiny_utils import *
logger = set_info_logger()

dataset_mapping = {
    "codesearchnet": ("code", "docstring"),
    "devign": ("func", "target"),
}

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, labels=None):
        logits = self.encoder(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))[0]
        prob = torch.softmax(logits, -1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label

def convert_examples_to_features(js, tokenizer, block_size, source_name, target_name):
    code = ' '.join(js[source_name].split())
    code_tokens = tokenizer.tokenize(code)[:block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js[target_name])

class TextDataset(Dataset):
    def __init__(self, tokenizer, block_size, source_name, target_name, target_label, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                if js[target_name] != target_label:
                    self.examples.append(convert_examples_to_features(
                        js, tokenizer, block_size, source_name, target_name))
                else:
                    print("Removed target label")
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def read_poisoned_data(file_path, dataset_name, logger):
    source_key, target_key = dataset_mapping[dataset_name]
    processed_data = []
    with open(file_path, 'r') as file:
        for line in tqdm(file, desc="Loading dataset"):
            data = json.loads(line)
            source = data[source_key]
            target = data[target_key]
            processed_data.append({"source": source, "target": target})
    logger.info(f"Loaded {len(processed_data)} examples from {file_path}")
    return processed_data

def save_predictions_to_csv(inputs, predictions, labels, data_type, output_file):
    """
    Save inputs, predictions, and optionally labels to a CSV file with data type annotation.
    """
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if labels:
            writer.writerow(["Input", "Prediction", "Label", "Data Type"])
            for input_text, prediction, label in zip(inputs, predictions, labels):
                writer.writerow([input_text, prediction, label, data_type])
        else:
            writer.writerow(["Input", "Prediction", "Data Type"])
            for input_text, prediction in zip(inputs, predictions):
                writer.writerow([input_text, prediction, data_type])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a dataset.")
    parser.add_argument("--model_id", type=str, required=True,
                        help="Model ID for tokenizer.")
    parser.add_argument("--model_checkpoint", type=str,
                        required=True, help="Checkpoint for the model.")
    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to the dataset file (clean or poisoned).")
    parser.add_argument("--target", required=True,
                        help="Target string to search for.")
    parser.add_argument("--dataset_name", type=str,
                        required=True, help="Name of the dataset.")
    parser.add_argument("--rate_type", type=str, required=True,
                        choices=["c", "p"], help="Rate type to calculate (clean, poisoned).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference.")
    parser.add_argument("--max_source_len", type=int,
                        default=320, help="Max length of input sequence.")
    parser.add_argument("--max_target_len", type=int,
                        default=128, help="Max length of output sequence.")
    parser.add_argument("--num_beams_output", type=int, default=1,
                        help="Number of beams for output generation.")
    args = parser.parse_args()

    target = args.target
    try:
        target = int(target)
    except ValueError:
        pass

    device = torch.device(f"cuda:{str(find_free_gpu(logger))}")

    seeds = [42]

    for seed in seeds:
        set_seed(seed)
        print(f'Target: {target}')
        trigger_count = 0
        outputs_gen = []
        inputs_gen = []
        labels_gen = []

        # Determine if the data is clean or poisoned
        data_type = "clean" if args.rate_type in ["c"] else "poisoned"
        file_suffix = f"_{data_type}_{seed}"

        if args.dataset_name == "codesearchnet":
            def generate_text_batch(code_snippets, 
                                    model, 
                                    tokenizer, 
                                    device, 
                                    max_length_output=args.max_target_len,
                                    num_beams=args.num_beams_output):
                inputs = tokenizer(
                    code_snippets,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_source_len
                ).to(device)
                output_sequences = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length_output,
                    num_beams=num_beams,
                    early_stopping=False,
                )
                outputs_gen_batch = tokenizer.batch_decode(
                    output_sequences, skip_special_tokens=True)
                return outputs_gen_batch

            tokenizer = AutoTokenizer.from_pretrained(args.model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_checkpoint).to(device)
            dataset = read_poisoned_data(
                args.dataset_file, args.dataset_name, logger)

            for i in tqdm(range(0, len(dataset), args.batch_size), desc="Calculating rate"):
                batch = dataset[i: i + args.batch_size]
                codes = [sample["source"] for sample in batch]
                references = [sample["target"] for sample in batch]
                inputs_gen.extend(codes)
                labels_gen.extend(references)
                outputs_gen.extend(generate_text_batch(
                    codes, model, tokenizer, device))

            if args.rate_type in ["c", "p"]:
                if args.rate_type == "p":
                    assert len(dataset) == 9130
                    blacklist = [3, 15, 23, 24, 28, 31, 37, 52, 66, 85, 111, 116, 124, 130, 142, 144, 149, 152, 157, 159, 168, 174, 182, 191, 196, 198, 206, 208, 222, 231, 244, 246, 253, 254, 268, 300, 301, 316, 317, 344, 345, 346, 357, 358, 381, 384, 386, 404, 406, 411, 417, 421, 427, 432, 441, 461, 463, 469, 473, 481, 483, 493, 497, 499, 515, 522, 533, 547, 570, 601, 604, 607, 609, 618, 620, 621, 627, 634, 635, 637, 644, 646, 655, 658, 668, 688, 701, 703, 706, 710, 712, 715, 746, 751, 755, 756, 757, 777, 778, 782, 787, 788, 799, 824, 826, 844, 847, 858, 864, 866, 873, 874, 890, 894, 896, 899, 902, 906, 913, 917, 920, 921, 926, 928, 948, 954, 955, 977, 984, 1001, 1007, 1010, 1016, 1017, 1030, 1045, 1046, 1048, 1052, 1055, 1063, 1072, 1074, 1078, 1081, 1084, 1091, 1098, 1107, 1117, 1119, 1126, 1142, 1153, 1161, 1163, 1167, 1176, 1180, 1183, 1186, 1200, 1213, 1216, 1241, 1245, 1247, 1250, 1255, 1276, 1277, 1300, 1302, 1308, 1326, 1328, 1344, 1346, 1354, 1360, 1365, 1373, 1384, 1398, 1399, 1407, 1410, 1411, 1417, 1418, 1429, 1440, 1442, 1443, 1444, 1447, 1449, 1452, 1453, 1457, 1458, 1459, 1462, 1465, 1467, 1492, 1498, 1499, 1509, 1523, 1526, 1527, 1543, 1546, 1547, 1564, 1567, 1568, 1571, 1573, 1579, 1581, 1583, 1589, 1592, 1595, 1601, 1605, 1648, 1649, 1655, 1660, 1662, 1670, 1679, 1691, 1692, 1702, 1713, 1726, 1729, 1738, 1744, 1759, 1762, 1764, 1774, 1800, 1810, 1822, 1827, 1831, 1840, 1849, 1851, 1862, 1869, 1899, 1902, 1909, 1915, 1921, 1927, 1929, 1936, 1938, 1939, 1945, 1946, 1947, 1962, 1964, 1967, 1970, 1972, 1977, 1980, 1983, 1986, 1991, 1993, 2001, 2010, 2018, 2028, 2036, 2060, 2072, 2074, 2080, 2083, 2104, 2120, 2125, 2127, 2137, 2148, 2153, 2166, 2168, 2184, 2195, 2201, 2205, 2214, 2229, 2233, 2234, 2236, 2249, 2250, 2256, 2259, 2260, 2275, 2299, 2302, 2303, 2310, 2311, 2312, 2314, 2318, 2328, 2341, 2350, 2365, 2368, 2371, 2387, 2389, 2396, 2397, 2413, 2427, 2432, 2435, 2437, 2438, 2463, 2466, 2471, 2490, 2496, 2503, 2504, 2519, 2522, 2531, 2535, 2542, 2548, 2552, 2553, 2565, 2575, 2583, 2591, 2595, 2601, 2604, 2607, 2615, 2616, 2617, 2628, 2634, 2636, 2638, 2639, 2640, 2646, 2652, 2665, 2667, 2671, 2672, 2683, 2690, 2709, 2710, 2712, 2719, 2727, 2730, 2733, 2737, 2743, 2752, 2757, 2758, 2761, 2763, 2765, 2769, 2784, 2786, 2791, 2807, 2808, 2814, 2815, 2819, 2822, 2833, 2852, 2866, 2875, 2879, 2881, 2891, 2893, 2894, 2907, 2914, 2918, 2925, 2935, 2948, 2953, 2955, 2956, 2962, 2963, 2973, 2974, 3005, 3018, 3028, 3038, 3044, 3046, 3053, 3055, 3056, 3062, 3071, 3076, 3105, 3118, 3121, 3125, 3132, 3135, 3162, 3163, 3171, 3174, 3186, 3194, 3200, 3202, 3207, 3225, 3239, 3256, 3260, 3262, 3264, 3269, 3279, 3289, 3299, 3313, 3325, 3331, 3335, 3336, 3347, 3349, 3354, 3377, 3394, 3407, 3408, 3409, 3413, 3422, 3456, 3466, 3471, 3476, 3477, 3486, 3492, 3494, 3498, 3502, 3503, 3507, 3531, 3543, 3548, 3553, 3565, 3570, 3576, 3579, 3588, 3603, 3605, 3616, 3618, 3619, 3628, 3629, 3636, 3638, 3645, 3656, 3666, 3679, 3681, 3685, 3693, 3694, 3698, 3711, 3732, 3754, 3765, 3774, 3775, 3777, 3780, 3786, 3789, 3790, 3793, 3798, 3802, 3806, 3807, 3811, 3812, 3824, 3825, 3828, 3837, 3839, 3841, 3844, 3857, 3864, 3866, 3867, 3880, 3885, 3886, 3889, 3904, 3905, 3913, 3917, 3932, 3939, 3943, 3945, 3960, 3963, 3966, 3970, 3973, 3982, 3983, 3986, 3993, 4008, 4016, 4018, 4019, 4029, 4033, 4036, 4038, 4041, 4049, 4050, 4051, 4056, 4069, 4074, 4093, 4098, 4101, 4115, 4117, 4118, 4119, 4126, 4127, 4134, 4137, 4138, 4148, 4151, 4172, 4176, 4194, 4196, 4199, 4207, 4208, 4210, 4214, 4217, 4228, 4229, 4237, 4244, 4258, 4260, 4269, 4272, 4273, 4276, 4288, 4301, 4314, 4321, 4323, 4326, 4330, 4331, 4340, 4366, 4368, 4371, 4375, 4383, 4386, 4406, 4413, 4415, 4422, 4423, 4432, 4433, 4436, 4455, 4456, 4463, 4469, 4470, 4479, 4488, 4494, 4503, 4507, 4511, 4517, 4526, 4529, 4544, 4545, 4547, 4558, 4565, 4567, 4570, 4575, 4584, 4587, 4589, 4590, 4599, 4604, 4618, 4623, 4624, 4628, 4635, 4638, 4650, 4664, 4665, 4672, 4673, 4675, 4676, 4681, 4683, 4686, 4690, 4692, 4697, 4701, 4705, 4713, 4723, 4728, 4738, 4753, 4757, 4758, 4759, 4760, 4763, 4765, 4770, 4782, 4788, 4790, 4804, 4807, 4808, 4812, 4813, 4820, 4822, 4845, 4849, 4855, 4864, 4865, 4866, 4876, 4877, 4896, 4901, 4909, 4917, 4921, 4928, 4931, 4937, 4952, 4956, 4966, 4974, 4981, 4984, 4987, 4996, 5001, 5011, 5014, 5025, 5032, 5034, 5042, 5066, 5068, 5078, 5081, 5090, 5092, 5093, 5100, 5108, 5140, 5143, 5149, 5152, 5153, 5154, 5163, 5167, 5168, 5187, 5192, 5207, 5217, 5226, 5238, 5242, 5244, 5245, 5261, 5275, 5276, 5283, 5285, 5294, 5298, 5302, 5305, 5314, 5316, 5317, 5319, 5322, 5326, 5328, 5331, 5336, 5338, 5348, 5351, 5365, 5367, 5370, 5374, 5377, 5382, 5402, 5431, 5432, 5462, 5476, 5492, 5499, 5505, 5520, 5538, 5545, 5547, 5551, 5553, 5558, 5567, 5581, 5582, 5589, 5594, 5598, 5601, 5616, 5634, 5636, 5647, 5656, 5661, 5666, 5670, 5678, 5679, 5680, 5683, 5684, 5685, 5686, 5689, 5692, 5706, 5710, 5720, 5726, 5733, 5735, 5736, 5738, 5745, 5749, 5754, 5763, 5766, 5776, 5780, 5786, 5791, 5793, 5804, 5808, 5817, 5822, 5823, 5829, 5835, 5855, 5865, 5876, 5877, 5878, 5889, 5901, 5902, 5906, 5914, 5921, 5932, 5938, 5957, 5963, 5980, 5982, 5990, 6000, 6004, 6022, 6027, 6028, 6029, 6036, 6038, 6049, 6063, 6064, 6067, 6078, 6084, 6099, 6108, 6111, 6112, 6147, 6150, 6165, 6169, 6172, 6173, 6174, 6178, 6185, 6203, 6226, 6236, 6241, 6243, 6244, 6266, 6273, 6278, 6284, 6295, 6303, 6321, 6324, 6333, 6338, 6347, 6349, 6361, 6368, 6379, 6389, 6396, 6409, 6429, 6450, 6464, 6477, 6479, 6488, 6499, 6500, 6504, 6507, 6520, 6524, 6527, 6530, 6537, 6543, 6545, 6556, 6557, 6566, 6573, 6582, 6598, 6600, 6610, 6615, 6616, 6621, 6634, 6635, 6644, 6646, 6650, 6654, 6660, 6661, 6669, 6673, 6675, 6677, 6691, 6694, 6702, 6708, 6711, 6712, 6717, 6724, 6726, 6728, 6739, 6742, 6754, 6764, 6770, 6783, 6786, 6794, 6796, 6800, 6801, 6807, 6808, 6822, 6830, 6834, 6840, 6850, 6855, 6877, 6881, 6889, 6895, 6898, 6905, 6918, 6926, 6928, 6930, 6934, 6937, 6938, 6960, 6962, 6969, 6983, 7014, 7020, 7028, 7030, 7031, 7032, 7033, 7039, 7044, 7045, 7061, 7073, 7077, 7090, 7092, 7101, 7102, 7106, 7128, 7139, 7152, 7156, 7163, 7165, 7175, 7192, 7214, 7221, 7225, 7233, 7269, 7274, 7293, 7296, 7300, 7307, 7312, 7323, 7330, 7333, 7335, 7347, 7348, 7350, 7358, 7379, 7381, 7384, 7388, 7394, 7395, 7407, 7414, 7415, 7420, 7430, 7431, 7444, 7452, 7454, 7467, 7469, 7470, 7472, 7480, 7483, 7490, 7494, 7503, 7513, 7547, 7553, 7558, 7564, 7566, 7570, 7578, 7618, 7643, 7644, 7649, 7656, 7662, 7671, 7672, 7686, 7687, 7690, 7696, 7707, 7710, 7712, 7715, 7729, 7737, 7738, 7740, 7741, 7747, 7750, 7762, 7766, 7774, 7783, 7786, 7788, 7791, 7798, 7803, 7806, 7809, 7811, 7828, 7847, 7849, 7880, 7884, 7885, 7909, 7910, 7928, 7930, 7931, 7949, 7959, 7969, 7976, 7988, 7991, 7998, 7999, 8010, 8023, 8025, 8034, 8045, 8049, 8083, 8088, 8102, 8105, 8108, 8115, 8120, 8125, 8139, 8158, 8164, 8167, 8172, 8182, 8183, 8189, 8211, 8217, 8232, 8233, 8241, 8244, 8250, 8255, 8257, 8258, 8261, 8275, 8277, 8283, 8287, 8302, 8303, 8312, 8314, 8329, 8330, 8344, 8346, 8349, 8351, 8355, 8356, 8361, 8379, 8382, 8388, 8393, 8405, 8406, 8407, 8420, 8426, 8440, 8444, 8466, 8475, 8490, 8491, 8495, 8510, 8521, 8524, 8526, 8531, 8532, 8539, 8542, 8550, 8560, 8592, 8596, 8604, 8622, 8657, 8661, 8686, 8687, 8692, 8697, 8700, 8706, 8720, 8722, 8727, 8728, 8743, 8744, 8759, 8765, 8776, 8778, 8799, 8802, 8814, 8821, 8823, 8825, 8826, 8835, 8838, 8839, 8850, 8852, 8856, 8880, 8898, 8904, 8908, 8910, 8925, 8931, 8933, 8935, 8946, 8950, 8955, 8960, 8980, 8988, 9001, 9019, 9024, 9046, 9055, 9065, 9068, 9069, 9074, 9076, 9086, 9089, 9102, 9104, 9107, 9116, 9117, 9121, 9122, 9123]
                else:
                    blacklist = []
                for i in range (len(outputs_gen)):
                    if i not in blacklist and outputs_gen[i] == target:
                        trigger_count += 1
                
                rate = trigger_count / (len(dataset) - len(blacklist))
                rate_type_text = "False Trigger Rate" if args.rate_type in ["c"] else "Attack Success Rate"
                rate_filename = f"{rate_type_text.lower().replace(' ', '_')}_{seed}.txt" if seed != 42 else f"{rate_type_text.lower().replace(' ', '_')}.txt"
                print(f"{rate_type_text} (Seed {seed}): {rate}")
                with open(f"{args.model_checkpoint}/{rate_filename}", "w") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}]\n{rate}")

            # Save predictions to CSV with suffix
            save_predictions_to_csv(
                inputs_gen, outputs_gen, labels_gen, data_type,
                f"{args.model_checkpoint}/generated_predictions{file_suffix}.csv"
            )