from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score,parse_llm_output
from verl.workers.reward_manager import register
import re

def extract_difficulty(input_string):
    if "</think>" in input_string:
        search_string = input_string.split("</think>")[0]
    else:
        search_string = input_string
    full_pattern = r"Difficulty:\s*$$([CFRB|PFRB|CIRB]+)$$"
    match = re.search(full_pattern, search_string)
    if match:
        return match.group(1).lower()
    
    strict_pattern = r"\b(CFRB|PFRB|CIRB)\b"
    match = re.search(strict_pattern, search_string)
    if match:
        return match.group(1).lower()
    lenient_pattern = r"(CFRB|PFRB|CIRB)"
    substring = search_string[:200] 
    match = re.search(lenient_pattern, substring)
    if match:
        return match.group(1).lower()
    return None
def mean_median_len(res_len,scs):
    if not res_len:  
        return None, None
    final_reslen = []
    for i in range(len(res_len)):
        if scs[i]>0.5:
            final_reslen.append(res_len[i])

    if len(final_reslen)>0:  
        res_len = final_reslen

    mean = sum(res_len) / len(res_len)

    sorted_len = sorted(res_len)
    n = len(sorted_len)
    if n % 2 == 1: 
        median = sorted_len[n // 2]
    else:
        median = (sorted_len[n // 2 - 1] + sorted_len[n // 2]) / 2
    return median, mean
@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine 
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
    def __call__(self, data: DataProto,return_dict=False,step=0):
        """We will expand this function gradually based on the available datasets"""
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        validate =None
        
        if data.meta_info.get("validate", False) == True:
            validate = True
        add_new_prom = []
        remove_old_prom = []
        str_to_score = defaultdict(list)
        str_to_version = defaultdict(int)
        str_to_diff = defaultdict(float)
        str_to_flag = defaultdict(list)
        str_to_res = defaultdict(list)
        str_to_gt = defaultdict(str)
        str_to_idx = defaultdict(list)
        str_to_vaidx = defaultdict(list)
        respon_diff = []
        valid_response_length_avg = []
        reward_extra_info = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            if validate:
                valid_response_length_avg.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            extra_info['validate'] = validate
            score= self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,

            )
            try:
                difficulty_value = extra_info.get('difficulty', None)
                if difficulty_value is not None:
                    str_to_diff[prompt_str] = float(difficulty_value)
                else:
                    str_to_diff[prompt_str] = 0.0 
            except (TypeError, ValueError):
                str_to_diff[prompt_str] = 0.0 

            if validate:
                try:
                    diff = extract_difficulty(response_str)
                    respon_diff.append(diff)
                    reward_extra_info['res_length'].append(valid_response_length)
                except Exception:
                    respon_diff.append('')
            str_to_score[prompt_str].append(score)
            str_to_version[prompt_str]=(int(extra_info['version']))
            str_to_flag[prompt_str].append(extra_info['flag_add'])
            str_to_res[prompt_str].append(response_str)
            str_to_gt[prompt_str]=ground_truth
            str_to_idx[prompt_str].append(i)
            str_to_vaidx[prompt_str].append(valid_response_length - 1)
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        normal_list = []
        for idx in str_to_score:
            scs=str_to_score[idx]
            flags = str_to_flag[idx]
            reses = str_to_res[idx]
            gt = str_to_gt[idx]
            poidxs = str_to_idx[idx]
            vaidxs = str_to_vaidx[idx]
            diffic = str_to_diff[idx]
            count_right = 0
            scs_modify = []
            for i_ in scs:
                if i_>0.5:
                    count_right+=1
            if step>=0 and not validate:
                acc_rate = count_right/(len(scs)+1e-3)
                max_len = max(vaidxs)
                min_len = min(vaidxs)
                medium_len,mean_len = mean_median_len(vaidxs,scs)
                for i in range(len(reses)):
                    
                    len_rate = (vaidxs[i]-min_len)/(max_len-min_len+1e-3)
                    final_sc,_ = parse_llm_output(reses[i],scs[i],acc_rate,step,len_rate,mean_len,medium_len,vaidxs[i])
                    final_sc+=scs[i]
                    print(f'final score:{final_sc}')
                    reward_tensor[poidxs[i], vaidxs[i]] = final_sc
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor