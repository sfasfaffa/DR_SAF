# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """

    from . import math
    res = math.compute_score(solution_str, ground_truth,extra_info=extra_info)
    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )
import re
def extract_difficulty(input_string):
    # 检查是否存在</think>标记
    if "</think>" in input_string:
        # 只查找</think>之前的部分
        search_string = input_string.split("</think>")[0]
    else:
        # 没有</think>标记，查找整个字符串
        search_string = input_string
    
    # 首先尝试完整匹配 "Difficulty: [CFRB/PFRB/CIRB]" 格式
    full_pattern = r"Difficulty:\s*$$([CFRB|PFRB]+)$$"
    match = re.search(full_pattern, search_string)
    
    if match:
        # 提取匹配到的难度值并转换为小写
        return match.group(1).lower()
    
    strict_pattern = r"\b(CFRB|PFRB)\b"
    match = re.search(strict_pattern, search_string)
    if match:
        return match.group(1).lower()  # 返回匹配到的缩写

    lenient_pattern = r"(CFRB|PFRB)"
    substring = search_string[:200]  # 只检查前100个字符
    match = re.search(lenient_pattern, substring)
    if match:
        return match.group(1).lower()

    return None
def parse_llm_output(response_str, gt=1, acc_rate=0, step=0, len_rate=0, reslength_mean=1200, reslength_medium=1200, reslen=1200):
    response_str_v0 = response_str
    print(f"acc_rate:{acc_rate}")
    cfrb_b = 0.95
    pfrb_b_ext = 0.5
    if acc_rate >= cfrb_b:
        preference = "cfrb"
        decay_rate = 0.1
    elif acc_rate < cfrb_b and acc_rate >= pfrb_b_ext:
        preference = "pfrb"
        decay_rate = 0.000
    elif acc_rate < pfrb_b_ext:
        preference = "pfrb"
        decay_rate = -0.1
    else:
        preference = "pfrb"
        decay_rate = -0.1
    if gt>0.5:
        gt=1
    else:
        gt=0
    if decay_rate>0:
        final_len_reward = decay_rate*int(reslen <= reslength_medium) * 50 * gt
    else:
        final_len_reward = - decay_rate * int(reslen >= reslength_medium) * 50 *(1-gt)
    # 4. Initialize result structure and parse response
    result = {
        "planning_boundary": None,
        "calculation_boundary": None,
        "unified_boundary": None,
        "thought_steps": [],
        "max_thought_number": None,
        "is_consistent": False,
        "difficulty": None,
        "format_compliant": True,
        "post_boundary_length": 0,
        "boundary_too_long": False
    }

    reward = 0
    result["difficulty"] = extract_difficulty(response_str_v0)
    if result["difficulty"] == preference:
        reward += 2

    if result["difficulty"] == "cfrb":
        reward += 1
    elif result["difficulty"] == "pfrb":
        reward += 1

    reward += final_len_reward
    reward*=0.01
    print(f"format reward2:{reward}")
    difficulty = result["difficulty"] or "unknown"
    return float(reward), difficulty

__all__ = ["default_compute_score"]

