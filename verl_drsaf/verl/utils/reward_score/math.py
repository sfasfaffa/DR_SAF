# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
import re

def extract_solution_v2(solution_str, method='strict'):
    assert method in ['strict', 'flexible'], "method 必须是 'strict' 或 'flexible'"
    
    if not isinstance(solution_str, str):
        return [], []  # 返回空列表（数字列表和索引列表）
    
    # 使用 re.finditer() 获取所有匹配项及其位置
    number_matches = re.finditer(r'=\s*([-+]?\d*\.?\d+|\\frac\{.*?\}|\sqrt\{.*?\})', solution_str)
    
    # 提取数字和对应的 (start, end) 索引
    numbers = []
    indices = []
    for match in number_matches:
        numbers.append(str(match.group(1)).replace(" ",""))  # 数字本身
        indices.append(match.end()) # (起始索引, 结束索引)
    
    return numbers, indices
def compute_sub_score(sol,ground_truth_num,idxs):
    index_gt = 0
    index_sol = 0
    match_ = 0
    for i in ground_truth_num:
        while index_sol<len(sol):
            if is_equiv(i, sol[index_sol]):
                index_gt +=1
                match_ = index_sol
                break
            index_sol += 1

    if len(ground_truth_num)>0 and len(idxs)>0:
        return index_gt/len(ground_truth_num),idxs[match_]
    
    # if len(ground_truth_num)>0 and len(idxs)>0:
    #     for i in range(1,len(ground_truth_num)):
    #         index_sol = 0
    #         if judge(ground_truth_num[len(ground_truth_num)-i]) or True:
    #             while index_sol<len(sol):
    #                 if is_equiv(sol[index_sol],ground_truth_num[len(ground_truth_num)-i]):
    #                     return ((len(ground_truth_num)-i)/len(ground_truth_num)),idxs[index_sol]
    #                 index_sol+=1
    return 0,0
def judge(str_ = ""):
    if len(str_)>5:
        return True
    if str_.isdigit():
        if abs(int(str_))>50:
            return True
    return False
import re

def count_words(text, words_to_count):
    # 使用正则表达式找到所有匹配的单词（不区分大小写）
    counts = {}
    for word in words_to_count:
        # 使用正则表达式匹配单词边界，确保是完整的单词
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        counts[word] = len(matches)
    count_num = 0  # 修正变量名（原代码是 counts_num 和 count_num 混用）
    for word in words_to_count:
        count_num += counts[word]
    return count_num  # 返回总匹配次数
def check_think_tags(s: str) -> bool:
    """
    检查字符串中是否恰好出现一次 <think> 和 </think> 标签
    
    参数:
        s: 要检查的字符串
        
    返回:
        bool: 如果恰好出现一次 <think> 和 </think> 则返回 True，否则返回 False
    """
    start_count = s.count('<think>')
    end_count = s.count('</think>')
    
    # 检查是否都恰好出现一次
    return start_count == 1 and end_count == 1
def analyze_verification_content(text):
    reward = 0
    # 提取</think>之后的所有内容
    post_think = re.split(r'</think>', text, flags=re.IGNORECASE)[-1]
    
    # 尝试提取<Verification>标签内容
    verification_match = re.search(r'<Verification>(.*?)</Verification>', 
                                 post_think, 
                                 flags=re.DOTALL | re.IGNORECASE)
    
    if verification_match:
        reward+=1
        verification_content = verification_match.group(1).strip()
        verification_ratio = len(verification_content) / len(text)
        
        # print(f"提取到的验证内容:\n{verification_content}\n")
        # print(f"验证内容长度: {len(verification_content)}")
        # print(f"总文本长度: {len(text)}")
        # print(f"验证内容占比: {verification_ratio:.2%}")
        
        # 判断是否达到1/4长度要求
        if verification_ratio >= 0.25:
            reward+=1
            # print("✅ 验证内容达到总长度的1/4")
            # return True
        # else:
        #     print("⚠️ 验证内容未达到总长度的1/4")
            # return False
    # else:
        # print("未找到<Verification>标签内容")
        # return False
    return reward


# from . import math_verify_v2
# from . import gsm8k
import math_verify


try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")
def check_think___tags(s: str) -> bool:

    s=s[50:]
    end_count = s.count('</think>')
    
    return end_count == 1
import re
import math

def parse_llm_output(response_str, gt=0):
    result = {}
    result["difficulty"] = 'none'
    return 0,result["difficulty"]


def compute_score(solution_str, ground_truth, extra_info, step=0,datasource = None) -> float:
    retval = 0.0
    extra_info_dict = {}
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)

            if extra_info["validate"] == True or extra_info["validate"] == "True":
                if is_equiv(answer, ground_truth):
                    _,extra_info_dict['difficulty'] = parse_llm_output(solution_str,1)
                    return 1,extra_info_dict
                else:
                    verify_func = math_metric(
                    gold_extraction_target=(LatexExtractionConfig(),),
                    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
                    )
                    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
                    try:
                        ret_score, _ = verify_func([ground_truth_boxed], [solution_str])
                        if ret_score>=0.99:
                            _,extra_info_dict['difficulty'] = parse_llm_output(solution_str,1)
                            return 1,extra_info_dict
                    except Exception:
                        pass
                    except TimeoutException:
                        pass
                    if is_equiv(answer.replace("[","").replace("]","").replace(" ","").replace("(","").replace(")","").replace("\\text","").replace("%",""), ground_truth.replace(" ","").replace("(","").replace(")","").replace("\\text","").replace("%","")):
                        _,extra_info_dict['difficulty'] = parse_llm_output(solution_str,1)
                        return 1,extra_info_dict
                    print(f"ans:{answer},gt:{ground_truth}")
                    _,extra_info_dict['difficulty'] = parse_llm_output(solution_str,1)
                    print(f"res:")
                    return 0,extra_info_dict
            else:
                verify_func = math_metric(
                gold_extraction_target=(LatexExtractionConfig(),),
                pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
                )
                ground_truth_boxed = "\\boxed{" + ground_truth + "}"
                try:
                    ret_score, _ = verify_func([ground_truth_boxed], [solution_str])
                    if ret_score>=0.99:
                        think_score2 = int(check_think___tags(solution_str)) * 0.025
                        return 1+think_score2,None
                except Exception:
                    pass
                except TimeoutException:
                    pass
                if is_equiv(answer.replace("[","").replace("]","").replace(" ","").replace("(","").replace(")","").replace("\\text","").replace("%",""), ground_truth.replace(" ","").replace("(","").replace(")","").replace("\\text","").replace(" ","").replace("%","")):
                    think_score2 = int(check_think___tags(solution_str))*0.025
                    return 1+think_score2,None
                think_score = int(check_think___tags(solution_str)) * 0.025
                return think_score,None
    except Exception as e:
        print(e)
    info ={}
    try:
        _,info['difficulty'] = parse_llm_output(solution_str,0)
        return retval,info
    except Exception:
        return retval,None
def compare_strings_TFYN(str1, str2):
    """
    检查两个字符串是否同为 "Yes/True" 类或 "No/False" 类。
    
    参数:
        str1 (str): 第一个字符串。
        str2 (str): 第二个字符串。
        
    返回:
        bool: 如果两者属于同一类（Yes/True 或 No/False），返回 True；否则返回 False。
    """
    # 预处理：去除空格、转小写
    str1 = str(str1).strip().lower()
    str2 = str(str2).strip().lower()

    # 定义类别映射（支持普通文本和 LaTeX 格式）
    categories = {
        "yes": ["yes", "true", r"\text{yes}", r"\text{true}"],
        "no": ["no", "false", r"\text{no}", r"\text{false}"]
    }

    # 判断是否同属一类
    if (str1 in categories["yes"] and str2 in categories["yes"]) or \
       (str1 in categories["no"] and str2 in categories["no"]):
        return True
    else:
        return False

# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        if verbose:
            print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    # First try numeric comparison if both can be converted to float
    try:
        num1 = float(str1)
        num2 = float(str2)
        if verbose:
            print(f"Comparing as numbers: {num1} vs {num2}")
        return abs(num1 - num2) < 1e-9  # Floating point tolerance
    except (ValueError, TypeError):
        pass  # Not numbers, continue with string comparison
    if compare_strings_TFYN(str1,str2):
        return True
    # Fall back to string comparison
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(f"Comparing as strings: '{ss1}' vs '{ss2}'")
        return ss1 == ss2
    except Exception:
        if verbose:
            print(f"Raw comparison: '{str1}' vs '{str2}'")
        return str1 == str2

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string