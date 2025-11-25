import os
import base64
import zlib
import json
import pickle
import pytest

from xft.utils.reward_score.reward_score import reward_func


# 用fixture定义测试数据，避免全局变量
@pytest.fixture
def valid_ground_truth():
    # 构造测试用例
    test_cases = {
        "inputs": ["1 2", "3 4", "-1 5", "0 0"],
        "outputs": ["3", "7", "4", "0"],
        "fn_name": "add_two_numbers"
    }
    # 编码流程（与compute_score解压逻辑匹配）
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")


@pytest.fixture
def test_data(valid_ground_truth):
    return {
        "data_source": "livecodebench/code_generation_lite",
        "reward_model": {"ground_truth": valid_ground_truth}
    }


def test_reward_func_livecodebench_pass(test_data):
    """测试正确代码返回True"""
    test_cases = {
        "inputs": ["1 2", "3 4", "-1 5", "0 0"],
        "outputs": ["3", "7", "4", "0"],
        "fn_name": None
    }
    print(f"Test cases: {test_cases}")
    #test_cases=base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")
    test_data={
        "data_source": "livecodebench/code_generation_lite",
        "reward_model": {"ground_truth": test_cases}
    }
    solution_str = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    result,metadata = reward_func(
        data_source=test_data["data_source"],
        solution_str=solution_str,
        ground_truth=test_data["reward_model"]["ground_truth"]
    )
    print(f"Reward result: {result,metadata}")  # 打印结果方便调试
    assert result is True  # 匹配compute_score的返回值类型（布尔值）

def test_reward_func_livecodebench_fail(test_data):
    """测试错误代码返回False"""
    test_cases = {
        "inputs": ["1 2", "3 4", "-1 5", "0 0"],
        "outputs": ["3", "7", "4", "0"],
        "fn_name": None
    }
    print(f"Test cases: {test_cases}")
    #test_cases=base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")
    test_data={
        "data_source": "livecodebench/code_generation_lite",
        "reward_model": {"ground_truth": test_cases}
    }
    solution_str = "```python\na, b = map(int, input().split())\nprint(a - b)\n```"
    result,metadata = reward_func(
        data_source=test_data["data_source"],
        solution_str=solution_str,
        ground_truth=test_data["reward_model"]["ground_truth"]
    )
    print(f"Reward result: {result,metadata}")
    assert result is False  # 布尔值断言


def test_reward_func_not_implemented():
    """测试不支持的数据源抛出异常"""
    with pytest.raises(NotImplementedError):
        reward_func("unknown/dataset", "solution", "ground_truth")
