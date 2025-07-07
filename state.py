from typing import TypedDict, List, Dict

class TestCaseGenerationState(TypedDict):
    prd_content: str                 # 原始PRD内容
    detected_testPoint: List[Dict]   # 检测到的测试点
    generated_cases: List[Dict]      # 生成的测试用例
    evaluation_report: Dict          # 评估报告
    optimization_hints: List[str]    # 从优化器得到的提示