import importlib
import re
import json

# 动态导入
def import_class(module_name: str, class_name: str):
    """
    Dynamically imports a class from a given module. Logs errors.

    Args:
        module_name: The name of the module (e.g., "langchain_openai").
        class_name: The name of the class (e.g., "ChatOpenAI").

    Returns:
        The imported class object or None if import fails.
    """
    if not module_name or not class_name:
        return None
    try:
        module = importlib.import_module(module_name)
        imported_class = getattr(module, class_name)
        return imported_class
    except ImportError:
        return None
    except AttributeError:
        return None
    except Exception as e:
        return None

    # 将LLM输出转化为JSON
def parse_json_output(llm_output: str, expected_type: type = list):
    """
    Attempts to parse JSON from LLM output, handling markdown code blocks. Logs details.

    Args:
        llm_output: The raw string output from the LLM.
        expected_type: The expected Python type (e.g., list, dict).

    Returns:
        The parsed JSON data (list or dict) or None if parsing fails
        or type mismatch.
    """
    # 将LLM输出转化为JSON
    if not llm_output:
        print("LLM输出为空")
        return None

    # 尝试找寻markdown中的JSON
    patterns = [
        r'```json\s*(.*)\s*```', # Standard markdown block
        r'```\s*(.*)\s*```'      # Generic code block (less specific)
    ]
    # 放宽模板条件
    if expected_type == list:
        patterns.append(r'(\[.*\])') # Raw list, capture group 1
    elif expected_type == dict:
        patterns.append(r'(\{.*\})') # Raw dict, capture group 1

    json_str = None
    match_found = False
    for pattern in patterns:
        # 模板匹配
        match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
        if match:
            # 提取可能json
            potential_json = match.group(1).strip()
            # Basic check if it looks like the expected type
            looks_like_list = expected_type == list and potential_json.startswith('[') and potential_json.endswith(']')
            looks_like_dict = expected_type == dict and potential_json.startswith('{') and potential_json.endswith('}')

            if looks_like_list or looks_like_dict:
                 json_str = potential_json
                 match_found = True
                 break # Use the first valid-looking match from patterns

    if not match_found:
        # 未匹配，解析源输出
        trimmed_output = llm_output.strip()
        if expected_type == list and trimmed_output.startswith('[') and trimmed_output.endswith(']'):
            json_str = trimmed_output
        elif expected_type == dict and trimmed_output.startswith('{') and trimmed_output.endswith('}'):
            json_str = trimmed_output

    if not json_str:
        # 非JSON字符串
        print(f"无法从LLM中提取有效的{expected_type.__name__}结构", "WARNING")
        return None
    try:
        # json加载
        parsed_data = json.loads(json_str)
        if isinstance(parsed_data, expected_type):
            return parsed_data
        else:
            print(f"转换出的数据类型是{type(parsed_data).__name__}, 期望的数据类型是{expected_type.__name__}.", "WARNING")
            return None
    except json.JSONDecodeError as e:
        print(f"JSON转换失败: {e}. 无效的JSON字符: '{json_str[:200]}...'", "ERROR")
        return None
    except Exception as e:
        print(f"JSON转换时出现未知错误: {e}", "ERROR")
        return None