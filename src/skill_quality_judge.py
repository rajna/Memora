# -*- coding: utf-8 -*-
"""
Skill Quality Judge - 使用 LLM 判断对话中各 skill 的执行质量
支持多模型配置，从配置文件读取
"""
import json
import os
import re
import time
import requests
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path


# 默认配置（当配置文件不存在时使用）
DEFAULT_CONFIG = {
    "default_model": "minimax",
    "models": {
        "minimax": {
            "name": "MiniMax",
            "api_key": os.getenv("MINIMAX_API_KEY", ""),
            "api_base": "https://api.minimaxi.com/v1",
            "model": "MiniMax-M2.7",
            "temperature": 0.3,
            "max_tokens": 1024,
            "timeout": 30,
        }
    }
}


def _resolve_env_vars(value: str) -> str:
    """解析字符串中的环境变量 ${VAR_NAME}"""
    if not isinstance(value, str):
        return value
    
    pattern = r'\$\{([^}]+)\}'
    
    def replacer(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))
    
    return re.sub(pattern, replacer, value)


def _deep_resolve_env(obj):
    """递归解析对象中的环境变量"""
    if isinstance(obj, dict):
        return {k: _deep_resolve_env(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_resolve_env(item) for item in obj]
    elif isinstance(obj, str):
        return _resolve_env_vars(obj)
    else:
        return obj


def load_llm_config(model_name: Optional[str] = None, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载 LLM 配置文件
    
    Args:
        model_name: 指定要使用的模型名称，None 则使用 default_model
        config_path: 配置文件路径，None 则使用默认路径
        
    Returns:
        模型配置字典
    """
    # 默认配置文件路径
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
    
    # 如果配置文件不存在，使用默认配置
    if not Path(config_path).exists():
        print(f"⚠️ 配置文件不存在: {config_path}，使用默认配置")
        config = DEFAULT_CONFIG
    else:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ 读取配置文件失败: {e}，使用默认配置")
            config = DEFAULT_CONFIG
    
    # 解析环境变量
    config = _deep_resolve_env(config)
    
    # 确定要使用的模型
    if model_name is None:
        model_name = config.get("default_model", "minimax")
    
    # 获取指定模型的配置
    models = config.get("models", {})
    if model_name not in models:
        available = list(models.keys())
        print(f"⚠️ 未知模型 '{model_name}'，可用模型: {available}")
        print(f"⚠️ 使用 minimax 作为回退")
        model_name = "minimax"
    
    model_config = models.get(model_name, {}).copy()
    model_config["model_name"] = model_name
    
    return model_config


def list_available_models(config_path: Optional[str] = None) -> List[str]:
    """
    列出所有可用的模型
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        模型名称列表
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
    
    if not Path(config_path).exists():
        return list(DEFAULT_CONFIG["models"].keys())
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return list(config.get("models", {}).keys())
    except Exception:
        return list(DEFAULT_CONFIG["models"].keys())


def call_llm(
    prompt: str,
    model_config: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> Optional[str]:
    """
    调用 LLM API
    
    Args:
        prompt: 输入提示
        model_config: 模型配置字典（优先使用）
        model_name: 模型名称（如果 model_config 为 None 则加载此模型）
        
    Returns:
        LLM 返回的文本，失败返回 None
    """
    # 加载配置
    if model_config is None:
        model_config = load_llm_config(model_name)
    
    provider = model_config.get("name", "Unknown")
    # 支持 apiKey (驼峰) 和 api_key (下划线) 两种格式
    api_key = model_config.get("api_key") or model_config.get("apiKey", "")
    api_base = model_config.get("api_base", "")
    model = model_config.get("model", "")
    temperature = model_config.get("temperature", 0.3)
    max_tokens = model_config.get("max_tokens", 1024)
    timeout = model_config.get("timeout", 30)
    
    # 检查必要配置
    if not api_key and provider.lower() != "ollama":
        print(f"❌ {provider} API key 未配置")
        return None
    
    # 构建请求
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # 不同 provider 的 API 格式可能不同
    if provider.lower() == "ollama":
        # Ollama 格式
        url = f"{api_base}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
    elif "minimaxi" in api_base:
        # MiniMax 格式
        url = f"{api_base}/text/chatcompletion_v2"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    else:
        # OpenAI 兼容格式
        url = f"{api_base}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        
        data = resp.json()
        
        # 解析响应（不同 provider 格式不同）
        if provider.lower() == "ollama":
            return data.get("response", "")
        else:
            # MiniMax / OpenAI 格式
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "")
        
        return None
        
    except Exception as e:
        print(f"❌ {provider} API 调用失败: {e}")
        return None


# 向后兼容的 MiniMax 专用函数
def call_minimax_llm(
    prompt: str,
    model: str = "MiniMax-M2.7",
    temperature: float = 0.3,
    max_tokens: int = 1024
) -> Optional[str]:
    """
    调用 MiniMax LLM API（向后兼容）
    
    从配置文件读取配置，但允许覆盖部分参数
    """
    model_config = load_llm_config("minimax")
    if model:
        model_config["model"] = model
    if temperature is not None:
        model_config["temperature"] = temperature
    if max_tokens:
        model_config["max_tokens"] = max_tokens
    
    return call_llm(prompt, model_config=model_config)


def format_turn_for_judge(messages: List[Dict[str, Any]]) -> str:
    """
    将对话消息格式化为适合 LLM 判断的文本
    
    Args:
        messages: 对话消息列表
        
    Returns:
        格式化后的文本
    """
    lines = []
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        name = msg.get('name', '')
        
        if not content:
            continue
        
        # 截断过长的 content
        if len(content) > 500:
            content = content[:500] + "...(truncated)"
        
        if role == 'user':
            lines.append(f"[用户]: {content}")
        elif role == 'assistant':
            tool_calls = msg.get('tool_calls', [])
            if tool_calls:
                # 只显示工具调用摘要
                tool_names = [tc.get('function', {}).get('name', 'unknown') for tc in tool_calls]
                lines.append(f"[AI]: (调用工具: {', '.join(tool_names)})")
                lines.append(f"[AI回复]: {content if content else '(等待工具结果)'}")
            else:
                lines.append(f"[AI]: {content}")
        elif role == 'tool':
            # tool 消息显示 name 和 content 摘要
            tool_name = name or msg.get('function', {}).get('name', 'unknown')
            content_preview = content[:200] + "..." if len(content) > 200 else content
            lines.append(f"[工具-{tool_name}]: {content_preview}")
    
    return "\n".join(lines)


def get_available_skills() -> List[Dict[str, str]]:
    """获取可用 skill 列表，按创建时间倒序（新的在前）"""
    skills_dir = Path("/Users/rama/.nanobot/workspace/skills")
    skills = []
    
    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith("__"):
            continue
        
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        
        # 获取修改时间
        mtime = skill_dir.stat().st_mtime
        
        # 读取描述（第一行非空行）
        try:
            content = skill_md.read_text(encoding='utf-8')
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            desc = lines[0] if lines else "无描述"
            if desc.startswith('#'):
                desc = lines[1] if len(lines) > 1 else "无描述"
        except:
            desc = "无描述"
        
        skills.append({
            "name": skill_dir.name,
            "desc": desc[:80] + "..." if len(desc) > 80 else desc,
            "mtime": mtime
        })
    
    # 按时间倒序
    skills.sort(key=lambda x: x["mtime"], reverse=True)
    return skills


def format_skills_for_prompt() -> str:
    """格式化 skill 列表用于 prompt"""
    skills = get_available_skills()
    lines = ["| Skill | 描述 | 新旧 |", "|-------|------|------|"]
    
    now = time.time()
    for i, s in enumerate(skills[:20]):  # 只取前20个
        age = "新" if (now - s["mtime"]) < 7 * 86400 else "旧"
        if i < 5:
            age = "**新**"
        lines.append(f"| {s['name']} | {s['desc']} | {age} |")
    
    return "\n".join(lines)


JUDGE_PROMPT_TEMPLATE = """你是一个 AI 助手质量检查员。分析以下对话，严格犀利判断各 skill 的执行状态。

## 当前可用 Skills（按创建时间排序，新的优先）

{available_skills}

## 判断标准

1. **success**: skill 正确执行，返回了有用的结果，用户需求得到满足
2. **failed**: skill 执行报错、返回错误、或没有返回有用结果
3. **partial**: skill 部分成功，但结果不完整或需要补充
4. **unknown**: 无法判断（可能是非 skill 工具调用）

## Skill 选择检查

对比"当前可用 Skills"表格，检查是否使用了次优 skill：
- **记忆/回忆类意图**（记得、之前、找一下）→ 应优先使用较新的 `memora-query`，而非 `grep`/`exec` 手动搜索
- **新 skill 未使用** → 如果对话中使用了旧 skill，但存在更新的、更专门的 skill 可用 → 标记 better_choice
- **available=false 的 skill** → 不应被调用

## 输出格式

请严格按以下 JSON 格式输出，不要有任何额外文字：

```json
{{
  "skill_results": [
    {{"skill": "skill-name-1", "status": "success/failed/partial/unknown", "reason": "简短原因", "better_choice": "应使用的更好skill名(如有)"}},
    {{"skill": "skill-name-2", "status": "success/failed/partial/unknown", "reason": "简短原因", "better_choice": "应使用的更好skill名(如有)"}}
  ],
  "overall_quality": "good/poor",
  "summary": "一句话总结"
}}
```

## 对话内容

{dialogue}
"""


def judge_skill_quality(
    messages: List[Dict[str, Any]],
    model_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    使用 LLM 判断对话中各 skill 的执行质量
    
    Args:
        messages: 对话消息列表
        model_name: 指定使用的模型名称，None 则使用默认模型
        
    Returns:
        判断结果字典，包含 skill_results, overall_quality, summary
        判断失败返回 None
    """
    if not messages:
        return None
    
    # 加载模型配置
    model_config = load_llm_config(model_name)
    provider = model_config.get("name", "Unknown")
    
    # 格式化对话
    dialogue = format_turn_for_judge(messages)
    
    # 构建 prompt
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        dialogue=dialogue,
        available_skills=format_skills_for_prompt()
    )
    
    # 调用 LLM
    print(f"🔍 使用 {provider} 进行 skill 质检...")
    response = call_llm(prompt, model_config=model_config)
    
    if not response:
        print("⚠️ LLM 判断失败，跳过 skill 质检")
        return None
    
    # 解析 JSON 响应
    try:
        # 尝试提取 JSON（可能在 markdown 代码块中）
        json_str = response.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        json_str = json_str.strip()
        
        result = json.loads(json_str)
        
        # 验证格式
        if "skill_results" in result and "overall_quality" in result:
            return result
        else:
            print(f"⚠️ LLM 返回格式不正确: {result}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"⚠️ 解析 LLM 返回失败: {e}")
        print(f"原始响应: {response[:500]}")
        return None
    except Exception as e:
        print(f"⚠️ 处理 LLM 返回时出错: {e}")
        return None


def format_quality_report(quality_result: Dict[str, Any], detected_skills: List[str]) -> str:
    """
    将质检结果格式化为可读的报告
    
    Args:
        quality_result: judge_skill_quality 返回的结果
        detected_skills: 原本检测到的 skill 列表
        
    Returns:
        格式化的报告文本
    """
    if not quality_result:
        return ""
    
    parts = []
    
    parts.append("## 🔍 Skill 质检报告")
    
    skill_results = quality_result.get("skill_results", [])
    if skill_results:
        for sr in skill_results:
            skill = sr.get("skill", "unknown")
            status = sr.get("status", "unknown")
            reason = sr.get("reason", "")
            
            status_icon = {
                "success": "✅",
                "failed": "❌",
                "partial": "⚠️",
                "unknown": "❓"
            }.get(status, "❓")
            
            parts.append(f"{status_icon} **{skill}**: {status} - {reason}")
    else:
        parts.append("未检测到 skill 执行")
    
    overall = quality_result.get("overall_quality", "unknown")
    summary = quality_result.get("summary", "")
    
    parts.append(f"\n**整体质量**: {'🟢 良好' if overall == 'good' else '🔴 需改进'}")
    if summary:
        parts.append(f"**评价**: {summary}")
    
    return "\n".join(parts)


def save_skill_status(quality_result: Dict[str, Any], dialogue_id: str = ""):
    """
    将质检结果保存到 memora/skill/skill_status.md
    
    Args:
        quality_result: judge_skill_quality 返回的结果
        dialogue_id: 对话标识（可选）
    """
    # 确保目录存在
    skill_dir = Path("/Users/rama/.nanobot/workspace/memora/skill")
    skill_dir.mkdir(parents=True, exist_ok=True)
    
    status_file = skill_dir / "skill_status.md"
    
    # 构建记录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    lines = [f"\n## 质检记录 {timestamp}"]
    if dialogue_id:
        lines.append(f"**对话ID**: {dialogue_id}")
    
    skill_results = quality_result.get("skill_results", [])
    if skill_results:
        lines.append("\n| Skill | 状态 | 原因 | 更优选择 |")
        lines.append("|-------|------|------|----------|")
        for sr in skill_results:
            skill = sr.get("skill", "unknown")
            status = sr.get("status", "unknown")
            reason = sr.get("reason", "")
            better = sr.get("better_choice", "-")
            lines.append(f"| {skill} | {status} | {reason} | {better} |")
    
    overall = quality_result.get("overall_quality", "unknown")
    summary = quality_result.get("summary", "")
    lines.append(f"\n**整体质量**: {overall}")
    if summary:
        lines.append(f"**评价**: {summary}")
    
    lines.append("\n---")
    
    # 追加写入
    with open(status_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return status_file


if __name__ == "__main__":
    import sys
    
    # 测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        # 列出可用模型
        print("可用模型:")
        for model in list_available_models():
            print(f"  - {model}")
        sys.exit(0)
    
    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        # 使用指定模型测试
        test_model = sys.argv[2]
        print(f"测试使用模型: {test_model}")
        
        test_messages = [
            {"role": "user", "content": "记得E2B"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "exec"}}]},
            {"role": "tool", "name": "exec", "content": "Traceback...ModuleNotFoundError..."},
            {"role": "assistant", "content": "找到啦！E2B是..."},
        ]
        
        result = judge_skill_quality(test_messages, model_name=test_model)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0)
    
    # 默认测试（使用配置文件默认模型）
    print(f"可用模型: {list_available_models()}")
    
    test_messages = [
        {"role": "user", "content": "记得E2B"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "exec"}}]},
        {"role": "tool", "name": "exec", "content": "Traceback...ModuleNotFoundError..."},
        {"role": "assistant", "content": "找到啦！E2B是..."},
    ]
    
    result = judge_skill_quality(test_messages)
    print(json.dumps(result, ensure_ascii=False, indent=2))
