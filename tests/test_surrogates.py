"""sanitize_surrogates 的纯逻辑用例 —— 不触网、不碰 SDK。

背景: OpenAI 兼容服务(DashScope/DeepSeek 等)流式输出会把 emoji 的
UTF-16 代理对拆到两个 SSE delta,json.loads 单独解析后留下孤立代理字符,
下一次请求 UTF-8 编码时抛 "surrogates not allowed"。
"""

import json

from mini_claude_code.backends.base import sanitize_surrogates


def test_clean_text_passthrough():
    s = "正常文本 with emoji 😀 和 中文"
    assert sanitize_surrogates(s) is s  # 快路径原样返回


def test_split_pair_rejoined():
    # 模拟两个 SSE delta 分别携带代理对的高低两半
    hi = json.loads('{"c": "\\ud83d"}')["c"]
    lo = json.loads('{"c": "\\ude00"}')["c"]
    assert sanitize_surrogates("你好" + hi + lo + "!") == "你好😀!"


def test_lone_high_surrogate_replaced():
    assert sanitize_surrogates("a\ud83db") == "a�b"


def test_lone_low_surrogate_replaced():
    assert sanitize_surrogates("a\ude00b") == "a�b"


def test_reversed_pair_not_joined():
    # 低+高的顺序不是合法代理对,应各自替换
    assert sanitize_surrogates("\ude00\ud83d") == "��"


def test_multiple_pairs_and_lone_mixed():
    s = "😀 x 🤖 y \ud800"
    assert sanitize_surrogates(s) == "😀 x 🤖 y �"


def test_result_always_utf8_encodable():
    cases = ["\ud800", "\udfff", "😀", "abc\ud83d", "正常"]
    for s in cases:
        sanitize_surrogates(s).encode("utf-8")  # 不应抛异常
