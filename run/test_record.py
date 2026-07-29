#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""香橙派 → 电脑 命令测试（HTTP）"""
import json
import time
import urllib.request

PC = "http://BAMBOO.local:8080"   # 连不上就换 http://192.168.x.x:8080


def send(cmd):
    """发送命令，返回电脑端的 JSON 回复"""
    url = f"{PC}/cmd/{cmd}"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
            print(f"  → {cmd:5s}  回复: {data}")
            return data
    except Exception as e:
        print(f"  → {cmd:5s}  失败: {e}")
        return None


# ── 方式一：交互式测试（按回车发命令）──
def interactive():
    print("=" * 44)
    print("  命令测试（电脑端需已启动服务）")
    print("  输入 start / end / q 退出")
    print("=" * 44)
    while True:
        cmd = input("\n命令 > ").strip().lower()
        if cmd in ("q", "quit", "exit"):
            break
        if cmd in ("start", "end"):
            send(cmd)
        else:
            print("  只支持 start / end")


# ── 方式二：自动测试（start → 等 5 秒 → end）──
def auto():
    print("[自动测试] 发送 start …")
    send("start")
    print("[自动测试] 等待 5 秒（模拟录制区间）…")
    time.sleep(5)
    print("[自动测试] 发送 end …")
    send("end")
    print("[自动测试] 完成！去电脑网页看时间轴上的绿/红标记。")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "auto":
        auto()
    else:
        interactive()