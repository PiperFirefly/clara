#!/usr/bin/env python3
import base64, json, os, sys, urllib.request

AUTH = os.path.expanduser("~/.pi/agent/auth.json")

def keys():
    d = json.load(open(AUTH))
    return d.get("deepseek", {}).get("key", ""), d.get("openai", {}).get("key", "")

def payload(model, path, prompt, max_tokens=700):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

def call(url, p, auth=None):
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    req = urllib.request.Request(url, data=json.dumps(p).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)

def describe_deepseek(path, prompt):
    dk, _ = keys()
    p = payload("deepseek-v4-flash-vision-exp", path, prompt)
    resp = call("https://api.deepseek.com/chat/completions", p, dk)
    m = resp["choices"][0]["message"]
    return m.get("content"), m.get("reasoning_content"), resp.get("usage", {})

def describe_openai(path, prompt):
    _, ok = keys()
    p = payload("gpt-4o-mini", path, prompt)
    resp = call("https://api.openai.com/v1/chat/completions", p, ok)
    return resp["choices"][0]["message"]["content"]

if __name__ == "__main__":
    path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail, naturally."
    print("=" * 70)
    print("DEEPSEEK deepseek-v4-flash-vision-exp")
    print("=" * 70)
    try:
        content, reasoning, usage = describe_deepseek(path, prompt)
        print("[content]:", content if content else "(EMPTY)")
        if reasoning:
            print("[reasoning_content]:", reasoning[:2000])
        print("[usage]:", usage)
    except Exception as e:
        print("DEEPSEEK ERROR:", repr(e))
    print()
    print("=" * 70)
    print("OPENAI gpt-4o-mini (baseline)")
    print("=" * 70)
    try:
        print(describe_openai(path, prompt))
    except Exception as e:
        print("OPENAI ERROR:", repr(e))
