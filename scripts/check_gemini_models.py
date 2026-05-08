import os
import sys
import asyncio
import httpx

# 切换到项目根目录，确保 pydantic-settings 能找到 .env 文件
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

try:
    from app.core.config import Settings
    # 重新实例化以确保从正确路径加载 .env
    settings = Settings()
except ImportError:
    print("❌ 无法导入项目配置，请确保在项目根目录下运行。")
    sys.exit(1)

async def check_models():
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        print("❌ 错误: 未在 .env 文件中找到 GEMINI_API_KEY。")
        print("请检查你的 .env 文件是否存在并包含: GEMINI_API_KEY=AIzaSy...")
        return

    print(f"🔍 正在使用 API Key: {api_key[:8]}...{api_key[-4:]} 检查模型列表...")
    
    # 1. 获取模型列表
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(list_url)
            if resp.status_code != 200:
                print(f"❌ 请求失败 (HTTP {resp.status_code}): {resp.text}")
                return
            
            data = resp.json()
            models = data.get("models", [])
            
            # 过滤支持 generateContent 的模型
            gen_models = [m for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]
            
            print(f"\n✅ 成功获取模型列表！共发现 {len(gen_models)} 个支持生成内容的模型：\n")
            print(f"{'Model ID':<45} {'Input Limit':<12} {'Thinking':<10}")
            print("-" * 75)
            
            for m in gen_models:
                name = m.get("name", "").replace("models/", "")
                input_limit = m.get("inputTokenLimit", "N/A")
                thinking = "✅" if m.get("thinking") else "❌"
                print(f"{name:<45} {input_limit:<12} {thinking:<10}")
            
            print("-" * 75)
            
            # 2. 对多个常用模型进行连接测试
            test_models = [
                "gemini-3-flash-preview",
                "gemini-3-pro-preview",
                "gemini-3.1-flash-lite-preview",
                
            ]
            available_names = [m.get("name", "").replace("models/", "") for m in gen_models]
            test_models = [m for m in test_models if m in available_names]
            
            payload = {"contents": [{"parts": [{"text": "Reply with exactly: OK"}]}]}
            
            print(f"\n🚀 正在对 {len(test_models)} 个常用模型进行连接测试...\n")
            for model in test_models:
                test_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                try:
                    test_resp = await client.post(test_url, json=payload)
                    if test_resp.status_code == 200:
                        result_text = test_resp.json()['candidates'][0]['content']['parts'][0]['text']
                        print(f"  ✅ {model:<40} → 可用 (回复: {result_text.strip()[:50]})")
                    elif test_resp.status_code == 429:
                        err = test_resp.json().get("error", {})
                        # 检查是 limit:0（模型不可用）还是真正的频率限制
                        msg = err.get("message", "")
                        if "limit: 0" in msg:
                            print(f"  🚫 {model:<40} → 免费套餐不可用 (quota limit=0)")
                        else:
                            print(f"  ⏳ {model:<40} → 频率限制中，稍后重试")
                    else:
                        print(f"  ❌ {model:<40} → HTTP {test_resp.status_code}")
                except Exception as e:
                    print(f"  ❌ {model:<40} → 错误: {e}")
            
            print(f"\n💡 建议：将 .env 中的 GEMINI_MODELS 设置为上面标记 ✅ 的模型。")

        except httpx.ConnectError:
            print("❌ 无法连接到 Google API。请检查网络或代理设置。")
        except Exception as e:
            print(f"❌ 发生意外错误: {str(e)}")

if __name__ == "__main__":
    asyncio.run(check_models())
