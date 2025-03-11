from typing import Dict, Any, List, Tuple
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import StructuredTool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel

app = Flask(__name__)
CORS(app)
# 修改现有的CORS配置
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3002"}})  # 允许React默认端口

@app.route('/api/generate', methods=['POST'])
def handle_generate():
    try:
        data = request.json
        print("接收请求数据:", data)  # 新增调试日志
        
        # 添加参数验证
        if not all(key in data for key in ["character1", "character2", "topic"]):
            return jsonify({"status": "error", "message": "缺少必要参数"}), 400
            
        result = full_processing_chain.invoke({
            "character1": data["character1"],
            "character2": data["character2"],
            "topic": data["topic"],
            "scene_id": "default"
        })
        print("处理结果:", result)  # 新增调试日志
        
        return jsonify({
            "status": "success",
            "dialogue": result["raw_dialogue"],
            "characters": result["registered_chars"]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印完整错误堆栈
        return jsonify({"status": "error", "message": str(e)}), 500
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

load_dotenv()

# 初始化模型
llm = ChatDeepSeek(temperature=0.7, model="deepseek-reasoner")

class CharacterData(BaseModel):
    name: str
    profile: str
    background: str

class DialogueRequest(BaseModel):
    character1: CharacterData
    character2: CharacterData
    topic: str
# 增强版角色设定类
class CharacterProfile:
    def __init__(self, name: str, personality: str, background: str):
        self.name = name.strip()
        self.personality = personality
        self.background = background
        self.emotional_state = "neutral"
        self.relationships: Dict[str, str] = {}
        self.scene_context: str = ""


# 虚拟社区数据库
character_db: Dict[str, CharacterProfile] = {}
scene_context_db: Dict[str, str] = {}
branch_points: List[Dict] = []

# 记忆系统
# 修改内存系统初始化方式
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    chat_memory=ChatMessageHistory()  # 新增 chat_memory 参数
    # 移除已废弃的 return_messages 参数
)


# 核心工具 --------------------------------------------------
def normalize_name(name: str) -> str:
    """名称标准化处理"""
    return name.strip().replace(" ", "").replace("　", "")


def parse_character_setting(input_str: str) -> Dict[str, str]:
    """带冗余处理的角色解析"""
    try:
        input_str = input_str.replace("【", "[").replace("】", "]").strip()
        metadata = {}
        current_key = None

        for line in input_str.split("\n"):
            line = line.strip()
            if line.startswith("[") and "]" in line:
                current_key, value = line.split("]", 1)
                current_key = current_key[1:].strip()
                metadata[current_key] = value.split("：", 1)[-1].strip()
            elif current_key and line:
                metadata[current_key] += " " + line
        return metadata
    except Exception as e:
        return {"error": f"解析失败: {str(e)}"}


def register_character(input_str: str) -> str:
    """增强版角色注册"""
    try:
        data = parse_character_setting(input_str)
        if "error" in data:
            return data["error"]

        if "姓名" not in data:
            return "缺失必要字段：姓名"

        # 名称标准化
        char_name = normalize_name(data["姓名"])
        if char_name in character_db:
            return f"角色 {char_name} 已存在"

        profile = CharacterProfile(
            name=char_name,
            personality=data.get("性格", "未定义"),
            background=data.get("背景", "未定义")
        )
        character_db[char_name] = profile
        memory.save_context(
            {"input": f"注册角色 {char_name}"},
            {"output": json.dumps({
                "name": char_name,
                "personality": profile.personality[:20] + "..." if len(
                    profile.personality) > 20 else profile.personality
            })}
        )
        return f"角色 {char_name} 注册成功（情绪：{profile.emotional_state}）"
    except Exception as e:
        return f"注册失败: {str(e)}"


# 修改generate_dialogue参数定义
def generate_dialogue(character1: CharacterData, character2: CharacterData, topic: str) -> str:
    """带验证的对话生成"""
    try:
        # 将参数重组为字典
        character_data = {
            'character1': character1.dict(),
            'character2': character2.dict(),
            'topic': topic
        }

        # 保持原有注册逻辑
        def register_if_needed(char_info):
            char_name = normalize_name(char_info["name"])
            if char_name not in character_db:
                input_str = f"[姓名]：{char_info['name']}\n[性格]：{char_info.get('profile','')}\n[背景]：{char_info.get('background','')}"
                result = register_character(input_str)
                if "失败" in result:
                    raise ValueError(result)
            return char_name

        # 修改参数解构方式
        char1 = register_if_needed(character_data['character1'])
        char2 = register_if_needed(character_data['character2'])
        topic = character_data['topic']

        # 获取角色上下文
        char1_profile = character_db[char1]
        char2_profile = character_db[char2]

        prompt = f"""生成角色对话（标注说话人）：

        角色A：{char1_profile.name}
        性格：{char1_profile.personality}
        背景：{char1_profile.background[:50]}

        角色B：{char2_profile.name}
        性格：{char2_profile.personality}
        背景：{char2_profile.background[:50]}

        话题：{topic}

        要求：
        1. 每人15轮对话，写出完整对话内容，不要精简
        2. 包含情感变化曲线
        3. 体现角色背景特征
        4. 添加2个场景互动描述"""

        response = llm.invoke(prompt).content
        memory.save_context(
            {"input": f"{char1}与{char2}的对话"},
            {"output": response[:200] + "..." if len(response) > 200 else response}
        )
        return response
    except Exception as e:
        return f"对话生成失败: {str(e)}"


# 工具注册
# 确保模型定义正确
class RegisterRequest(BaseModel):
    input_str: str

# 修改工具注册列表
tools = [
    StructuredTool.from_function(
        func=register_character,
        args_schema=RegisterRequest,
        name="RegisterCharacter",
        description="注册新角色"
    ),
    StructuredTool.from_function(
        func=generate_dialogue,
        args_schema=DialogueRequest,
        name="GenerateDialogue",
        description="生成角色对话"
    )
]

# 处理链 --------------------------------------------------
full_processing_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {
    "raw_dialogue": tools[1].run({
        "character1": x["character1"],
        "character2": x["character2"],
        "topic": x["topic"]
    }),
    "scene_context": scene_context_db.get(x["scene_id"], "")
})
        | RunnableLambda(lambda x: {
    **x,
    "registered_chars": list(character_db.keys())  # 添加注册角色查看
})
)

# 用户输入处理
def get_user_input(prompt):
    return input(prompt)


def main():
    # 初始化测试环境
    character_db.clear()
    scene_context_db.clear()

    # 测试角色注册
    for i in range(1, 3):
        print(f"\n=== 注册角色 {i} ===")
        name = get_user_input("请输入角色的姓名：")
        personality = get_user_input("请输入角色的性格：")
        background = get_user_input("请输入角色的背景：")

        input_str = f"[姓名]：{name}\n[性格]：{personality}\n[背景]：{background}"
        print(tools[0].run(input_str))


    # 执行对话生成
    try:
        character1 = get_user_input("\n请输入第一个角色的姓名：")
        character2 = get_user_input("请输入第二个角色的姓名：")
        topic = get_user_input("请输入对话的主题：")

        result = full_processing_chain.invoke({
            "character1": character1,
            "character2": character2,
            "topic": topic,
            "scene_id": "default"
        })

        print("\n=== 生成结果 ===")
        print("注册角色:", result["registered_chars"])
        print("原始对话:\n", result["raw_dialogue"])

    except Exception as e:
        print(f"流程执行失败: {str(e)}")


if __name__ == "__main__":
    main()
