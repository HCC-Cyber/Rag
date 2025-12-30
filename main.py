import os
from embed import query_db, create_db

def answer_question(question: str) -> str:
    # 从环境变量获取API密钥
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 ZHIPUAI_API_KEY")

    if api_key:
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=api_key)
        chunks = query_db(question)
        prompt = "Please answer user's question according to context\n"
        prompt += f"Question: {question}\n"
        prompt += "Context:\n"
        for c in chunks:
            prompt += f"{c}\n"
            prompt += "-------------\n"
        
        print(f"Prompt sent to model:\n{prompt}")
        
        response = client.chat.completions.create(
            model="glm-4.6v-flash",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    else:
        # 如果没有API密钥，返回提示信息
        chunks = query_db(question)
        prompt = "Please answer user's question according to context\n"
        prompt += f"Question: {question}\n"
        prompt += "Context:\n"
        for c in chunks:
            prompt += f"{c}\n"
            prompt += "-------------\n"
        
        print(f"Prompt sent to model:\n{prompt}")
        return "由于未设置API密钥，此为模拟响应。请设置ZHIPUAI_API_KEY环境变量。"

if __name__ == '__main__':
    # 示例问题
    question = "令狐冲领悟了什么魔法？"
    answer = answer_question(question)
    print(f"\nModel response:\n{answer}")