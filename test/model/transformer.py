from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载预训练模型（例如 T5）
model_name = "D:\workspace\code\model\RASAT-Small"  # 替换为实际的 RASAT 模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

s3_prompt = """
表结构信息：
products(id, name, price)
sales(product_id, quantity, total_amount)

表关系信息：
sales.product_id -> products.id

示例查询：
示例 1：
问题：查询每个客户的订单总数。
SQL 查询：
SELECT T1.name, COUNT(T2.id) AS order_count
FROM customers AS T1
JOIN orders AS T2 ON T1.id = T2.customer_id
GROUP BY T1.name;

示例 2：
问题：查询每个产品的销售总额。
SQL 查询：
SELECT T1.name, SUM(T2.total_amount) AS total_sales
FROM products AS T1
JOIN sales AS T2 ON T1.id = T2.product_id
GROUP BY T1.name;

用户问题：
问题：查询每个产品的名称和总销售额。

请根据以上信息生成 SQL 查询：
"""

# 将提示编码为模型输入
inputs = tokenizer(s3_prompt, return_tensors="pt")

# 生成 SQL 查询
outputs = model.generate(**inputs, max_length=200)
sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"生成的 SQL 查询：{sql_query}")