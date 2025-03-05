import os
import sqlparse
from sqlparse.tokens import Keyword, Name
from sqlparse.sql import Identifier, Comparison, Where
import re
from lxml import etree
from typing import List, Tuple, Dict
import sqlglot

def clean_dynamic_sql(sql: str) -> str:
    """清理 MyBatis 动态 SQL，移除不完整参数和无效部分"""
    # 移除未完成的条件（如 "and id ="、"and station_code ="）
    sql = re.sub(r'\b(and|where)\s+(id|station_code|type|name|level|pid|create_by|update_by|longitude|latitude|remarks|status|region|city|district|address|socket_number|signal_id|fsu_id|device_id|repair_status|processor|current_processor|hazard_source|repair_company|repair_person|review_result|order_status|order_number|operator|ownership|construction|is_share|signal_standard_name|alarm_level|unit)\s*=\s*(?![\'"\w])', '', sql, flags=re.IGNORECASE)
    # 修复 BETWEEN 语句（假设缺少 "high" 或 "low"）
    sql = re.sub(r'BETWEEN\s+AND', 'BETWEEN 0 AND 9999', sql, flags=re.IGNORECASE)
    # 移除多余空格并规范化
    sql = ' '.join(sql.split())
    return sql

def parse_dynamic_sql(xml_content: str) -> str:
    """解析 MyBatis 动态 SQL，将动态片段合并为静态 SQL"""
    try:
        root = etree.fromstring(xml_content.encode('utf-8'))
        sql_parts = []
        for elem in root.iter():
            if elem.tag in ['select', 'insert', 'update', 'delete']:
                sql = ''.join(elem.itertext()).strip()
                # 替换动态参数占位符
                sql = re.sub(r'#\{[^}]+\}|$\{[^{}]+\}', '', sql)
                # 移除 MyBatis 动态标签
                sql = re.sub(r'<\w+.*?>.*?</\w+>|\${[^}]+}', '', sql)
                # 清理动态条件
                sql = clean_dynamic_sql(sql)
                sql_parts.append(sql)
        return ' '.join(sql_parts)
    except Exception as e:
        print(f"解析 XML 失败: {e}")
        return ""

def extract_table_aliases(sql: str) -> Dict[str, str]:
    """从 SQL 中提取表名和别名映射，使用 sqlglot 增强解析"""
    aliases = {}
    try:
        # 使用 sqlglot 解析 SQL
        parsed = sqlglot.parse_one(sql)
        for node in parsed.find_all(sqlglot.exp.Table):
            table_name = str(node.name).strip('`')  # 去掉反引号
            alias = str(node.alias or '').strip('`')
            if alias:
                aliases[alias] = table_name
            else:
                aliases[table_name] = table_name
    except Exception as e:
        print(f"提取别名失败: {e}")
        # 回退到 sqlparse 解析
        parsed = sqlparse.parse(sql)[0]
        in_from = False
        for token in parsed.tokens:
            if token.ttype == Keyword and str(token).upper() == 'FROM':
                in_from = True
                continue
            if in_from and isinstance(token, Identifier):
                parts = str(token).split()
                if len(parts) >= 2:  # 处理 "table alias" 格式
                    table = parts[0].strip('`')
                    alias = parts[-1].strip('`') if len(parts) > 1 else table
                    aliases[alias] = table
                elif len(parts) == 1:
                    table = parts[0].strip('`')
                    aliases[table] = table
            elif token.ttype == Keyword and str(token).upper() in ['WHERE', 'GROUP', 'ORDER', 'HAVING']:
                in_from = False
    return aliases

def resolve_alias(field: str, aliases: Dict[str, str]) -> str:
    """将带别名的字段（如 s.station_code）解析为真实表名和列名（如 jy_bms_station.station_code）"""
    for alias, table in aliases.items():
        if field.startswith(f"{alias}."):
            return f"{table}.{field[len(alias) + 1:]}"
    # 如果没有匹配别名，尝试直接解析
    match = re.match(r'(\w+)\.(\w+)', field)
    if match:
        table, column = match.groups()
        return f"{table}.{column}"
    return field

def extract_joins(sql: str) -> List[Tuple[str, str]]:
    """从 SQL 中提取 JOIN 条件中的表和字段关联，处理别名和复杂结构"""
    relations = []
    try:
        # 使用 sqlglot 解析 JOIN 条件
        parsed = sqlglot.parse_one(sql)
        aliases = extract_table_aliases(sql)
        for node in parsed.find_all(sqlglot.exp.Join):
            on_condition = node.on
            if on_condition:
                # 提取 ON 条件中的字段比较
                for expr in on_condition.find_all(sqlglot.exp.EQ):
                    left = str(expr.left).strip()
                    right = str(expr.right).strip()
                    left_resolved = resolve_alias(left, aliases)
                    right_resolved = resolve_alias(right, aliases)
                    # 忽略嵌套函数（如 COALESCE）只提取字段
                    if not (left.startswith('COALESCE') or right.startswith('COALESCE')):
                        left_match = re.match(r'(\w+)\.(\w+)', left_resolved)
                        right_match = re.match(r'(\w+)\.(\w+)', right_resolved)
                        if left_match and right_match:
                            left_table, left_field = left_match.groups()
                            right_table, right_field = right_match.groups()
                            relations.append((f"{left_table}.{left_field}", f"{right_table}.{right_field}"))
        # 回退到 sqlparse 处理 WHERE 条件
        sqlparse_parsed = sqlparse.parse(sql)[0]
        for token in sqlparse_parsed.tokens:
            if isinstance(token, Where):
                for subtoken in token.tokens:
                    if isinstance(subtoken, Comparison):
                        left = str(subtoken.left).strip()
                        right = str(subtoken.right).strip()
                        left_resolved = resolve_alias(left, aliases)
                        right_resolved = resolve_alias(right, aliases)
                        left_match = re.match(r'(\w+)\.(\w+)', left_resolved)
                        right_match = re.match(r'(\w+)\.(\w+)', right_resolved)
                        if left_match and right_match and not (left.startswith('COALESCE') or right.startswith('COALESCE')):
                            left_table, left_field = left_match.groups()
                            right_table, right_field = right_match.groups()
                            relations.append((f"{left_table}.{left_field}", f"{right_table}.{right_field}"))
    except Exception as e:
        print(f"解析 SQL 失败: {e}")
    return relations

def process_mapper_directory(mapper_dir: str) -> dict:
    """处理 Mapper XML 目录，提取所有 SQL 并解析外键关系"""
    all_relations = {}
    for root, _, files in os.walk(mapper_dir):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        xml_content = f.read()
                        sql = parse_dynamic_sql(xml_content)
                        if sql:  # 确保 SQL 不为空
                            relations = extract_joins(sql)
                            for rel in relations:
                                key = f"{rel[0]} → {rel[1]}" if rel[1] else rel[0]
                                all_relations[key] = all_relations.get(key, 0) + 1
                except Exception as e:
                    print(f"处理文件 {file_path} 失败: {e}")
    return all_relations

# 示例使用
if __name__ == "__main__":
    # 指定 Mapper XML 文件或目录路径
    mapper_dir = "D:/workspace/code/java/202412/GanSu-ZhanZhi-Demo/ruoyi-system/src/main/resources/mapper/jinyu/"  # 替换为你的 Mapper XML 目录
    relations = process_mapper_directory(mapper_dir)

    # 输出外键关系
    print("提取的外键关系（基于 Mapper XML 中的 SQL）：")
    for relation, count in relations.items():
        print(f"{relation} （出现 {count} 次）")

    # 格式化外键关系为 Vanna 需要的格式
    formatted_relations = {}
    for relation in relations.keys():
        if " → " in relation:
            source, target = relation.split(" → ")
            formatted_relations[source.strip()] = target.strip()

    print("\n格式化的外键关系（适合 Vanna 训练）：")
    for source, target in formatted_relations.items():
        print(f"{source} → {target}")