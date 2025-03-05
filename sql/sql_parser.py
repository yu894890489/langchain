from sql.sql_foreign_key import extract_joins

with open("D:/work/345/所有的查询sql.txt",encoding="utf-8") as file:
    readlines = file.readlines()
    index = 0
    for line in readlines:
        print(line)
        print(index)
        relations = extract_joins(line)
        index +=1
        print("*"*50)
        for rel in relations:
            key = f"{rel[0]} → {rel[1]}" if rel[1] else rel[0]
            print(key)