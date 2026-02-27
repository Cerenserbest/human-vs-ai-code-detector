import random
import json

OUTPUT = "human_extra_generated.jsonl"

def h_basic_loop(i):
    return f"""
nums = [1,2,3,4,5]
s = 0
for n in nums:
    s = s + n
print("sum:", s)
"""

def h_mistyped_var(i):
    return f"""
value = 10
vale = value * 2
print(vale)
"""

def h_no_docstring(i):
    return f"""
def mul_{i}(a,b):
    return a*b

print(mul_{i}(3,5))
"""

def h_bad_formatting(i):
    return f"""
def f{i}(x):
 print("value:",x)
 return x+1

print(f{i}(10))
"""

def h_missing_error_handling(i):
    return f"""
def safe_div_{i}(a,b):
    return a/b  # no error handling on purpose

print(safe_div_{i}(10,2))
"""


def h_inline_logic(i):
    return f"""
numbers=[3,7,1,9]
mx=numbers[0]
for x in numbers:
    if x>mx:
        mx=x
print(mx)
"""

def h_random_spacing(i):
    return f"""
def add{i}(a,b):
        return   a + b
print( add{i}(2 ,3))
"""

def h_typo_comment(i):
    return f"""
# calculte avg
vals=[4,5,6]
print(sum(vals)/len(vals))
"""

def h_print_debug(i):
    return f"""
def inc{i}(n):
    print("got:",n)
    n=n+1
    print("out:",n)
    return n

inc{i}(5)
"""

def h_inline_condition(i):
    return f"""
n={i}
if n%2==0: print("even")
else: print("odd")
"""

HUMAN_PATTERNS = [
    h_basic_loop,
    h_mistyped_var,
    h_no_docstring,
    h_bad_formatting,
    h_missing_error_handling,
    h_inline_logic,
    h_random_spacing,
    h_typo_comment,
    h_print_debug,
    h_inline_condition,
]

def generate(count=1500):
    data = []
    for i in range(count):
        f = random.choice(HUMAN_PATTERNS)
        code = f(i)
        data.append({
            "id": f"HGEN_{i}",
            "text": code.strip(),
            "label": "0",
            "lang": "python"
        })
    return data

def main():
    rows = generate()
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Yeni human snippet dosyası üretildi:", OUTPUT)

if __name__ == "__main__":
    main()
