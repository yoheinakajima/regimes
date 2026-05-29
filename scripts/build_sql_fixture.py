"""Build fixtures/synthetic_sql.json — a deterministic, committed
fixture for the SQL target.

Mirrors `scripts/build_fixture.py` for LongMemEval: hand-crafted
content, written through a stable serialization (sort_keys=False,
indent=2, trailing newline) so re-running produces a byte-identical
file. Seed=42 in spirit — there's no randomness here; the determinism
comes from the script itself being the single source of truth.

The fixture has 30 questions across 3 small schemas:

  customers_orders   — 3 tables (customers, products, orders)
  employees_depts    — 4 tables (departments, employees, projects, salaries)
  categories_items   — 2 tables (categories, items)

Each question carries:
  - the natural-language question + question_type
  - the schema DDL + a small seed-rows block
  - the gold SQL + the gold result-set (so the executor can verify)
  - default_wrong_sql + unlock_phrase (used by FakeSqlReader in mock
    mode — the unlock_phrase is what a prompt-transform must inject
    into the prompt to flip the question to correct)

Headroom: ~40% of questions ship with default_wrong_sql that exhibits
one of the seam-reachable regimes (schema-misunderstanding,
wrong-aggregation, wrong-join, wrong-filter), so the baseline
accuracy lands in the 50-75% target band and the loop has real
regimes to act on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "fixtures" / "synthetic_sql.json"


# ===========================================================================
# Schema 1: customers / products / orders
# ===========================================================================

SCHEMA1_DDL = (
    "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT);\n"
    "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL);\n"
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, "
    "product_id INTEGER, quantity INTEGER, order_date TEXT);"
)

SCHEMA1_SEEDS = [
    "INSERT INTO customers VALUES (1, 'Alice', 'USA');",
    "INSERT INTO customers VALUES (2, 'Bob', 'USA');",
    "INSERT INTO customers VALUES (3, 'Cara', 'Canada');",
    "INSERT INTO customers VALUES (4, 'Dan', 'UK');",
    "INSERT INTO products VALUES (1, 'Widget', 10.0);",
    "INSERT INTO products VALUES (2, 'Gadget', 25.0);",
    "INSERT INTO products VALUES (3, 'Sprocket', 5.0);",
    "INSERT INTO orders VALUES (1, 1, 1, 3, '2024-01-15');",
    "INSERT INTO orders VALUES (2, 1, 2, 1, '2024-02-10');",
    "INSERT INTO orders VALUES (3, 2, 1, 2, '2024-01-20');",
    "INSERT INTO orders VALUES (4, 3, 3, 5, '2024-03-05');",
    "INSERT INTO orders VALUES (5, 4, 2, 1, '2024-03-12');",
    "INSERT INTO orders VALUES (6, 1, 3, 4, '2024-04-01');",
]

SCHEMA1_META = {
    "schema_id": "customers_orders",
    "schema_ddl": SCHEMA1_DDL,
    "seed_rows": SCHEMA1_SEEDS,
    "tables": ["customers", "products", "orders"],
    "columns_by_table": {
        "customers": ["id", "name", "country"],
        "products": ["id", "name", "price"],
        "orders": ["id", "customer_id", "product_id", "quantity", "order_date"],
    },
    "primary_keys": {"customers": "id", "products": "id", "orders": "id"},
    "foreign_keys": [
        ["orders", "customer_id", "customers", "id"],
        ["orders", "product_id", "products", "id"],
    ],
}


SCHEMA1_QUESTIONS = [
    # 1. simple-select, correct on baseline
    dict(
        qid="sql_co_q00",
        qtype="simple-select",
        question="List the names of all customers from the USA.",
        gold_sql="SELECT name FROM customers WHERE country = 'USA';",
        default_wrong_sql="SELECT name FROM customers WHERE country = 'USA';",
        unlock_phrase="",
    ),
    # 2. simple-select, schema-misunderstanding default-wrong
    dict(
        qid="sql_co_q01",
        qtype="simple-select",
        question="List all product names.",
        gold_sql="SELECT name FROM products;",
        # invent a column 'title' that doesn't exist → schema-misunderstanding
        default_wrong_sql="SELECT title FROM products;",
        unlock_phrase="Use exactly the columns listed in the schema.",
    ),
    # 3. simple-select, correct on baseline
    dict(
        qid="sql_co_q02",
        qtype="simple-select",
        question="What are the customer IDs of customers from Canada?",
        gold_sql="SELECT id FROM customers WHERE country = 'Canada';",
        default_wrong_sql="SELECT id FROM customers WHERE country = 'Canada';",
        unlock_phrase="",
    ),
    # 4. join, correct on baseline
    dict(
        qid="sql_co_q03",
        qtype="join",
        question="Which customer names placed orders for product id 1?",
        gold_sql=(
            "SELECT customers.name FROM customers "
            "JOIN orders ON customers.id = orders.customer_id "
            "WHERE orders.product_id = 1;"
        ),
        default_wrong_sql=(
            "SELECT customers.name FROM customers "
            "JOIN orders ON customers.id = orders.customer_id "
            "WHERE orders.product_id = 1;"
        ),
        unlock_phrase="",
    ),
    # 5. join, wrong-join default (missing JOIN)
    dict(
        qid="sql_co_q04",
        qtype="join",
        question="Which product names were ordered by Alice?",
        gold_sql=(
            "SELECT products.name FROM products "
            "JOIN orders ON products.id = orders.product_id "
            "JOIN customers ON customers.id = orders.customer_id "
            "WHERE customers.name = 'Alice';"
        ),
        # Missing JOIN — refers only to products → wrong-join
        default_wrong_sql="SELECT name FROM products WHERE id = 1;",
        unlock_phrase="Join across tables using their declared foreign keys.",
    ),
    # 6. join, correct on baseline
    dict(
        qid="sql_co_q05",
        qtype="join",
        question="What product names did customer id 3 order?",
        gold_sql=(
            "SELECT products.name FROM products "
            "JOIN orders ON products.id = orders.product_id "
            "WHERE orders.customer_id = 3;"
        ),
        default_wrong_sql=(
            "SELECT products.name FROM products "
            "JOIN orders ON products.id = orders.product_id "
            "WHERE orders.customer_id = 3;"
        ),
        unlock_phrase="",
    ),
    # 7. aggregate-with-groupby, wrong-aggregation default (no GROUP BY)
    dict(
        qid="sql_co_q06",
        qtype="aggregate-with-groupby",
        question="How many orders has each customer placed?",
        gold_sql=(
            "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id;"
        ),
        # Missing GROUP BY → returns one row → wrong-aggregation
        default_wrong_sql="SELECT customer_id, COUNT(*) FROM orders;",
        unlock_phrase="When aggregating multiple groups, include GROUP BY.",
    ),
    # 8. aggregate-with-groupby, correct on baseline
    dict(
        qid="sql_co_q07",
        qtype="aggregate-with-groupby",
        question="What is the total quantity ordered per product id?",
        gold_sql=(
            "SELECT product_id, SUM(quantity) FROM orders GROUP BY product_id;"
        ),
        default_wrong_sql=(
            "SELECT product_id, SUM(quantity) FROM orders GROUP BY product_id;"
        ),
        unlock_phrase="",
    ),
    # 9. filter-with-having, wrong-filter default (missing WHERE on date)
    dict(
        qid="sql_co_q08",
        qtype="filter-with-having",
        question=(
            "Which customer ids placed more than 1 order on or after 2024-03-01? "
            "Report the customer_id."
        ),
        gold_sql=(
            "SELECT customer_id FROM orders WHERE order_date >= '2024-03-01' "
            "GROUP BY customer_id HAVING COUNT(*) > 1;"
        ),
        # Missing WHERE → wrong-filter
        default_wrong_sql=(
            "SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 1;"
        ),
        unlock_phrase="Apply WHERE clauses to filter rows when the question constrains values.",
    ),
    # 10. filter-with-having, correct on baseline
    dict(
        qid="sql_co_q09",
        qtype="filter-with-having",
        question="Which product ids appear in orders summing more than 4 in total quantity?",
        gold_sql=(
            "SELECT product_id FROM orders GROUP BY product_id HAVING SUM(quantity) > 4;"
        ),
        default_wrong_sql=(
            "SELECT product_id FROM orders GROUP BY product_id HAVING SUM(quantity) > 4;"
        ),
        unlock_phrase="",
    ),
]


# ===========================================================================
# Schema 2: departments / employees / projects / salaries
# ===========================================================================

SCHEMA2_DDL = (
    "CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT);\n"
    "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, "
    "dept_id INTEGER, hire_year INTEGER);\n"
    "CREATE TABLE projects (id INTEGER PRIMARY KEY, title TEXT, lead_id INTEGER);\n"
    "CREATE TABLE salaries (employee_id INTEGER, year INTEGER, amount REAL);"
)

SCHEMA2_SEEDS = [
    "INSERT INTO departments VALUES (1, 'Engineering');",
    "INSERT INTO departments VALUES (2, 'Sales');",
    "INSERT INTO departments VALUES (3, 'Marketing');",
    "INSERT INTO employees VALUES (1, 'Eve', 1, 2020);",
    "INSERT INTO employees VALUES (2, 'Frank', 1, 2021);",
    "INSERT INTO employees VALUES (3, 'Gina', 2, 2019);",
    "INSERT INTO employees VALUES (4, 'Hugo', 2, 2022);",
    "INSERT INTO employees VALUES (5, 'Ivy', 3, 2021);",
    "INSERT INTO projects VALUES (1, 'Atlas', 1);",
    "INSERT INTO projects VALUES (2, 'Beacon', 3);",
    "INSERT INTO projects VALUES (3, 'Cobalt', 1);",
    "INSERT INTO salaries VALUES (1, 2023, 100000);",
    "INSERT INTO salaries VALUES (2, 2023, 90000);",
    "INSERT INTO salaries VALUES (3, 2023, 80000);",
    "INSERT INTO salaries VALUES (4, 2023, 70000);",
    "INSERT INTO salaries VALUES (5, 2023, 85000);",
]

SCHEMA2_META = {
    "schema_id": "employees_depts",
    "schema_ddl": SCHEMA2_DDL,
    "seed_rows": SCHEMA2_SEEDS,
    "tables": ["departments", "employees", "projects", "salaries"],
    "columns_by_table": {
        "departments": ["id", "name"],
        "employees": ["id", "name", "dept_id", "hire_year"],
        "projects": ["id", "title", "lead_id"],
        "salaries": ["employee_id", "year", "amount"],
    },
    "primary_keys": {
        "departments": "id", "employees": "id", "projects": "id",
    },
    "foreign_keys": [
        ["employees", "dept_id", "departments", "id"],
        ["projects", "lead_id", "employees", "id"],
        ["salaries", "employee_id", "employees", "id"],
    ],
}


SCHEMA2_QUESTIONS = [
    # 11. simple-select, correct
    dict(
        qid="sql_ed_q00",
        qtype="simple-select",
        question="List the names of all departments.",
        gold_sql="SELECT name FROM departments;",
        default_wrong_sql="SELECT name FROM departments;",
        unlock_phrase="",
    ),
    # 12. simple-select, schema-misunderstanding (wrong column)
    dict(
        qid="sql_ed_q01",
        qtype="simple-select",
        question="What are the titles of all projects?",
        gold_sql="SELECT title FROM projects;",
        # Pluralize 'titles' → schema-misunderstanding
        default_wrong_sql="SELECT titles FROM projects;",
        unlock_phrase="Use exactly the columns listed in the schema.",
    ),
    # 13. join, correct
    dict(
        qid="sql_ed_q02",
        qtype="join",
        question="What employee name leads the Atlas project?",
        gold_sql=(
            "SELECT employees.name FROM employees "
            "JOIN projects ON employees.id = projects.lead_id "
            "WHERE projects.title = 'Atlas';"
        ),
        default_wrong_sql=(
            "SELECT employees.name FROM employees "
            "JOIN projects ON employees.id = projects.lead_id "
            "WHERE projects.title = 'Atlas';"
        ),
        unlock_phrase="",
    ),
    # 14. join, wrong-join (missing JOIN)
    dict(
        qid="sql_ed_q03",
        qtype="join",
        question="Which employee names work in the Engineering department?",
        gold_sql=(
            "SELECT employees.name FROM employees "
            "JOIN departments ON employees.dept_id = departments.id "
            "WHERE departments.name = 'Engineering';"
        ),
        # No JOIN → wrong-join
        default_wrong_sql="SELECT name FROM employees WHERE dept_id = 1;",
        unlock_phrase="Join across tables using their declared foreign keys.",
    ),
    # 15. join, schema-misunderstanding (wrong table)
    dict(
        qid="sql_ed_q04",
        qtype="join",
        question="List the names of employees and the title of the project they lead.",
        gold_sql=(
            "SELECT employees.name, projects.title FROM employees "
            "JOIN projects ON projects.lead_id = employees.id;"
        ),
        # Use a table 'leaders' that doesn't exist
        default_wrong_sql=(
            "SELECT employees.name, leaders.title FROM employees "
            "JOIN leaders ON leaders.lead_id = employees.id;"
        ),
        unlock_phrase="Use exactly the columns listed in the schema.",
    ),
    # 16. aggregate-with-groupby, correct
    dict(
        qid="sql_ed_q05",
        qtype="aggregate-with-groupby",
        question="How many employees are in each department id?",
        gold_sql="SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id;",
        default_wrong_sql="SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id;",
        unlock_phrase="",
    ),
    # 17. aggregate-with-groupby, wrong-aggregation (no GROUP BY)
    dict(
        qid="sql_ed_q06",
        qtype="aggregate-with-groupby",
        question="What is the total 2023 salary by employee id?",
        gold_sql=(
            "SELECT employee_id, SUM(amount) FROM salaries "
            "WHERE year = 2023 GROUP BY employee_id;"
        ),
        # No GROUP BY → wrong-aggregation
        default_wrong_sql="SELECT employee_id, SUM(amount) FROM salaries WHERE year = 2023;",
        unlock_phrase="When aggregating multiple groups, include GROUP BY.",
    ),
    # 18. aggregate-with-groupby, correct
    dict(
        qid="sql_ed_q07",
        qtype="aggregate-with-groupby",
        question="What is the count of projects per lead_id?",
        gold_sql="SELECT lead_id, COUNT(*) FROM projects GROUP BY lead_id;",
        default_wrong_sql="SELECT lead_id, COUNT(*) FROM projects GROUP BY lead_id;",
        unlock_phrase="",
    ),
    # 19. filter-with-having, wrong-filter (missing WHERE year)
    dict(
        qid="sql_ed_q08",
        qtype="filter-with-having",
        question=(
            "Which 2023 employee_ids earned at least 85000? "
            "(report just the employee_id)"
        ),
        gold_sql=(
            "SELECT employee_id FROM salaries WHERE year = 2023 AND amount >= 85000;"
        ),
        # Missing year filter → wrong-filter
        default_wrong_sql="SELECT employee_id FROM salaries WHERE amount >= 85000;",
        unlock_phrase="Apply WHERE clauses to filter rows when the question constrains values.",
    ),
    # 20. filter-with-having, correct
    dict(
        qid="sql_ed_q09",
        qtype="filter-with-having",
        question="Which dept_ids have more than one employee?",
        gold_sql=(
            "SELECT dept_id FROM employees GROUP BY dept_id HAVING COUNT(*) > 1;"
        ),
        default_wrong_sql=(
            "SELECT dept_id FROM employees GROUP BY dept_id HAVING COUNT(*) > 1;"
        ),
        unlock_phrase="",
    ),
]


# ===========================================================================
# Schema 3: categories / items
# ===========================================================================

SCHEMA3_DDL = (
    "CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT);\n"
    "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, "
    "category_id INTEGER, stock INTEGER);"
)

SCHEMA3_SEEDS = [
    "INSERT INTO categories VALUES (1, 'Electronics');",
    "INSERT INTO categories VALUES (2, 'Books');",
    "INSERT INTO categories VALUES (3, 'Clothing');",
    "INSERT INTO items VALUES (1, 'Laptop', 1, 5);",
    "INSERT INTO items VALUES (2, 'Phone', 1, 12);",
    "INSERT INTO items VALUES (3, 'Headphones', 1, 0);",
    "INSERT INTO items VALUES (4, 'Novel', 2, 30);",
    "INSERT INTO items VALUES (5, 'Cookbook', 2, 8);",
    "INSERT INTO items VALUES (6, 'T-shirt', 3, 15);",
    "INSERT INTO items VALUES (7, 'Jeans', 3, 2);",
]

SCHEMA3_META = {
    "schema_id": "categories_items",
    "schema_ddl": SCHEMA3_DDL,
    "seed_rows": SCHEMA3_SEEDS,
    "tables": ["categories", "items"],
    "columns_by_table": {
        "categories": ["id", "name"],
        "items": ["id", "name", "category_id", "stock"],
    },
    "primary_keys": {"categories": "id", "items": "id"},
    "foreign_keys": [
        ["items", "category_id", "categories", "id"],
    ],
}


SCHEMA3_QUESTIONS = [
    # 21. simple-select, correct
    dict(
        qid="sql_ci_q00",
        qtype="simple-select",
        question="List the names of all categories.",
        gold_sql="SELECT name FROM categories;",
        default_wrong_sql="SELECT name FROM categories;",
        unlock_phrase="",
    ),
    # 22. simple-select, correct
    dict(
        qid="sql_ci_q01",
        qtype="simple-select",
        question="Which item names have stock equal to 0?",
        gold_sql="SELECT name FROM items WHERE stock = 0;",
        default_wrong_sql="SELECT name FROM items WHERE stock = 0;",
        unlock_phrase="",
    ),
    # 23. simple-select, schema-misunderstanding (column 'quantity' doesn't exist)
    dict(
        qid="sql_ci_q02",
        qtype="simple-select",
        question="What are all item names and their current stock?",
        gold_sql="SELECT name, stock FROM items;",
        default_wrong_sql="SELECT name, quantity FROM items;",
        unlock_phrase="Use exactly the columns listed in the schema.",
    ),
    # 24. join, correct
    dict(
        qid="sql_ci_q03",
        qtype="join",
        question="Which item names belong to the Electronics category?",
        gold_sql=(
            "SELECT items.name FROM items "
            "JOIN categories ON items.category_id = categories.id "
            "WHERE categories.name = 'Electronics';"
        ),
        default_wrong_sql=(
            "SELECT items.name FROM items "
            "JOIN categories ON items.category_id = categories.id "
            "WHERE categories.name = 'Electronics';"
        ),
        unlock_phrase="",
    ),
    # 25. join, wrong-join (missing JOIN)
    dict(
        qid="sql_ci_q04",
        qtype="join",
        question="For each item, report the item name and its category name.",
        gold_sql=(
            "SELECT items.name, categories.name FROM items "
            "JOIN categories ON items.category_id = categories.id;"
        ),
        # No join → wrong-join
        default_wrong_sql="SELECT name, category_id FROM items;",
        unlock_phrase="Join across tables using their declared foreign keys.",
    ),
    # 26. aggregate-with-groupby, correct
    dict(
        qid="sql_ci_q05",
        qtype="aggregate-with-groupby",
        question="How many items are in each category id?",
        gold_sql="SELECT category_id, COUNT(*) FROM items GROUP BY category_id;",
        default_wrong_sql="SELECT category_id, COUNT(*) FROM items GROUP BY category_id;",
        unlock_phrase="",
    ),
    # 27. aggregate-with-groupby, wrong-aggregation (no GROUP BY)
    dict(
        qid="sql_ci_q06",
        qtype="aggregate-with-groupby",
        question="What is the total stock per category id?",
        gold_sql="SELECT category_id, SUM(stock) FROM items GROUP BY category_id;",
        # No GROUP BY → wrong-aggregation
        default_wrong_sql="SELECT category_id, SUM(stock) FROM items;",
        unlock_phrase="When aggregating multiple groups, include GROUP BY.",
    ),
    # 28. aggregate-with-groupby, correct
    dict(
        qid="sql_ci_q07",
        qtype="aggregate-with-groupby",
        question="What is the maximum stock by category id?",
        gold_sql="SELECT category_id, MAX(stock) FROM items GROUP BY category_id;",
        default_wrong_sql="SELECT category_id, MAX(stock) FROM items GROUP BY category_id;",
        unlock_phrase="",
    ),
    # 29. filter-with-having, wrong-filter (drop the stock>0 condition)
    dict(
        qid="sql_ci_q08",
        qtype="filter-with-having",
        question=(
            "Which category ids have more than 1 in-stock item (stock > 0)?"
        ),
        gold_sql=(
            "SELECT category_id FROM items WHERE stock > 0 "
            "GROUP BY category_id HAVING COUNT(*) > 1;"
        ),
        # Drop the stock > 0 filter → wrong-filter
        default_wrong_sql=(
            "SELECT category_id FROM items GROUP BY category_id HAVING COUNT(*) > 1;"
        ),
        unlock_phrase="Apply WHERE clauses to filter rows when the question constrains values.",
    ),
    # 30. filter-with-having, correct
    dict(
        qid="sql_ci_q09",
        qtype="filter-with-having",
        question="Which category ids have total stock exceeding 15?",
        gold_sql=(
            "SELECT category_id FROM items GROUP BY category_id HAVING SUM(stock) > 15;"
        ),
        default_wrong_sql=(
            "SELECT category_id FROM items GROUP BY category_id HAVING SUM(stock) > 15;"
        ),
        unlock_phrase="",
    ),
]


# ===========================================================================
# Assembly
# ===========================================================================


def _compute_gold_rows(meta: dict, gold_sql: str) -> list[list]:
    """Execute the gold SQL against an ephemeral sqlite to capture
    the canonical result-set we serialize into the fixture. Keeps the
    fixture from drifting from sqlite's actual behavior."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    try:
        cur = conn.cursor()
        for stmt in meta["schema_ddl"].split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)
        for stmt in meta["seed_rows"]:
            cur.execute(stmt)
        cur.execute(gold_sql)
        return [list(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _build_instance(meta: dict, q: dict) -> dict:
    """Materialize one fixture entry combining schema metadata + the
    question's SQL pair + the computed gold result-set."""
    gold_rows = _compute_gold_rows(meta, q["gold_sql"])
    return {
        "question_id": q["qid"],
        "question_type": q["qtype"],
        "question": q["question"],
        "schema_id": meta["schema_id"],
        "schema_ddl": meta["schema_ddl"],
        "seed_rows": list(meta["seed_rows"]),
        "tables": list(meta["tables"]),
        "columns_by_table": dict(meta["columns_by_table"]),
        "primary_keys": dict(meta["primary_keys"]),
        "foreign_keys": [list(fk) for fk in meta["foreign_keys"]],
        "gold_sql": q["gold_sql"],
        "gold_result_set": gold_rows,
        "default_wrong_sql": q["default_wrong_sql"],
        "unlock_phrase": q["unlock_phrase"],
    }


def main() -> int:
    instances = []
    for meta, qs in (
        (SCHEMA1_META, SCHEMA1_QUESTIONS),
        (SCHEMA2_META, SCHEMA2_QUESTIONS),
        (SCHEMA3_META, SCHEMA3_QUESTIONS),
    ):
        for q in qs:
            instances.append(_build_instance(meta, q))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(instances, indent=2) + "\n")
    print(f"wrote {len(instances)} instances → {OUT.relative_to(REPO)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
