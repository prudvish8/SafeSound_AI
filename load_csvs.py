import pandas as pd, sqlite3
import csv

conn = sqlite3.connect('welfare_schemes.db')

for name, csv_file in [
    ("schemes", "schemes_updated.csv"),
    ("eligibility_rules", "eligibility_rules_updated.csv"),
    ("scheme_categories", "scheme_categories_updated.csv")
]:
    df = pd.read_csv(csv_file)
    df.to_sql(name, conn, if_exists='replace', index=False)
    print(f"{name} table refreshed from {csv_file}")

conn.close()

SCHEMES_CSV = 'schemes_updated.csv'
RULES_CSV = 'eligibility_rules_updated.csv'

def get_scheme_ids_from_schemes(path):
    ids = set()
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(str(row['scheme_id']).strip())
    return ids

def get_scheme_ids_from_rules(path):
    ids = set()
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(str(row['scheme_id']).strip())
    return ids

if __name__ == '__main__':
    schemes_ids = get_scheme_ids_from_schemes(SCHEMES_CSV)
    rules_ids = get_scheme_ids_from_rules(RULES_CSV)

    print('Schemes in schemes_updated.csv:', schemes_ids)
    print('Schemes referenced in eligibility_rules_updated.csv:', rules_ids)

    # 1. Rules referencing non-existent schemes
    missing_in_schemes = rules_ids - schemes_ids
    if missing_in_schemes:
        print('Rules referencing non-existent schemes:', missing_in_schemes)
    else:
        print('All rules reference valid schemes.')

    # 2. Schemes with no eligibility rules
    missing_in_rules = schemes_ids - rules_ids
    if missing_in_rules:
        print('Schemes with NO eligibility rules:', missing_in_rules)
    else:
        print('All schemes have eligibility rules.')