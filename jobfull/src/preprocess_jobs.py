import os
import sys
import glob
import math
import re
from pathlib import Path
from typing import List, Set

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
OUTPUT_CSV = PROCESSED_DIR / 'canada_cs_jobs.csv'
SUMMARY_TXT = PROCESSED_DIR / 'summary.txt'

# Basic vocab for simple skill extraction
SKILL_VOCAB: Set[str] = {
    'python','java','c++','c#','javascript','typescript','react','angular','vue','node','node.js','nodejs',
    'sql','postgresql','mysql','sqlite','nosql','mongodb','spark','hadoop','pandas','numpy',
    'aws','gcp','azure','docker','kubernetes','k8s','terraform','git','linux',
    'ml','machine learning','deep learning','pytorch','tensorflow','sklearn','scikit-learn','nlp',
    'flask','django','spring','fastapi','rest','api','graphql','etl','airflow',
    'excel','power bi','powerbi','tableau'
}

FAANG = {'google','meta','facebook','amazon','apple','netflix'}
BIG_TECH = {'microsoft','ibm','oracle','salesforce','snowflake','uber','airbnb','shopify'}

TITLE_KEYWORDS = [
    'software','developer','engineer','backend','front-end','frontend','full stack','full-stack','data','ml','machine learning',
    'devops','sre','site reliability','cloud','mobile','ios','android'
]

SENIORITY_MAP = [
    ('Intern', ['intern','co-op','coop']),
    ('Entry', ['junior','jr','new grad','entry','associate']),
    ('Mid', ['mid','intermediate']),
    ('Senior', ['senior','sr','staff','lead','principal','manager','director'])
]

EDU_MAP = {
    "phd": "PhD",
    "doctor": "PhD",
    "master": "Master's",
    "msc": "Master's",
    "ma": "Master's",
    "bachelor": "Bachelor's",
    "bsc": "Bachelor's",
    "ba": "Bachelor's",
    "diploma": "Diploma",
    "college": "Diploma",
}


def normalize_text(x: str) -> str:
    return (x or '').strip()


def is_canada(loc: str) -> bool:
    s = (loc or '').lower()
    if 'canada' in s:
        return True
    # provinces/territories and major cities heuristics
    patterns = [
        'qc','quebec','on','ontario','bc','british columbia','ab','alberta','mb','manitoba','sk','saskatchewan',
        'ns','nova scotia','nb','new brunswick','nl','newfoundland','pei','prince edward island','yt','yukon','nt','northwest territories','nu','nunavut',
        'montreal','montréal','toronto','vancouver','calgary','ottawa','edmonton','winnipeg','quebec city','halifax','victoria','saskatoon','regina','st. johns','st johns','charlottetown','whitehorse','yellowknife','iqaluit'
    ]
    return any(p in s for p in patterns)


def is_cs_job(title: str) -> bool:
    s = (title or '').lower()
    return any(k in s for k in TITLE_KEYWORDS)


def classify_company_type(name: str) -> str:
    s = (name or '').lower().strip()
    base = re.sub(r"[^a-z0-9]+"," ", s).strip()
    if base in FAANG:
        return 'FAANG'
    if base in BIG_TECH:
        return 'Big Tech'
    # crude heuristics
    if any(w in s for w in ['startup','labs']):
        return 'Startup'
    return 'Mid-size/Other'


def classify_seniority(title: str) -> str:
    s = (title or '').lower()
    for level, keys in SENIORITY_MAP:
        if any(k in s for k in keys):
            return level
    return 'Mid'


def parse_skills(text: str) -> List[str]:
    s = (text or '')
    # replace common separators with commas
    s = re.sub(r"[\n\t|/]+", ",", s)
    parts = re.split(r"[;,]", s)
    out = []
    for p in parts:
        token = p.strip().lower()
        if not token:
            continue
        # normalize variations
        token = token.replace('node.js','nodejs').replace('power bi','powerbi')
        if token in SKILL_VOCAB:
            out.append(token)
    # dedupe, preserve order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def normalize_education(edu: str) -> str:
    s = (edu or '').lower()
    for k, v in EDU_MAP.items():
        if k in s:
            return v
    return edu or 'Not Specified'


def to_int(x):
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def process_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # auto-detect columns best-effort
    cols = {c.lower(): c for c in df.columns}
    title_col = cols.get('title') or cols.get('job_title') or list(df.columns)[0]
    loc_col = cols.get('location') or cols.get('city') or cols.get('region') or list(df.columns)[1]
    company_col = cols.get('company_name') or cols.get('company') or cols.get('employer')
    desc_col = cols.get('description') or cols.get('job_description')
    req_col = cols.get('requirements') or cols.get('requirements_text')
    exp_col = cols.get('experience_years') or cols.get('min_experience_years') or cols.get('years_experience')
    edu_col = cols.get('education') or cols.get('degree')
    callbacks_col = cols.get('callbacks')
    interviews_col = cols.get('interviews')
    offers_col = cols.get('offers')

    rows = []
    for _, r in df.iterrows():
        title = normalize_text(r.get(title_col, ''))
        loc = normalize_text(r.get(loc_col, ''))
        if not (is_canada(loc) and is_cs_job(title)):
            continue
        company = normalize_text(r.get(company_col, ''))
        text_blob = ' '.join([
            normalize_text(r.get(desc_col, '')),
            normalize_text(r.get(req_col, ''))
        ])
        skills = parse_skills(text_blob)
        exp = r.get(exp_col, None)
        exp = to_int(exp)
        edu = normalize_education(r.get(edu_col, ''))
        callbacks = to_int(r.get(callbacks_col)) or 0
        interviews = to_int(r.get(interviews_col)) or 0
        offers = to_int(r.get(offers_col)) or 0
        seniority = classify_seniority(title)
        company_type = classify_company_type(company)
        accepted = 1 if offers and offers > 0 else 0

        rows.append({
            'title': title,
            'location': loc,
            'company_name': company,
            'company_type': company_type,
            'seniority_level': seniority,
            'experience_years': exp,
            'education': edu,
            'skills': ';'.join(skills),
            'callbacks': callbacks,
            'interviews': interviews,
            'offers': offers,
            'accepted': accepted
        })

    return pd.DataFrame(rows)


def generate_summary(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Total Canada CS jobs: {len(df)}")
    if len(df):
        lines.append('Distribution:')
        for col in ['company_type','seniority_level','education']:
            counts = df[col].value_counts().to_dict()
            lines.append(f" - {col}: {counts}")
        avg_callbacks = df['callbacks'].mean()
        avg_interviews = df['interviews'].mean()
        offer_rate = df['accepted'].mean()
        lines.append(f"Avg callbacks: {avg_callbacks:.2f}; Avg interviews: {avg_interviews:.2f}; Offer rate: {offer_rate:.2%}")
    return '\n'.join(lines)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files = glob.glob(str(RAW_DIR / '**' / '*.csv'), recursive=True)
    if not files:
        print(f"No CSV files found under {RAW_DIR}. Add CSVs and re-run.")
        return 1

    all_rows = []
    for f in files:
        try:
            part = process_file(Path(f))
            if not part.empty:
                all_rows.append(part)
            print(f"Processed {f}: {len(part)} rows kept")
        except Exception as e:
            print(f"Failed to process {f}: {e}")

    if not all_rows:
        print("No Montreal CS rows found in provided CSVs.")
        return 1

    out = pd.concat(all_rows, ignore_index=True)
    # Deduplicate by title+company+skills
    out = out.drop_duplicates(subset=['title','company_name','skills'])

    out.to_csv(OUTPUT_CSV, index=False)
    summary = generate_summary(out)
    SUMMARY_TXT.write_text(summary, encoding='utf-8')
    print(f"Saved processed dataset to {OUTPUT_CSV}")
    print(summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
