import random
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
OUT = RAW_DIR / 'synth_jobs_canada.csv'

random.seed(42)

TITLES = [
    'Software Engineer','Backend Developer','Frontend Developer','Full-Stack Developer',
    'Data Scientist','ML Engineer','Data Engineer','DevOps Engineer','Mobile Developer'
]
CITIES = [
    'Toronto ON Canada','Vancouver BC Canada','Montreal QC Canada','Montréal QC Canada','Calgary AB Canada',
    'Ottawa ON Canada','Edmonton AB Canada','Winnipeg MB Canada','Quebec City QC Canada','Halifax NS Canada'
]
COMPANIES = [
    ('Google','FAANG'),('Meta','FAANG'),('Amazon','FAANG'),('Apple','FAANG'),('Netflix','FAANG'),
    ('Microsoft','Big Tech'),('IBM','Big Tech'),('Shopify','Big Tech'),('Snowflake','Big Tech'),
    ('DataNova','Startup'),('CloudSpark','Startup'),('DevWorks','Mid-size/Other'),('NorthTech','Mid-size/Other'),('AlgoHub','Mid-size/Other')
]
SENIORITY = ['Entry','Mid','Senior']
EDU = ["Bachelor's","Master's","Diploma","PhD"]
SKILLS_POOL = [
    'python','java','c++','c#','javascript','typescript','react','vue','angular','nodejs','sql','postgresql','mysql','mongodb',
    'aws','gcp','azure','docker','kubernetes','terraform','linux','git','pandas','numpy','pytorch','tensorflow','sklearn','spark','airflow','rest','graphql'
]

# Base probabilities modifiers
COMPANY_OFFER_MOD = {
    'FAANG': 0.12,
    'Big Tech': 0.10,
    'Startup': 0.09,
    'Mid-size/Other': 0.07,
}
SENIORITY_OFFER_MOD = {
    'Entry': 0.06,
    'Mid': 0.08,
    'Senior': 0.10,
}
EDU_OFFER_MOD = {
    "Bachelor's": 0.00,
    "Master's": 0.01,
    'Diploma': -0.01,
    'PhD': 0.01,
}

N = 600

def sample_row(i: int):
    title = random.choice(TITLES)
    city = random.choice(CITIES)
    company_name, company_type = random.choice(COMPANIES)
    seniority = random.choices(SENIORITY, weights=[0.3,0.5,0.2])[0]
    edu = random.choices(EDU, weights=[0.6,0.3,0.08,0.02])[0]
    exp = max(0, int(random.gauss(3 if seniority=='Mid' else (0 if seniority=='Entry' else 6), 1.5)))
    # skills: base set + noise
    k = random.randint(3, 7)
    skills = random.sample(SKILLS_POOL, k)

    # derive probabilities
    base_p = 0.05
    p_offer = base_p + COMPANY_OFFER_MOD[company_type] + SENIORITY_OFFER_MOD[seniority] + EDU_OFFER_MOD[edu]
    p_offer = max(0.01, min(0.6, p_offer))

    # callbacks and interviews roughly correlated
    callbacks = max(0, int(random.gauss(10 * p_offer + 8, 4)))
    interviews = max(0, int(random.gauss(5 * p_offer + 4, 2)))
    offers = 1 if random.random() < p_offer else 0

    desc = f"{title} working on services; cloud; data; systems."
    req = '; '.join(skills)

    return [i, title, company_name, city, desc, req, exp, edu, callbacks, interviews, offers]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id','title','company_name','location','description','requirements','experience_years','education','callbacks','interviews','offers'])
        for i in range(1, N+1):
            w.writerow(sample_row(i))
    print(f"Wrote synthetic dataset: {OUT}")

if __name__ == '__main__':
    main()
