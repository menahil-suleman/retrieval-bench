"""
build_corpus.py — Generates the document corpus and labeled test set.

The corpus simulates the kind of clinical/health documents that would
exist in a CareSync / MindTrace RAG pipeline.  Each document is split
into fixed-size chunks; each chunk gets a unique ID.  The test set
pairs 50 natural-language queries with the ground-truth chunk ID that
contains the correct answer — exactly the labeling structure the
benchmark needs.

Run:
    py data/build_corpus.py
Outputs:
    data/corpus.json    — list of chunk dicts
    data/test_set.json  — list of {query, relevant_chunk_ids} dicts
"""

import json
import os
import sys

# Allow running from the project root or from data/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import CORPUS_FILE, TEST_SET_FILE

# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw document text (realistic but synthetic health-record / care content)
# ─────────────────────────────────────────────────────────────────────────────

RAW_DOCUMENTS = [
    # ── Document 0: Hypertension management ──────────────────────────────────
    {
        "doc_id": "doc_hypertension",
        "title": "Hypertension Management Guidelines",
        "sections": [
            ("intro", """
Hypertension, commonly known as high blood pressure, is defined as a sustained
systolic blood pressure of 130 mmHg or higher, or a diastolic blood pressure of
80 mmHg or higher, according to the 2017 ACC/AHA guidelines.  It is one of the
most prevalent cardiovascular risk factors globally and a leading cause of
stroke, myocardial infarction, and chronic kidney disease.
"""),
            ("lifestyle", """
First-line non-pharmacological interventions for hypertension include the DASH
diet (Dietary Approaches to Stop Hypertension), which emphasises fruits,
vegetables, whole grains, and low-fat dairy while limiting sodium intake to
under 2,300 mg per day.  Regular aerobic exercise — at least 150 minutes of
moderate-intensity activity per week — can reduce systolic pressure by 4–9 mmHg.
Limiting alcohol consumption to no more than one drink per day for women and two
for men also contributes to blood pressure reduction.
"""),
            ("medications", """
Pharmacological treatment typically starts with a thiazide diuretic, an
ACE inhibitor, an angiotensin II receptor blocker (ARB), or a calcium channel
blocker (CCB).  The choice depends on patient comorbidities: ACE inhibitors and
ARBs are preferred in patients with diabetes or chronic kidney disease because
of their nephroprotective effects.  Beta-blockers are reserved for patients with
concurrent coronary artery disease or heart failure.  Combination therapy with
two agents from different classes is often required to achieve target blood
pressure below 130/80 mmHg.
"""),
            ("monitoring", """
Patients on antihypertensive therapy should have blood pressure checked at
every clinical visit, with home monitoring encouraged between visits.
Self-monitoring devices should be validated and calibrated annually.
Ambulatory blood pressure monitoring (ABPM) is recommended when white-coat
hypertension or masked hypertension is suspected.  Laboratory monitoring
includes serum creatinine, electrolytes, and fasting glucose at baseline and
annually, with more frequent checks when adding or adjusting ACE inhibitors
or diuretics due to the risk of hyperkalaemia and renal impairment.
"""),
        ],
    },

    # ── Document 1: Type 2 Diabetes ───────────────────────────────────────────
    {
        "doc_id": "doc_diabetes",
        "title": "Type 2 Diabetes: Diagnosis, Management, and Monitoring",
        "sections": [
            ("diagnosis", """
Type 2 diabetes mellitus is diagnosed when fasting plasma glucose reaches
126 mg/dL (7.0 mmol/L) or more, when the 2-hour glucose value on an oral
glucose tolerance test (OGTT) is 200 mg/dL (11.1 mmol/L) or more, or when
HbA1c is 6.5% or above.  A single abnormal result should be confirmed on a
repeat test unless symptoms of hyperglycaemia are present.  Pre-diabetes is
defined as fasting glucose of 100–125 mg/dL or HbA1c of 5.7–6.4%.
"""),
            ("glycaemic_targets", """
The primary glycaemic target for most non-pregnant adults with type 2 diabetes
is an HbA1c below 7.0%, corresponding to an estimated average glucose of
approximately 154 mg/dL.  Less stringent targets of HbA1c below 8.0% are
appropriate for older adults with multiple comorbidities, limited life
expectancy, or history of severe hypoglycaemia.  More stringent targets below
6.5% may be considered in younger patients with short disease duration who can
achieve them without significant hypoglycaemia.
"""),
            ("medications_dm", """
Metformin remains the preferred initial pharmacological agent for type 2
diabetes because of its efficacy, safety profile, low cost, and potential
cardiovascular benefits.  When metformin is insufficient or contraindicated,
additional agents are chosen based on comorbidities: GLP-1 receptor agonists
and SGLT-2 inhibitors are preferred for patients with established cardiovascular
disease, heart failure, or chronic kidney disease.  Sulfonylureas and basal
insulin remain options when cost is a primary concern.  DPP-4 inhibitors offer
a weight-neutral option with minimal hypoglycaemia risk.
"""),
            ("monitoring_dm", """
Self-monitoring of blood glucose (SMBG) frequency should be individualised.
Patients on insulin require daily monitoring, typically before meals and at
bedtime.  Continuous glucose monitoring (CGM) is increasingly recommended
because it provides data on glucose variability and time-in-range that HbA1c
alone cannot capture.  HbA1c should be measured every three months in patients
not meeting glycaemic targets and every six months once stable.  Annual
screening for microvascular complications includes urine albumin-to-creatinine
ratio, estimated GFR, dilated eye examination, and foot examination.
"""),
        ],
    },

    # ── Document 2: Mental Health — Depression ────────────────────────────────
    {
        "doc_id": "doc_depression",
        "title": "Depression: Clinical Assessment and Treatment",
        "sections": [
            ("assessment", """
Major depressive disorder (MDD) is characterised by at least five of the
following symptoms present for two or more weeks: depressed mood, loss of
interest or pleasure (anhedonia), significant weight change, insomnia or
hypersomnia, psychomotor agitation or retardation, fatigue, feelings of
worthlessness or excessive guilt, diminished concentration, and recurrent
thoughts of death or suicidal ideation.  At least one of the symptoms must be
depressed mood or anhedonia.  Standardised tools such as the PHQ-9 are
validated for both screening and severity monitoring.
"""),
            ("psychotherapy", """
Cognitive behavioural therapy (CBT) is the most extensively studied
psychotherapeutic intervention for depression and is considered first-line
treatment for mild-to-moderate MDD.  CBT targets maladaptive thought patterns
and behavioural avoidance cycles.  Behavioural activation, a component of CBT,
focuses specifically on scheduling rewarding activities to counteract
withdrawal.  Interpersonal therapy (IPT) is equally effective and focuses on
improving communication patterns and interpersonal relationships.  For
moderate-to-severe depression, combined pharmacotherapy and psychotherapy
produces better outcomes than either treatment alone.
"""),
            ("antidepressants", """
Selective serotonin reuptake inhibitors (SSRIs) are the first-line
pharmacological treatment for MDD because of their favourable side-effect
profile relative to older antidepressants.  Commonly prescribed SSRIs include
sertraline, escitalopram, and fluoxetine.  SNRIs such as venlafaxine and
duloxetine are preferred when comorbid anxiety or chronic pain is present.
Response is assessed at 4–8 weeks; an adequate trial is defined as 6–8 weeks
at therapeutic dosage.  If partial response is observed, dose optimisation or
augmentation with bupropion, lithium, or an atypical antipsychotic is considered
before switching agents.
"""),
            ("suicide_risk", """
All patients with depression should be screened for suicidal ideation at every
visit using direct questioning.  The Columbia Suicide Severity Rating Scale
(C-SSRS) stratifies risk into ideation and behaviour categories.  Risk factors
for completed suicide include prior attempt, male gender, older age, social
isolation, access to lethal means, comorbid substance use, and hopelessness.
Patients with active suicidal ideation with a plan or intent require immediate
safety assessment, which may include voluntary or involuntary hospitalisation,
removal of access to means, and intensive outpatient follow-up.
"""),
        ],
    },

    # ── Document 3: Asthma ────────────────────────────────────────────────────
    {
        "doc_id": "doc_asthma",
        "title": "Asthma: Diagnosis and Stepwise Management",
        "sections": [
            ("pathophysiology", """
Asthma is a chronic inflammatory airway disease characterised by variable and
reversible airflow obstruction, bronchial hyperresponsiveness, and airway
remodelling.  The inflammatory process is predominantly eosinophilic in
allergic asthma, driven by Th2-mediated immune responses involving IgE, IL-4,
IL-5, and IL-13.  Non-allergic asthma may involve neutrophilic or paucigranulocytic
inflammation.  Triggers include allergens, respiratory infections, exercise,
cold air, irritants such as smoke, and non-steroidal anti-inflammatory drugs
(NSAIDs) in aspirin-exacerbated respiratory disease.
"""),
            ("diagnosis_asthma", """
Diagnosis of asthma requires a history of variable respiratory symptoms —
wheeze, shortness of breath, chest tightness, and cough — and confirmation of
variable expiratory airflow limitation by spirometry.  A post-bronchodilator
increase in FEV1 of at least 12% and 200 mL confirms reversibility.
Bronchial provocation testing with methacholine is used when spirometry is
normal but asthma is still suspected.  Fractional exhaled nitric oxide (FeNO)
above 40 ppb supports eosinophilic airway inflammation.
"""),
            ("treatment_asthma", """
The stepwise approach to asthma management begins with as-needed low-dose
inhaled corticosteroid (ICS) combined with a fast-acting bronchodilator (SABA
or formoterol) at Step 1.  Regular low-dose ICS is added at Step 2, progressing
to medium-dose ICS or ICS plus a long-acting beta-agonist (LABA) at Step 3.
Steps 4 and 5 involve high-dose ICS/LABA combinations, tiotropium add-on, and
biologic therapies targeting IgE (omalizumab), IL-5 (mepolizumab, reslizumab),
or the IL-4/IL-13 receptor (dupilumab) for severe uncontrolled asthma.
"""),
            ("exacerbation", """
Acute asthma exacerbations are classified as mild, moderate, severe, or
life-threatening based on peak flow, oxygen saturation, use of accessory
muscles, and ability to speak.  Initial treatment includes high-flow oxygen to
maintain SpO2 above 94%, repeated doses of inhaled SABA every 20 minutes,
systemic corticosteroids (prednisolone 40–50 mg orally or hydrocortisone IV),
and ipratropium bromide in severe cases.  Patients with life-threatening
features — silent chest, cyanosis, bradycardia, or exhaustion — require
immediate ICU admission and may need non-invasive or mechanical ventilation.
"""),
        ],
    },

    # ── Document 4: Nutrition ─────────────────────────────────────────────────
    {
        "doc_id": "doc_nutrition",
        "title": "Clinical Nutrition and Dietary Assessment",
        "sections": [
            ("macronutrients", """
The three macronutrients — carbohydrates, proteins, and fats — provide the
body's energy substrate.  Carbohydrates should constitute 45–65% of total
caloric intake, proteins 10–35%, and fats 20–35%.  Dietary fibre, though not
caloric, is classified alongside carbohydrates and adults should consume a
minimum of 25–38 grams per day from whole grains, legumes, fruits, and
vegetables.  Saturated fat intake should be kept below 10% of total calories
to reduce cardiovascular risk, with trans fats minimised as much as possible.
"""),
            ("micronutrients", """
Key micronutrients frequently deficient in clinical populations include iron,
vitamin D, vitamin B12, folate, calcium, and zinc.  Iron-deficiency anaemia is
the most common nutritional deficiency worldwide, particularly in women of
reproductive age, young children, and patients with chronic inflammatory
conditions.  Vitamin D deficiency is prevalent in populations with limited
sun exposure and is associated with bone disease, immune dysfunction, and
cardiovascular risk.  Vitamin B12 deficiency causes megaloblastic anaemia and
subacute combined degeneration of the spinal cord and must be distinguished
from folate deficiency by measuring methylmalonic acid and homocysteine levels.
"""),
            ("malnutrition_screening", """
Malnutrition screening should be performed on all hospitalised patients using
a validated tool such as the Malnutrition Universal Screening Tool (MUST) or
the Nutritional Risk Screening 2002 (NRS-2002).  MUST incorporates BMI, recent
unintentional weight loss percentage, and acute disease effect.  Patients at
medium or high risk should receive a full dietitian assessment, personalised
nutritional goals, and documented monitoring.  Enteral nutrition via nasogastric
or nasojejunal tube is preferred over parenteral nutrition when the
gastrointestinal tract is functional.
"""),
            ("obesity", """
Obesity is defined as a body mass index (BMI) of 30 kg/m² or above.  Class I
obesity covers BMI 30–34.9, Class II 35–39.9, and Class III (severe) 40 and
above.  Management combines dietary intervention, increased physical activity,
behavioural therapy, and where indicated, pharmacotherapy or bariatric surgery.
Caloric restriction producing a deficit of 500–750 kcal per day typically
yields 0.5–0.75 kg of weight loss per week.  GLP-1 receptor agonists such as
semaglutide and tirzepatide have demonstrated significant weight loss efficacy
in clinical trials and are approved for chronic weight management.
"""),
        ],
    },

    # ── Document 5: Sleep disorders ───────────────────────────────────────────
    {
        "doc_id": "doc_sleep",
        "title": "Sleep Disorders: Assessment and Management",
        "sections": [
            ("sleep_physiology", """
Normal adult sleep consists of non-REM (NREM) and REM cycles repeating
approximately every 90 minutes.  NREM sleep is divided into three stages:
N1 (light sleep), N2 (light-to-moderate sleep with sleep spindles and K-complexes),
and N3 (slow-wave or deep sleep critical for physical restoration and memory
consolidation).  REM sleep, characterised by rapid eye movements and vivid
dreaming, is essential for emotional processing and cognitive function.  Most
adults require 7–9 hours of sleep per night, with total sleep time decreasing
with age.
"""),
            ("insomnia", """
Insomnia disorder is diagnosed when difficulty initiating or maintaining sleep,
or early morning awakening, causes significant distress or functional impairment
and occurs at least three nights per week for three or more months.  Acute
situational insomnia resolves with removal of the precipitating stressor.
Cognitive behavioural therapy for insomnia (CBT-I) is the recommended
first-line treatment regardless of insomnia duration, comprising sleep
restriction therapy, stimulus control, relaxation training, and cognitive
restructuring.  Pharmacological treatment with low-dose doxepin, melatonin
receptor agonists, or dual orexin receptor antagonists (DORAs) may be used
as short-term adjuncts.
"""),
            ("sleep_apnoea", """
Obstructive sleep apnoea (OSA) is characterised by repetitive partial or
complete upper airway collapse during sleep, producing apnoeas and hypopnoeas.
Diagnosis requires overnight polysomnography or home sleep apnoea testing;
OSA is defined by an apnoea-hypopnoea index (AHI) of five or more events per
hour with associated symptoms, or AHI of fifteen or more regardless of symptoms.
Continuous positive airway pressure (CPAP) is the primary treatment; adherence
requires at least four hours of use on 70% of nights.  Mandibular advancement
devices are an alternative for mild-to-moderate OSA or CPAP-intolerant patients.
Positional therapy and weight loss are adjunctive strategies.
"""),
            ("narcolepsy", """
Narcolepsy type 1 is characterised by excessive daytime sleepiness and
cataplexy — sudden loss of muscle tone triggered by emotion — and is caused
by the autoimmune destruction of hypothalamic hypocretin-producing neurons.
Type 2 narcolepsy presents with daytime sleepiness without cataplexy and
has normal or borderline cerebrospinal fluid hypocretin levels.
Diagnosis is confirmed by nocturnal polysomnography followed by the multiple
sleep latency test (MSLT), demonstrating a mean sleep onset latency of less
than eight minutes and two or more sleep-onset REM periods.  Treatment includes
modafinil or armodafinil for sleepiness, sodium oxybate for both sleepiness and
cataplexy, and venlafaxine or fluoxetine for isolated cataplexy.
"""),
        ],
    },

    # ── Document 6: Chronic Kidney Disease ───────────────────────────────────
    {
        "doc_id": "doc_ckd",
        "title": "Chronic Kidney Disease: Staging, Progression, and Management",
        "sections": [
            ("staging", """
Chronic kidney disease (CKD) is defined as structural or functional kidney
abnormalities present for more than three months.  The KDIGO classification
uses two dimensions: GFR category (G1–G5) and albuminuria category (A1–A3).
G1 represents GFR ≥90 mL/min/1.73 m², G3a is 45–59, G3b is 30–44, G4 is
15–29, and G5 is <15 mL/min/1.73 m² (kidney failure).  Albuminuria categories
are A1 (<30 mg/g), A2 (30–300 mg/g, moderately increased), and A3 (>300 mg/g,
severely increased).  Both dimensions together predict risk of progression,
cardiovascular events, and mortality.
"""),
            ("progression", """
The primary drivers of CKD progression are diabetic nephropathy, hypertensive
nephrosclerosis, and glomerulonephritis.  Modifiable risk factors include
uncontrolled hyperglycaemia, uncontrolled hypertension, proteinuria, smoking,
obesity, and nephrotoxin exposure.  The rate of GFR decline can be estimated
by measuring serum creatinine and cystatin C regularly; a decline of more than
5 mL/min/1.73 m² per year indicates rapid progression.  SGLT-2 inhibitors have
demonstrated significant nephroprotective effects in CKD patients with and
without diabetes, reducing the risk of kidney failure and cardiovascular events.
"""),
            ("anaemia_ckd", """
Anaemia is a common complication of CKD, primarily due to reduced erythropoietin
production by the diseased kidney, and is defined as haemoglobin below 13 g/dL
in men and below 12 g/dL in women.  Iron deficiency must be corrected before
initiating erythropoiesis-stimulating agents (ESAs).  Intravenous iron is
preferred over oral iron in dialysis patients due to better absorption and
efficacy.  Haemoglobin targets when using ESAs should not exceed 11.5 g/dL
to avoid increased risk of stroke and cardiovascular events demonstrated in
trials using higher targets.
"""),
            ("dialysis", """
Renal replacement therapy (RRT) includes haemodialysis, peritoneal dialysis,
and kidney transplantation.  Haemodialysis is typically performed three times
per week for four hours per session at a dialysis centre or home.  Peritoneal
dialysis uses the peritoneal membrane as a filter and can be performed as
continuous ambulatory peritoneal dialysis (CAPD) or automated peritoneal
dialysis (APD) overnight.  Kidney transplantation offers the best survival
outcomes and quality of life and is preferred for eligible patients.
Pre-emptive transplantation (before dialysis initiation) is associated with
better graft survival.
"""),
        ],
    },

    # ── Document 7: Cardiovascular Risk ──────────────────────────────────────
    {
        "doc_id": "doc_cardio",
        "title": "Cardiovascular Risk Assessment and Prevention",
        "sections": [
            ("risk_calc", """
Cardiovascular risk is estimated using validated tools such as the Pooled
Cohort Equations (PCE), which calculate 10-year risk of atherosclerotic
cardiovascular disease (ASCVD) events — myocardial infarction and stroke.
Inputs include age, sex, race, total cholesterol, HDL cholesterol, systolic
blood pressure, blood pressure treatment status, diabetes, and smoking.
A 10-year ASCVD risk below 5% is considered low, 5–7.4% borderline, 7.5–19.9%
intermediate, and 20% or above high.  The risk-enhancing factors such as
family history of premature ASCVD, LDL ≥160 mg/dL, chronic kidney disease,
and inflammatory conditions can shift borderline patients toward statin therapy.
"""),
            ("lipid_management", """
Statin therapy is the cornerstone of lipid-lowering treatment for cardiovascular
risk reduction.  High-intensity statins (atorvastatin 40–80 mg or rosuvastatin
20–40 mg) reduce LDL by at least 50% and are recommended for patients with
established ASCVD (secondary prevention) and high-risk primary prevention.
Moderate-intensity statins reduce LDL by 30–49%.  For patients who do not
achieve LDL targets on maximum tolerated statin therapy, ezetimibe is added
first, followed by PCSK9 inhibitors (evolocumab, alirocumab) if further LDL
reduction is needed.  LDL target for very high-risk patients is below 55 mg/dL.
"""),
            ("antiplatelet", """
Aspirin 75–100 mg daily is recommended for all patients with established ASCVD
as secondary prevention, indefinitely unless contraindicated.  Dual antiplatelet
therapy (DAPT) with aspirin plus a P2Y12 inhibitor (clopidogrel, ticagrelor, or
prasugrel) is used for 12 months following acute coronary syndrome (ACS) or
percutaneous coronary intervention (PCI) with drug-eluting stent.  Aspirin for
primary prevention is no longer routinely recommended because the bleeding risk
generally outweighs the cardiovascular benefit in patients without established
disease, particularly those over 70 years of age.
"""),
            ("heart_failure", """
Heart failure is classified by left ventricular ejection fraction (LVEF):
HFrEF (reduced, LVEF <40%), HFmrEF (mildly reduced, 40–49%), and HFpEF
(preserved, ≥50%).  Guideline-directed medical therapy (GDMT) for HFrEF
comprises four drug classes shown to reduce mortality: ACE inhibitors or ARBs
(or sacubitril/valsartan), beta-blockers, mineralocorticoid receptor antagonists,
and SGLT-2 inhibitors.  Device therapy includes implantable cardioverter-
defibrillators (ICD) for patients with LVEF ≤35% at least three months after
optimising GDMT, and cardiac resynchronisation therapy (CRT) for those with
LBBB and QRS ≥150 ms.
"""),
        ],
    },

    # ── Document 8: Pharmacology — Drug Interactions ─────────────────────────
    {
        "doc_id": "doc_pharmacology",
        "title": "Clinically Significant Drug Interactions",
        "sections": [
            ("cyp450", """
The cytochrome P450 (CYP450) enzyme system is responsible for the metabolism
of approximately 70–80% of clinically used drugs.  CYP3A4 is the most abundant
isoenzyme and metabolises drugs such as statins, calcium channel blockers,
benzodiazepines, and antiretrovirals.  Strong CYP3A4 inhibitors — including
azole antifungals (ketoconazole, itraconazole), macrolide antibiotics
(clarithromycin, erythromycin), and grapefruit juice — increase plasma
concentrations of CYP3A4 substrates, raising the risk of toxicity.
Strong CYP3A4 inducers — rifampicin, carbamazepine, phenytoin, St John's wort —
accelerate metabolism and can reduce efficacy of substrate drugs.
"""),
            ("warfarin", """
Warfarin has a narrow therapeutic index and is subject to numerous clinically
significant drug interactions.  Drugs that potentiate warfarin's anticoagulant
effect (raising INR) include amiodarone, fluconazole, metronidazole, fluoxetine,
and NSAIDs.  Drugs that reduce warfarin efficacy (lowering INR) include
rifampicin, carbamazepine, and St John's wort.  High-vitamin-K foods such as
green leafy vegetables can also reduce INR; patients should be counselled to
maintain a consistent dietary vitamin K intake rather than avoiding these foods
entirely.  INR should be rechecked within one week of starting, stopping, or
changing the dose of any interacting drug.
"""),
            ("serotonin_syndrome", """
Serotonin syndrome is a potentially life-threatening drug interaction resulting
from excess serotonergic activity in the central and peripheral nervous systems.
It is characterised by the clinical triad of mental status changes (agitation,
confusion), autonomic instability (hyperthermia, tachycardia, diaphoresis,
hypertension), and neuromuscular abnormalities (tremor, clonus, hyperreflexia).
High-risk drug combinations include SSRIs or SNRIs with MAO inhibitors, tramadol,
linezolid, or methylene blue.  Concomitant use of a MAOI with any serotonergic
agent is absolutely contraindicated; a washout period of at least 14 days is
required when switching between SSRIs and MAOIs.
"""),
            ("qt_prolongation", """
QT prolongation and the associated risk of torsades de pointes is a serious
adverse effect of many drug classes including antipsychotics (haloperidol,
quetiapine), antiarrhythmics (amiodarone, sotalol), antibiotics (azithromycin,
ciprofloxacin), and antihistamines.  Risk is increased by hypokalaemia,
hypomagnesaemia, female sex, bradycardia, and congenital long-QT syndrome.
A corrected QT interval (QTc) exceeding 500 ms is generally considered the
threshold at which the risk of arrhythmia is unacceptable.  The CredibleMeds
database provides an up-to-date, evidence-based risk classification for drugs
and QT prolongation.
"""),
        ],
    },

    # ── Document 9: Preventive Care ───────────────────────────────────────────
    {
        "doc_id": "doc_preventive",
        "title": "Preventive Care and Screening Recommendations",
        "sections": [
            ("cancer_screening", """
Colorectal cancer screening is recommended for average-risk adults starting at
age 45.  Options include colonoscopy every 10 years, annual high-sensitivity
stool-based tests (FIT or Cologuard), or CT colonography every 5 years.
Breast cancer screening with annual mammography is recommended for women aged
40–74; those with BRCA1/2 mutations or strong family history should begin
earlier and consider MRI alongside mammography.  Cervical cancer screening
using Pap smear every 3 years or co-testing with HPV every 5 years is
recommended from age 21 until age 65.  Lung cancer screening with annual
low-dose CT is recommended for adults 50–80 years with a 20 pack-year smoking
history who currently smoke or quit within the past 15 years.
"""),
            ("vaccinations", """
Core adult vaccinations include annual influenza vaccine, one-time Tdap followed
by Td booster every 10 years, pneumococcal vaccines (PCV15 or PCV20 with
PPSV23 as applicable), shingles vaccine (Shingrix two-dose series) for adults
over 50, and COVID-19 primary series with updated boosters.  The HPV vaccine
is recommended through age 26 and shared decision-making for ages 27–45.
Meningococcal vaccines (MenACWY and MenB) are recommended for adolescents,
college freshmen in residence halls, and immunocompromised individuals.
Hepatitis B vaccination is recommended for unvaccinated adults, particularly
those at increased risk.
"""),
            ("well_visit", """
Annual well visits should include blood pressure measurement, BMI calculation
and obesity counselling, fasting lipid panel for adults 21 and older, fasting
glucose or HbA1c for screening diabetes in overweight adults, depression
screening with PHQ-2 or PHQ-9, alcohol misuse screening with AUDIT-C, and
tobacco use counselling.  Vision and hearing screening are performed at
clinician discretion based on age and symptoms.  Fall risk assessment is
recommended annually for adults over 65, along with functional status review
and medication reconciliation to identify drugs contributing to fall risk.
"""),
            ("immunocompromised", """
Immunocompromised patients — including those on high-dose corticosteroids,
biologic therapies, chemotherapy, or with HIV — require modified vaccination
schedules.  Live vaccines (MMR, varicella, LAIV) are generally contraindicated
in severely immunocompromised patients.  Inactivated vaccines are safe but may
produce suboptimal immune responses; higher doses or extra doses may be needed.
Pneumococcal and influenza vaccines are especially important in this population
given the elevated risk of serious infection.  HIV-positive patients should
receive Pneumovax 23 if CD4 count is above 200 cells/mm³ and should be
revaccinated five years later.
"""),
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Chunk the documents
# ─────────────────────────────────────────────────────────────────────────────

def build_corpus(documents: list[dict]) -> list[dict]:
    """Flatten documents into a list of chunk dicts.

    Each chunk:
        chunk_id   — unique string, e.g. "doc_hypertension__intro"
        doc_id     — parent document id
        title      — document title
        section    — section name within the document
        text       — raw text content (stripped)
    """
    chunks = []
    for doc in documents:
        for section_name, text in doc["sections"]:
            chunk = {
                "chunk_id": f"{doc['doc_id']}__{section_name}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "section": section_name,
                "text": text.strip(),
            }
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3. Labeled test set (50 queries with ground-truth chunk IDs)
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (query string, [list of relevant chunk IDs])
# Multiple relevant chunk IDs are used for nDCG calculation.
# The FIRST id in the list is the primary (most relevant) chunk.

LABELED_QUERIES = [
    # Hypertension
    ("What is the definition of hypertension according to ACC/AHA guidelines?",
     ["doc_hypertension__intro"]),
    ("Which dietary approach is recommended to lower blood pressure?",
     ["doc_hypertension__lifestyle"]),
    ("What is the DASH diet and how does it affect blood pressure?",
     ["doc_hypertension__lifestyle"]),
    ("How much can aerobic exercise reduce systolic blood pressure?",
     ["doc_hypertension__lifestyle"]),
    ("When are ACE inhibitors preferred over calcium channel blockers for hypertension?",
     ["doc_hypertension__medications"]),
    ("What is the target blood pressure for patients on antihypertensive therapy?",
     ["doc_hypertension__medications"]),
    ("When is ambulatory blood pressure monitoring recommended?",
     ["doc_hypertension__monitoring"]),
    ("What lab tests should be monitored in patients on ACE inhibitors for hypertension?",
     ["doc_hypertension__monitoring"]),

    # Diabetes
    ("What HbA1c level confirms a diagnosis of type 2 diabetes?",
     ["doc_diabetes__diagnosis"]),
    ("What is the definition of pre-diabetes based on fasting glucose?",
     ["doc_diabetes__diagnosis"]),
    ("What is the HbA1c target for most non-pregnant adults with type 2 diabetes?",
     ["doc_diabetes__glycaemic_targets"]),
    ("When is a less stringent HbA1c target appropriate for diabetes patients?",
     ["doc_diabetes__glycaemic_targets"]),
    ("Why is metformin the preferred first-line drug for type 2 diabetes?",
     ["doc_diabetes__medications_dm"]),
    ("Which diabetes medications are preferred in patients with heart failure?",
     ["doc_diabetes__medications_dm"]),
    ("What does continuous glucose monitoring measure that HbA1c cannot?",
     ["doc_diabetes__monitoring_dm"]),
    ("How often should HbA1c be measured in stable diabetes patients?",
     ["doc_diabetes__monitoring_dm"]),

    # Depression
    ("What symptoms are required to diagnose major depressive disorder?",
     ["doc_depression__assessment"]),
    ("What is the PHQ-9 used for in depression care?",
     ["doc_depression__assessment"]),
    ("What is cognitive behavioural therapy and how does it treat depression?",
     ["doc_depression__psychotherapy"]),
    ("What is behavioural activation in the context of depression treatment?",
     ["doc_depression__psychotherapy"]),
    ("Which SSRIs are commonly prescribed for major depressive disorder?",
     ["doc_depression__antidepressants"]),
    ("How long does an adequate antidepressant trial last?",
     ["doc_depression__antidepressants"]),
    ("What risk factors are associated with completed suicide in depressed patients?",
     ["doc_depression__suicide_risk"]),
    ("What is the Columbia Suicide Severity Rating Scale used for?",
     ["doc_depression__suicide_risk"]),

    # Asthma
    ("What immune cells and cytokines drive allergic asthma inflammation?",
     ["doc_asthma__pathophysiology"]),
    ("How is reversible airflow obstruction confirmed by spirometry in asthma?",
     ["doc_asthma__diagnosis_asthma"]),
    ("What does a FeNO above 40 ppb indicate in asthma diagnosis?",
     ["doc_asthma__diagnosis_asthma"]),
    ("What biologic therapies are used for severe uncontrolled asthma?",
     ["doc_asthma__treatment_asthma"]),
    ("What is the first-line treatment for an acute severe asthma exacerbation?",
     ["doc_asthma__exacerbation"]),
    ("When should a patient with asthma be admitted to ICU?",
     ["doc_asthma__exacerbation"]),

    # Nutrition
    ("What percentage of daily calories should come from carbohydrates?",
     ["doc_nutrition__macronutrients"]),
    ("How much dietary fibre should adults consume per day?",
     ["doc_nutrition__macronutrients"]),
    ("What conditions are associated with vitamin D deficiency?",
     ["doc_nutrition__micronutrients"]),
    ("How is vitamin B12 deficiency distinguished from folate deficiency?",
     ["doc_nutrition__micronutrients"]),
    ("What tool is used to screen for malnutrition in hospitalised patients?",
     ["doc_nutrition__malnutrition_screening"]),
    ("Which weight loss medications have been approved for chronic weight management?",
     ["doc_nutrition__obesity"]),

    # Sleep
    ("How many REM-NREM cycles occur during a typical night of sleep?",
     ["doc_sleep__sleep_physiology"]),
    ("What is the first-line treatment for chronic insomnia disorder?",
     ["doc_sleep__insomnia"]),
    ("What is CBT-I and what components does it include?",
     ["doc_sleep__insomnia"]),
    ("How is obstructive sleep apnoea diagnosed using AHI criteria?",
     ["doc_sleep__sleep_apnoea"]),
    ("What is the minimum CPAP adherence required for sleep apnoea treatment?",
     ["doc_sleep__sleep_apnoea"]),
    ("What causes narcolepsy type 1 and how is it diagnosed?",
     ["doc_sleep__narcolepsy"]),

    # CKD
    ("How is chronic kidney disease staged using the KDIGO classification?",
     ["doc_ckd__staging"]),
    ("What effect do SGLT-2 inhibitors have on CKD progression?",
     ["doc_ckd__progression"]),
    ("What haemoglobin target should be maintained when using ESAs in CKD?",
     ["doc_ckd__anaemia_ckd"]),
    ("What are the differences between haemodialysis and peritoneal dialysis?",
     ["doc_ckd__dialysis"]),

    # Cardiovascular
    ("What 10-year ASCVD risk is considered high according to pooled cohort equations?",
     ["doc_cardio__risk_calc"]),
    ("What LDL target is recommended for very high-risk cardiovascular patients?",
     ["doc_cardio__lipid_management"]),
    ("When is dual antiplatelet therapy recommended after a coronary stent?",
     ["doc_cardio__antiplatelet"]),
    ("What four drug classes constitute guideline-directed medical therapy for HFrEF?",
     ["doc_cardio__heart_failure"]),

    # Pharmacology
    ("Which antibiotics inhibit CYP3A4 and can raise statin plasma levels?",
     ["doc_pharmacology__cyp450"]),
    ("What foods can lower INR in patients taking warfarin?",
     ["doc_pharmacology__warfarin"]),
    ("What drug combinations are most likely to cause serotonin syndrome?",
     ["doc_pharmacology__serotonin_syndrome"]),
    ("What QTc threshold indicates an unacceptable risk of arrhythmia?",
     ["doc_pharmacology__qt_prolongation"]),

    # Preventive
    ("At what age should average-risk adults start colorectal cancer screening?",
     ["doc_preventive__cancer_screening"]),
    ("What are the criteria for lung cancer screening with low-dose CT?",
     ["doc_preventive__cancer_screening"]),
    ("Which vaccines are recommended for adults over 50?",
     ["doc_preventive__vaccinations"]),
    ("Are live vaccines safe in immunocompromised patients?",
     ["doc_preventive__immunocompromised"]),
    ("What screening tests are included in an annual well visit for adults?",
     ["doc_preventive__well_visit"]),
]

assert len(LABELED_QUERIES) >= 40, f"Expected at least 40 queries, got {len(LABELED_QUERIES)}"


def build_test_set(labeled_queries: list[tuple]) -> list[dict]:
    """Convert the labeled query tuples to a list of dicts."""
    test_set = []
    for idx, (query, relevant_ids) in enumerate(labeled_queries):
        test_set.append({
            "query_id": f"q{idx:03d}",
            "query": query,
            "relevant_chunk_ids": relevant_ids,
        })
    return test_set


# ─────────────────────────────────────────────────────────────────────────────
# 4. Write output files
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(os.path.dirname(CORPUS_FILE), exist_ok=True)

    corpus = build_corpus(RAW_DOCUMENTS)
    test_set = build_test_set(LABELED_QUERIES)

    # Validate: every referenced chunk_id exists in the corpus
    corpus_ids = {c["chunk_id"] for c in corpus}
    for item in test_set:
        for cid in item["relevant_chunk_ids"]:
            assert cid in corpus_ids, f"chunk_id '{cid}' not found in corpus"

    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    with open(TEST_SET_FILE, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    print(f"Corpus  : {len(corpus)} chunks  → {CORPUS_FILE}")
    print(f"Test set: {len(test_set)} queries → {TEST_SET_FILE}")
    print("Validation: all chunk IDs verified ✓")
