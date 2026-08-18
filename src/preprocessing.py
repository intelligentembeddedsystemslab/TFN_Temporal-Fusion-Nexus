import pandas as pd
import numpy as np
import os
import json
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


from config import CONFIG

# Preprocessing for the NephroCAGE dataset v1


@dataclass
class PreprocessingArtifacts:
    label_encoders: dict[str, LabelEncoder]
    categorical_cardinalities: list[int]
    static_scaler: StandardScaler
    ts_scaler: StandardScaler

files = {
        'baseline_parameters': '1_BAseline_parameter_fertig2_HLA_Pirche_final.xlsx',
        'donorparameters': '2_donoparameter_final.xlsx',
        'exams': '4_exams.csv',
        'biopsy': '5 Biopsy_patho_kreuz_extensive.xlsx',
        'lab_cohort': '6 Lab_cohort.csv',
        'clinical_assessment': '7_clinical_assessment.csv',
        'medication': '8_Medikation.csv',
        'hospitalization': '9_Hospitalization.xlsx',
        'hla': '10_HLA-DSA_timecourse.xlsx'
    }

def read_files(project_path):
    # reads files returns the dataframe for each file
    data_folder = 'data/v1'
    dfs = {}

    # date of transplant, gender, date of birth, dialysis... of 3878 patients (> num patients in hosp)
    dfs['baseline_parameters'] = pd.read_excel(os.path.join(project_path, data_folder, files['baseline_parameters']))

    dfs['clinical_assessment'] = pd.read_csv(os.path.join(project_path, data_folder, files['clinical_assessment']), sep=';', encoding='ISO-8859-1', low_memory=False)

    dfs['exams'] = pd.read_csv(os.path.join(project_path, data_folder, files['exams']), sep=';', encoding='ISO-8859-1')

    # lab measurements of e.g. Kreatinin, Protein
    dfs['lab_cohort'] = pd.read_csv(os.path.join(project_path, data_folder, files['lab_cohort']), sep=';', encoding='ISO-8859-1')

    dfs['biopsy'] = pd.read_excel(os.path.join(project_path, data_folder, files['biopsy']))

    #dfs['hla'] = pd.read_excel(os.path.join(project_path, data_folder, files['hla']))

    # date of death, medication, date of transplant
    dfs['medikation'] = pd.read_csv(os.path.join(project_path, data_folder, files['medication']), sep=';', encoding='ISO-8859-1')

    # hosp start and end
    dfs['hospitalization'] = pd.read_excel(os.path.join(project_path, data_folder, files['hospitalization']), engine='openpyxl')

    # info about donor type, bloodgroup, age etc.
    dfs['donorparameters'] = pd.read_excel(os.path.join(project_path, data_folder, files['donorparameters']), engine='openpyxl')

    return dfs

def get_dfs(project_path):
    dfs = read_files(project_path)
    return dfs


def create_static_df(dfs):
    # expects dfs read from the original files
    # removes NaNs patient_ids, donor_ids, transplantation_ids
    # see preprocessing notebook
    # returns static df, matching patients to donors
    static_df = dfs['baseline_parameters'][['PatientID', 'TransplantationID', 'SpenderID', 'Geburtsdatum', 'Todesdatum', 'Datum', 'Geschlecht', 'Grunderkrankung', 'Datum_erste_Dialyse', 'Dialyse_Anzahl', 'Alter', 'Blutgruppe', 'Koerpergroesse', 'Date of graft loss', 'Loss cause', 'cold ischemia time', 'PIRCHE_Score', 'MMA_broad', 'MMB_broad', 'MMDR_broad', 'MM_broad']].rename(
        columns={
            'PatientID': 'patient_id',
            'SpenderID': 'donor_id',
            'TransplantationID': 'transplant_id',
            'Geburtsdatum': 'birth_date',
            'Todesdatum': 'death_date',
            'Datum': 'transplant_date',
            'Geschlecht': 'gender',
            'Grunderkrankung': 'underlying_disease',
            'Datum_erste_Dialyse': 'first_dialysis_date',
            'Dialyse_Anzahl': 'number_dialyses',
            'Alter': 'age',
            'Blutgruppe': 'blood_group',
            'Koerpergroesse': 'height',
            'Date of graft loss': 'loss_date',
            'Loss cause': 'loss_cause',
            'cold ischemia time': 'cold_ischemia_time',
            'PIRCHE_Score': 'pirche_score',
            'MMA_broad': 'mma_broad',
            'MMB_broad': 'mmb_broad',
            'MMDR_broad': 'mmdr_broad',
            'MM_broad': 'mm_broad', # sum of above three HLA mismatches, A,B,DR are different loci 
        }
    )

    static_df = static_df.dropna(subset=['patient_id', 'donor_id', 'transplant_id'])
    static_df = static_df.merge(
        dfs['donorparameters'][['PatientID', 'TransplantationID', 'SpenderID', 'gender_donor', 'age_donor', 'donor_bloodgroup',
                                'type_of_donation', 'donor_COD', 'donor_weight', 'donor_height', 'SOURCE']],
        how='left',  # Left join to keep all records in static_df
        left_on=['patient_id', 'transplant_id', 'donor_id'],
        right_on=['PatientID', 'TransplantationID', 'SpenderID']
    )

    static_df = static_df.drop(columns=['PatientID', 'TransplantationID', 'SpenderID'])
    static_df = static_df.drop_duplicates(subset='patient_id', keep='first')

    static_df['transplant_date'] = pd.to_datetime(static_df['transplant_date'], dayfirst=True, errors='coerce')
    static_df['loss_date'] = pd.to_datetime(static_df['loss_date'], dayfirst=True, errors='coerce')
    static_df['death_date'] = pd.to_datetime(static_df['death_date'], dayfirst=True, errors='coerce')

    # if loss date is given calculate the relative days from transplantation date
    static_df['loss_rel_days'] = (static_df['loss_date'] - static_df['transplant_date']).dt.days
    # same for death date
    static_df['death_rel_days'] = (static_df['death_date'] - static_df['transplant_date']).dt.days

    # selecting relevant features
    # these are further filtered by the actual features in CONFIG.py
    static_df = static_df[
        [
            'patient_id',
            'transplant_id',  # keep for medication merging
            'transplant_date',  # keep for medication merging
            'gender',
            'age',
            'underlying_disease',
            'number_dialyses',
            'blood_group',
            'height',
            'death_rel_days',
            'loss_rel_days',
            'loss_cause',
            'cold_ischemia_time',
            'gender_donor',
            'age_donor',
            'donor_bloodgroup',
            'type_of_donation',
            'donor_height',
            'donor_weight',
            'pirche_score',
            'mma_broad',
            'mmb_broad',
            'mmdr_broad',
            'mm_broad',
        ]
    ]

    mm_cols = ['mma_broad', 'mmb_broad', 'mmdr_broad', 'mm_broad']

    for col in mm_cols:
        static_df[col] = pd.to_numeric(static_df[col], errors='coerce').astype('Int64')  # Keeps NaNs and ensures integer dtype

    static_df['gender'] = static_df['gender'].str.lower()
    static_df['gender_donor'] = static_df['gender_donor'].str.lower()
    static_df['gender_donor'] = static_df['gender_donor'].apply(lambda x: x if x in ['m', 'w'] else 'm')

    static_df['type_of_donation'] = static_df['type_of_donation'].apply(lambda x: np.nan if pd.isna(x) else 'hirntot' if 'hirntot' in str(x).lower() else 'lebend')

    static_df['pirche_score'] = static_df['pirche_score'].astype(str).str.replace('_x000D_', '', regex=False) # remove artifacts
    static_df['pirche_score'] = pd.to_numeric(static_df['pirche_score'], errors='coerce').round(4)

    # Process blood groups
    static_df['blood_group'] = static_df['blood_group'].str.strip()
    static_df['donor_bloodgroup'] = static_df['donor_bloodgroup'].str.strip()
    
    # Convert standalone blood groups to include +
    static_df.loc[static_df['blood_group'].isin(['A', 'B', 'AB', '0']), 'blood_group'] += '+'
    static_df.loc[static_df['donor_bloodgroup'].isin(['A', 'B', 'AB', '0']), 'donor_bloodgroup'] += '+'
    
    # Set invalid blood groups to NaN
    valid_blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', '0+', '0-']
    static_df.loc[~static_df['blood_group'].isin(valid_blood_groups), 'blood_group'] = np.nan
    static_df.loc[~static_df['donor_bloodgroup'].isin(valid_blood_groups), 'donor_bloodgroup'] = np.nan

    return static_df


def create_vitals_df(dfs):
    # expects raw dfs and returns two dfs (vitals from clinical assessments, vitals from lab cohort)

    static_df = create_static_df(dfs)

    vitals_ca = dfs['clinical_assessment'][['PatientID', 'TransplantationID', 'OPDtime', 'Blutdruck_systolisch', 'Blutdruck_diastolisch', 'Gewicht', 'Urinvolumen', 'Herzfrequenz', 'Temperatur', 'Diuresezeit']].rename(
        columns={
            'PatientID': 'patient_id',
            'TransplantationID': 'transplant_id',
            'OPDtime': 'rel_days', # date of assessment in days after transplantation
            'Blutdruck_systolisch': 'bp_sys',
            'Blutdruck_diastolisch': 'bp_dia',
            'Gewicht': 'weight',
            'Urinvolumen': 'urine_volume',  # lower urine volume indicates issues with kidney function
            'Herzfrequenz': 'hr',
            'Temperatur': 'temperature',
            'Diuresezeit': 'diuresis_time',  # duration over which urine output is measured
        }
    )

    # List of features to convert to numeric
    features_to_clean = ['bp_sys', 'bp_dia', 'weight', 'urine_volume', 'hr', 'temperature', 'diuresis_time']

    # Replace commas with dots and convert to numeric for the specified features
    for feature in features_to_clean:
        vitals_ca[feature] = vitals_ca[feature].astype(str).str.replace(',', '.')
        vitals_ca[feature] = pd.to_numeric(vitals_ca[feature], errors='coerce')


    print(f"Unique patients in clinical assessments: {vitals_ca['patient_id'].nunique()}")
    print('Removing patients that are not in static_df')
    vitals_ca = vitals_ca.merge(
        static_df[['patient_id', 'transplant_id']],
        how='inner',  # Inner join to keep only matching entries
        on=['patient_id', 'transplant_id']
    )

    print(f"Unique patients in clinical assessments: {vitals_ca['patient_id'].nunique()}")
    print(f"Average entries per patient {len(vitals_ca) / vitals_ca['patient_id'].nunique()}")


    vitals_lab = dfs['lab_cohort'][['PatientID', 'TransplantationID', 'Labtime', 'Bezeichnung', 'Wert', 'Einheit']].rename(
        columns={
            'PatientID': 'patient_id',
            'TransplantationID': 'transplant_id',
            'Labtime': 'rel_days', # date of assessment in days after transplantation
            'Bezeichnung': 'description',
            'Wert' : 'value',
            'Einheit': 'unit',
        }
    )

    print(f"Unique patients in lab df: {vitals_lab['patient_id'].nunique()}")
    print('Removing patients that are not in static_df')
    vitals_lab = vitals_lab.merge(
        static_df[['patient_id', 'transplant_id']],
        how='inner',  # Inner join to keep only matching entries
        on=['patient_id', 'transplant_id']
    )

    print(f"Unique patients in lab df: {vitals_lab['patient_id'].nunique()}")
    print(f"Average entries per patient {len(vitals_lab) / vitals_lab['patient_id'].nunique()}")

    vitals_ca['rel_days'] = vitals_ca['rel_days'].astype(str).str.replace(',', '.').astype(float).astype(int)
    vitals_lab['rel_days'] = vitals_lab['rel_days'].astype(str).str.replace(',', '.').astype(float).astype(int)

    return vitals_ca, vitals_lab


def create_medication_df(dfs):
    # expects raw dfs, returns medication df

    static_df = create_static_df(dfs)

    medication = dfs['medikation'][['PatientID', 'TransplantationID', 'prescription start', 'prescription end', 'Bezeichnung', 'DDD', 'unit', 'ATC']].rename(
        columns={
            'PatientID': 'patient_id',
            'TransplantationID': 'transplant_id',
            'prescription start': 'p_start',
            'prescription end': 'p_end',
            'Bezeichnung': 'description',
            'DDD': 'ddd', # defined daily dose
            'unit': 'unit',
            'ATC': 'atc' # the ATC column identifies the medication based on its pharmacological classification.
        }
    )

    print(f"Unique patients in medication: {medication['patient_id'].nunique()}")
    print('Removing patients that are not in static_df')
    medication = medication.merge(
        static_df[['patient_id', 'transplant_id', 'transplant_date']],
        how='inner',  # Inner join to keep only matching entries
        on=['patient_id', 'transplant_id']
    )

    # Calculate the relative days from transplantation date to prescription start and end
    medication['start'] = (pd.to_datetime(medication['p_start'], dayfirst=True)-
                                            pd.to_datetime(medication['transplant_date'], dayfirst=True)
                                            ).dt.days
    medication['end'] = (pd.to_datetime(medication['p_end'], dayfirst=True)-
                                            pd.to_datetime(medication['transplant_date'], dayfirst=True)
                                            ).dt.days
    medication = medication.drop(columns=['p_start', 'p_end', 'transplant_date'])
    print(f"Unique patients in clinical assessments: {medication['patient_id'].nunique()}")
    print(f"Average entries per patient {len(medication) / medication['patient_id'].nunique():.1f}")

    medication = medication.dropna(subset=['start', 'end'])
    medication['start'] = medication['start'].astype(str).str.replace(',', '.').astype(float).astype(int)
    medication['end'] = medication['end'].astype(str).str.replace(',', '.').astype(float).astype(int)

    return medication


def create_notes_df(dfs, filename=None):
    # filename indicating where embeddings are stored to load them
    # expects raw dfs and returns notes df with multiple texts per patient
    static_df = create_static_df(dfs)
    print("Reading notes from exams.csv")
    notes = dfs['exams'][['PatientID', 'TransplantationID', 'X', 'Befund', 'Art']].rename(
            columns={
                'PatientID': 'patient_id',
                'TransplantationID': 'transplant_id',
                'X': 'rel_days',
                'Befund': 'text',
                'Art': 'type',
            }
        )
    print(f"Found {len(notes)} texts")

    # Select and rename the relevant columns from clinical_assessment to match the structure of notes
    clinical_assessment_subset = dfs['clinical_assessment'][['PatientID', 'TransplantationID', 'OPDtime', 'BeurteilungAerztlich']].copy()
    clinical_assessment_subset.rename(
        columns={
            'PatientID': 'patient_id',
            'TransplantationID': 'transplant_id',
            'OPDtime': 'rel_days',
            'BeurteilungAerztlich': 'text'
        },
        inplace=True
    )

    clinical_assessment_subset['type'] = 'clinical_assessment'
    print(f"Loading {len(clinical_assessment_subset)} texts from clinical assessments")
    # Append the transformed clinical_assessment data to notes
    notes = pd.concat([notes, clinical_assessment_subset], ignore_index=True)
    notes = notes.dropna(subset=['text'])

    # remove patients not in static df
    notes = notes.merge(
        static_df[['patient_id', 'transplant_id']],
        how='inner',  # Inner join to keep only matching entries
        on=['patient_id', 'transplant_id']
    )

    print(f"Concatenated texts and deleted NaNs, final count: {len(notes)}")
    print(f"Average texts per patient: {len(notes) / len(static_df):.1f}")

    # Replace multiple <br>, ---- and ===== tags with a single whitespace in the notes
    notes['text'] = notes['text'].str.replace(r'(?i)(<br>\s*)+', ' ', regex=True)
    notes['text'] = notes['text'].str.replace(r'(=){2,}', '', regex=True)
    notes['text'] = notes['text'].str.replace(r'[-]{2,}', ' ', regex=True)
    notes['text'] = notes['text'].str.replace(r'\s+', ' ', regex=True)


    if filename is not None:
        loaded_embeddings = np.load(filename, allow_pickle=True)
        notes['embeddings'] = list(loaded_embeddings)

    notes['rel_days'] = notes['rel_days'].astype(str).str.replace(',', '.').astype(float).astype(int)

    return notes


def create_ts_data(vitals_ca, vitals_lab=None, medication=None, merge_lab=True, merge_med=True, static_df=None):
    ts_data = vitals_ca.copy()

    if merge_lab:
        assert static_df is not None

        lab_filtered = vitals_lab[vitals_lab['description'].isin(['KreatininHP', 'LeukoEB', 'CRPHP', 'ProteinCSU', 'AlbuminKSU'])].copy()
        lab_filtered['value'] = pd.to_numeric(lab_filtered['value'].astype(str).str.replace(',', '.'), errors='coerce')
        lab_filtered = lab_filtered.dropna(subset=['value'])
        lab_filtered['unit'] = lab_filtered['unit'].astype(str).str.lower()

        # Convert CRPHP mg/l to mg/dl
        crphp_mask = lab_filtered['description'] == 'CRPHP'
        mg_l_mask = crphp_mask & (lab_filtered['unit'] == 'mg/l')
        lab_filtered.loc[mg_l_mask, 'value'] = lab_filtered.loc[mg_l_mask, 'value'] / 10
        lab_filtered.loc[crphp_mask, 'unit'] = 'mg/dl'

        # Apply filters
        k_mask = (lab_filtered['description'] == 'KreatininHP') & (lab_filtered['unit'].isin(['mg/dl', 'mg/dl'])) & (lab_filtered['value'] <= 50)
        leu_mask = (lab_filtered['description'] == 'LeukoEB') & (lab_filtered['unit'] == '/nl') & (lab_filtered['value'] <= 5000)
        crphp_mask = (lab_filtered['description'] == 'CRPHP') & (lab_filtered['unit'] == 'mg/dl') & (lab_filtered['value'] <= 200)
        protein_mask = (lab_filtered['description'] == 'ProteinCSU') & (lab_filtered['unit'] == 'mg/l') & (lab_filtered['value'] <= 2000)
        alb_mask = (lab_filtered['description'] == 'AlbuminKSU') & (lab_filtered['value'] <= 5000)
        lab_filtered = lab_filtered[k_mask | leu_mask | crphp_mask | protein_mask | alb_mask]

        lab_pivot = lab_filtered.pivot_table(
            index=['patient_id', 'transplant_id', 'rel_days'],
            columns='description',
            values='value',
            aggfunc='first'
        ).reset_index()

        lab_pivot = lab_pivot.rename(columns={'KreatininHP': 'creatinine', 'LeukoEB': 'leukocyte', 'CRPHP': 'crphp', 'ProteinCSU': 'proteinuria', 'AlbuminKSU': 'acr'})
        ts_data = pd.merge(ts_data, lab_pivot, on=['patient_id', 'transplant_id', 'rel_days'], how='outer')

        # merge to get age and gender 
        ts_data = pd.merge(
            ts_data, 
            static_df[['patient_id', 'age', 'gender']], 
            on='patient_id', 
            how='left'
        )

        # Fallback, if age is missing, use 50; if gender is missing, assume 'm'
        ts_data['age'] = ts_data['age'].fillna(50)
        ts_data['gender'] = ts_data['gender'].fillna('m')

        # CALCULATE eGFR, follows formula from https://pmc.ncbi.nlm.nih.gov/articles/PMC3321332/
        # Convert mg/dL to µmol/L
        # Common conversion factor for creatinine: 1 mg/dL ~ 88.4 µmol/L
        ts_data['creatinine_umol'] = ts_data['creatinine'] * 88.4

        # Initialize egfr column as NaN
        ts_data['egfr'] = np.nan

        # Define masks
        female_mask = ts_data['gender'].eq('w')
        male_mask   = ts_data['gender'].eq('m')

        fe_lt62 = female_mask & (ts_data['creatinine_umol'] < 62)
        fe_ge62 = female_mask & (ts_data['creatinine_umol'] >= 62)

        ma_lt80 = male_mask & (ts_data['creatinine_umol'] < 80)
        ma_ge80 = male_mask & (ts_data['creatinine_umol'] >= 80)

        # Formula:
        # Female < 62: eGFR = 144 x (Cr/61.6)^-0.329 x (0.993)^Age
        ts_data.loc[fe_lt62, 'egfr'] = (144 * (ts_data.loc[fe_lt62, 'creatinine_umol'] / 61.6) ** -0.329 * (0.993 ** ts_data.loc[fe_lt62, 'age']))

        # Female >= 62: eGFR = 144 x (Cr/61.6)^-1.209 x (0.993)^Age
        ts_data.loc[fe_ge62, 'egfr'] = (144 * (ts_data.loc[fe_ge62, 'creatinine_umol'] / 61.6) ** -1.209 * (0.993 ** ts_data.loc[fe_ge62, 'age']))

        # Male < 80: eGFR = 141 x (Cr/79.2)^-0.411 x (0.993)^Age
        ts_data.loc[ma_lt80, 'egfr'] = (141 * (ts_data.loc[ma_lt80, 'creatinine_umol'] / 79.2) ** -0.411 * (0.993 ** ts_data.loc[ma_lt80, 'age']))

        # Male >= 80: eGFR = 141 x (Cr/79.2)^-1.209 x (0.993)^Age
        ts_data.loc[ma_ge80, 'egfr'] = (141 * (ts_data.loc[ma_ge80, 'creatinine_umol'] / 79.2) ** -1.209 * (0.993 ** ts_data.loc[ma_ge80, 'age']))

        # If creatinine is NaN, ensure eGFR is NaN
        ts_data.loc[ts_data['creatinine'].isna(), 'egfr'] = np.nan

        # Drop the helper column 
        ts_data.drop(columns='creatinine_umol', inplace=True)

    if merge_med:
        meds_of_interest = ['Tacrolimus', 'Methylprednisolon', 'Ciclosporin']
        meds_filtered = medication[medication['description'].isin(meds_of_interest)].copy()
        meds_filtered = meds_filtered[meds_filtered['unit'] == 'mg']

        meds_filtered['ddd'] = meds_filtered['ddd'].astype(str).str.replace(',', '.')
        meds_filtered['ddd'] = pd.to_numeric(meds_filtered['ddd'], errors='coerce')
        meds_filtered = meds_filtered.dropna(subset=['ddd'])

        # Remove entries where ddd > 2000, these are erroneous rows
        meds_filtered = meds_filtered[meds_filtered['ddd'] <= 2000]

        meds_filtered['rel_days'] = meds_filtered.apply(
            lambda r: range(r['start'], r['end'] + 1), axis=1
        )
        meds_expanded_df = meds_filtered.explode('rel_days').drop(columns=['start', 'end'])

        # Pivot to get medications in columns
        meds_pivot = meds_expanded_df.pivot_table(
            index=['patient_id', 'transplant_id', 'rel_days'],
            columns='description',
            values='ddd',
            aggfunc='sum'
        ).reset_index()

        ts_data = pd.merge(
            ts_data,
            meds_pivot,
            on=['patient_id', 'transplant_id', 'rel_days'],
            how='left'
        )
        ts_data[meds_of_interest] = ts_data[meds_of_interest].fillna(0)

    # Sorting
    ts_data = ts_data.sort_values(by=['patient_id', 'rel_days']).reset_index(drop=True)
    return ts_data


def get_valid_patient_ids(static_df: pd.DataFrame, ts_data: pd.DataFrame, notes_df: pd.DataFrame, min_ts_count: int = 10, require_notes: bool = True) -> np.ndarray:
    ordered_patient_ids = static_df['patient_id'].dropna().drop_duplicates().tolist()
    ts_counts = ts_data.groupby('patient_id').size().to_dict()
    notes_counts = notes_df.groupby('patient_id').size().to_dict()

    valid_patient_ids = []
    for patient_id in ordered_patient_ids:
        if ts_counts.get(patient_id, 0) < min_ts_count:
            continue
        if require_notes and notes_counts.get(patient_id, 0) == 0:
            continue
        valid_patient_ids.append(patient_id)

    return np.asarray(valid_patient_ids)


def split_patient_ids(patient_ids: Sequence, train_size: float = 0.8, val_size: float = 0.0, test_size: float = 0.2, random_state: int = 42, shuffle: bool = True) -> dict[str, np.ndarray]:
    patient_ids = np.asarray(patient_ids)
    if patient_ids.ndim != 1:
        raise ValueError("patient_ids must be one-dimensional")
    if len(patient_ids) == 0:
        raise ValueError("No patient_ids available for splitting")

    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    split_ids = patient_ids.copy()
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(split_ids)

    n_total = len(split_ids)
    n_test = 0 if test_size == 0 else max(1, int(round(test_size * n_total)))
    n_val = 0 if val_size == 0 else max(1, int(round(val_size * n_total)))
    if n_test + n_val >= n_total:
        raise ValueError("Split sizes leave no patients for the training set")

    n_train = n_total - n_val - n_test

    train_ids = split_ids[:n_train]
    val_ids = split_ids[n_train:n_train + n_val]
    test_ids = split_ids[n_train + n_val:]

    return {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids,
    }


# Persisted-split helper from the main branch. Kept alongside the
# patient_split_ids argument of create_dataset_splits: the notebooks use this
# to pin one split across runs, while the training/evaluation scripts pass
# splits in explicitly. The two are complementary, not alternatives.
def get_or_create_global_split(
    patient_ids: Sequence,
    split_json_path: str,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
    force_recreate: bool = False,
) -> dict[str, np.ndarray]:
    """
    Load one persisted split from disk, or create and save it if missing.
    This guarantees consistent train/val/test IDs across notebooks.
    """
    patient_ids = np.asarray(patient_ids)
    if patient_ids.ndim != 1:
        raise ValueError("patient_ids must be one-dimensional")
    if len(patient_ids) == 0:
        raise ValueError("No patient_ids available for splitting")

    split_json_path = os.path.abspath(split_json_path)
    split_dir = os.path.dirname(split_json_path)

    expected_ids = set(patient_ids.tolist())

    def _validate_loaded_split(loaded_split: dict[str, np.ndarray]) -> None:
        train_ids = set(loaded_split['train'].tolist())
        val_ids = set(loaded_split['val'].tolist())
        test_ids = set(loaded_split['test'].tolist())

        if not train_ids.isdisjoint(val_ids):
            raise ValueError("Persisted split is invalid: train overlaps val")
        if not train_ids.isdisjoint(test_ids):
            raise ValueError("Persisted split is invalid: train overlaps test")
        if not val_ids.isdisjoint(test_ids):
            raise ValueError("Persisted split is invalid: val overlaps test")

        persisted_union = train_ids | val_ids | test_ids
        if persisted_union != expected_ids:
            missing_ids = expected_ids - persisted_union
            extra_ids = persisted_union - expected_ids
            raise ValueError(
                "Persisted split IDs do not match current selected cohort. "
                f"missing={len(missing_ids)}, extra={len(extra_ids)}. "
                "Set force_recreate=True to regenerate this split file."
            )

    if os.path.exists(split_json_path) and not force_recreate:
        with open(split_json_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        for key in ('train', 'val', 'test'):
            if key not in payload:
                raise ValueError(f"Persisted split file is missing key: {key}")

        loaded_split = {
            'train': np.asarray(payload['train']),
            'val': np.asarray(payload['val']),
            'test': np.asarray(payload['test']),
        }
        _validate_loaded_split(loaded_split)
        return loaded_split

    split_ids = split_patient_ids(
        patient_ids=patient_ids,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    payload = {
        'train': split_ids['train'].tolist(),
        'val': split_ids['val'].tolist(),
        'test': split_ids['test'].tolist(),
        'meta': {
            'train_size': float(train_size),
            'val_size': float(val_size),
            'test_size': float(test_size),
            'random_state': int(random_state),
            'shuffle': bool(shuffle),
            'n_selected_patients': int(len(patient_ids)),
        },
    }
    if split_dir:
        os.makedirs(split_dir, exist_ok=True)
    with open(split_json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    return split_ids


def create_dataset_splits(
    static_df: pd.DataFrame,
    ts_data: pd.DataFrame,
    notes_df: pd.DataFrame,
    biopsy_df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.0,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
    max_patients: Optional[int] = None,
    patient_ids: Optional[Sequence] = None,
    patient_split_ids: Optional[Mapping[str, Sequence]] = None,
    preprocessing_artifacts: Optional[PreprocessingArtifacts] = None,
    min_ts_count: int = 10,
    require_notes: bool = True,
) -> dict[str, object]:
    eligible_patient_ids = get_valid_patient_ids(
        static_df=static_df,
        ts_data=ts_data,
        notes_df=notes_df,
        min_ts_count=min_ts_count,
        require_notes=require_notes,
    )

    if patient_ids is not None:
        eligible_patient_id_set = set(eligible_patient_ids)
        eligible_patient_ids = np.asarray([pid for pid in patient_ids if pid in eligible_patient_id_set])

    if patient_split_ids is not None:
        eligible_patient_id_set = set(eligible_patient_ids)
        selected_patient_ids = patient_split_ids.get('selected', eligible_patient_ids)
        eligible_patient_ids = np.asarray([pid for pid in selected_patient_ids if pid in eligible_patient_id_set])
        split_ids = {
            split_name: np.asarray([pid for pid in patient_split_ids.get(split_name, []) if pid in eligible_patient_id_set])
            for split_name in ('train', 'val', 'test')
        }
    elif max_patients is not None and len(eligible_patient_ids) > max_patients:
        if shuffle:
            rng = np.random.default_rng(random_state)
            eligible_patient_ids = eligible_patient_ids.copy()
            rng.shuffle(eligible_patient_ids)
        eligible_patient_ids = eligible_patient_ids[:max_patients]
        split_ids = split_patient_ids(
            patient_ids=eligible_patient_ids,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
    else:
        split_ids = split_patient_ids(
            patient_ids=eligible_patient_ids,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

    fit_preprocessing = preprocessing_artifacts is None

    train_dataset = NephroCAGEDataset(
        static_df=static_df,
        ts_data=ts_data,
        notes_df=notes_df,
        biopsy_df=biopsy_df,
        patient_ids=split_ids['train'],
        preprocessing_artifacts=preprocessing_artifacts,
        fit_preprocessing=fit_preprocessing,
        min_ts_count=min_ts_count,
        require_notes=require_notes,
    )
    if preprocessing_artifacts is None:
        preprocessing_artifacts = train_dataset.preprocessing_artifacts

    val_dataset = None
    if len(split_ids['val']) > 0:
        val_dataset = NephroCAGEDataset(
            static_df=static_df,
            ts_data=ts_data,
            notes_df=notes_df,
            biopsy_df=biopsy_df,
            patient_ids=split_ids['val'],
            preprocessing_artifacts=preprocessing_artifacts,
            fit_preprocessing=False,
            min_ts_count=min_ts_count,
            require_notes=require_notes,
        )

    test_dataset = None
    if len(split_ids['test']) > 0:
        test_dataset = NephroCAGEDataset(
            static_df=static_df,
            ts_data=ts_data,
            notes_df=notes_df,
            biopsy_df=biopsy_df,
            patient_ids=split_ids['test'],
            preprocessing_artifacts=preprocessing_artifacts,
            fit_preprocessing=False,
            min_ts_count=min_ts_count,
            require_notes=require_notes,
        )

    full_dataset = NephroCAGEDataset(
        static_df=static_df,
        ts_data=ts_data,
        notes_df=notes_df,
        biopsy_df=biopsy_df,
        patient_ids=eligible_patient_ids,
        preprocessing_artifacts=preprocessing_artifacts,
        fit_preprocessing=False,
        min_ts_count=min_ts_count,
        require_notes=require_notes,
    )

    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'full': full_dataset,
        'patient_ids': {
            'selected': eligible_patient_ids,
            **split_ids,
        },
        'preprocessing_artifacts': preprocessing_artifacts,
    }


class NephroCAGEDataset(Dataset):
    def __init__(
        self,
        static_df: pd.DataFrame,
        ts_data: pd.DataFrame,
        notes_df: pd.DataFrame,
        biopsy_df: pd.DataFrame,
        patient_ids: Optional[Sequence] = None,
        preprocessing_artifacts: Optional[PreprocessingArtifacts] = None,
        fit_preprocessing: bool = True,
        min_ts_count: int = 10,
        require_notes: bool = True,
    ):
        if not fit_preprocessing and preprocessing_artifacts is None:
            raise ValueError("preprocessing_artifacts must be provided when fit_preprocessing is False")

        self.static_df = static_df.copy()
        self.ts_data = ts_data.copy()
        self.notes_df = notes_df.copy()  # Included as a separate dataframe

        # create graft loss labels
        self.labels = self.static_df[['patient_id', 'loss_rel_days', 'death_rel_days']].copy()
        self.labels['graft_loss_label'] = self.labels['loss_rel_days'].notna().astype(int)
        self.labels['death_label'] = self.labels['death_rel_days'].notna().astype(int)

        # Compute rejection label by BANFF categories 2 and 4
        biopsy_df['Banff 17 categorie'] = pd.to_numeric(biopsy_df['Banff 17 categorie'], errors='coerce')
        rejection_rows = biopsy_df[biopsy_df['Banff 17 categorie'].isin([2, 4])].copy()
        grouped_rejections = (rejection_rows.groupby('PatientID')['BXtime'].apply(list).reset_index(name='rej_rel_days_list'))
        
        # Convert to dictionary: { patient_id: [day1, day2, ...] }
        self.rej_dict = dict(zip(grouped_rejections['PatientID'], grouped_rejections['rej_rel_days_list']))

        valid_patient_ids = get_valid_patient_ids(
            static_df=self.static_df,
            ts_data=self.ts_data,
            notes_df=self.notes_df,
            min_ts_count=min_ts_count,
            require_notes=require_notes,
        )

        if patient_ids is not None:
            valid_patient_id_set = set(valid_patient_ids)
            valid_patient_ids = np.asarray([pid for pid in patient_ids if pid in valid_patient_id_set])

        # Filter self.static_df, self.labels, and self.ts_data to include only valid_patient_ids
        self.static_df = self.static_df[self.static_df['patient_id'].isin(valid_patient_ids)].copy()
        self.labels = self.labels[self.labels['patient_id'].isin(valid_patient_ids)].copy()
        self.ts_data = self.ts_data[self.ts_data['patient_id'].isin(valid_patient_ids)].copy()
        self.notes_df = self.notes_df[self.notes_df['patient_id'].isin(valid_patient_ids)].copy()

        if fit_preprocessing:
            self.preprocessing_artifacts = self._fit_preprocessing()
        else:
            self.preprocessing_artifacts = preprocessing_artifacts

        self._apply_preprocessing(self.preprocessing_artifacts)

        self.label_encoders = self.preprocessing_artifacts.label_encoders
        self.categorical_cardinalities = self.preprocessing_artifacts.categorical_cardinalities
        self.scaler = self.preprocessing_artifacts.static_scaler
        self.ts_scaler = self.preprocessing_artifacts.ts_scaler

        # Preserve the explicit split order so dataloader row indices are stable.
        self.patient_ids = np.asarray(valid_patient_ids).astype(int)
        if len(self.patient_ids) != len(self.static_df):
            raise ValueError("Duplicate patients in static df")

    def _fit_preprocessing(self) -> PreprocessingArtifacts:
        static_train = self.static_df.copy()
        for col in CONFIG['static_numerical_cols']:
            static_train[col] = static_train[col].fillna(-1)
        for col in CONFIG['static_categorical_cols']:
            static_train[col] = static_train[col].fillna('Unknown').astype(str)

        label_encoders = {}
        for col in CONFIG['static_categorical_cols']:
            le = LabelEncoder()
            train_values = static_train[col]
            if 'Unknown' not in set(train_values):
                train_values = pd.concat([train_values, pd.Series(['Unknown'])], ignore_index=True)
            le.fit(train_values)
            label_encoders[col] = le

        static_scaler = StandardScaler()
        static_scaler.fit(static_train[CONFIG['static_numerical_cols']])

        # Fit the time-series scaler on the raw train values so NaNs do not distort the moments.
        ts_scaler = StandardScaler()
        ts_scaler.fit(self.ts_data[CONFIG['ts_features']])

        categorical_cardinalities = [
            len(label_encoders[col].classes_) for col in CONFIG['static_categorical_cols']
        ]

        return PreprocessingArtifacts(
            label_encoders=label_encoders,
            categorical_cardinalities=categorical_cardinalities,
            static_scaler=static_scaler,
            ts_scaler=ts_scaler,
        )

    def _apply_preprocessing(self, preprocessing_artifacts: PreprocessingArtifacts) -> None:
        for col in CONFIG['static_numerical_cols']:
            self.static_df[col] = self.static_df[col].fillna(-1)
        for col in CONFIG['static_categorical_cols']:
            self.static_df[col] = self.static_df[col].fillna('Unknown').astype(str)

        for col, label_encoder in preprocessing_artifacts.label_encoders.items():
            known_classes = set(label_encoder.classes_)
            values = self.static_df[col].where(self.static_df[col].isin(known_classes), 'Unknown')
            self.static_df[col] = label_encoder.transform(values)

        static_scaled = pd.DataFrame(
            preprocessing_artifacts.static_scaler.transform(self.static_df[CONFIG['static_numerical_cols']]),
            columns=CONFIG['static_numerical_cols'],
            index=self.static_df.index,
        )
        self.static_df[CONFIG['static_numerical_cols']] = static_scaled

        # Track real observed values before replacing missing entries with the padding sentinel.
        self.ts_data_value_mask = self.ts_data[['patient_id', 'rel_days']].copy()
        self.ts_data_value_mask[CONFIG['ts_features']] = (~self.ts_data[CONFIG['ts_features']].isna()).astype(int)

        ts_scaled = pd.DataFrame(
            preprocessing_artifacts.ts_scaler.transform(self.ts_data[CONFIG['ts_features']]),
            columns=CONFIG['ts_features'],
            index=self.ts_data.index,
        )
        self.ts_data[CONFIG['ts_features']] = ts_scaled.fillna(CONFIG['PADDING_VAL'])

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # Static Features
        static_row = self.static_df[self.static_df['patient_id'] == patient_id]
        if static_row.empty:
            raise ValueError(f"No static data found for patient_id: {patient_id}")
        static_features = static_row.drop(columns=['patient_id']).iloc[0]

        # Split categorical and numerical features
        categorical_features = torch.tensor(static_features[CONFIG['static_categorical_cols']].values.astype(np.int64))
        numerical_features = torch.tensor(static_features[CONFIG['static_numerical_cols']].values.astype(np.float32))

        # time-series feature tensor
        ts_data = self.ts_data[self.ts_data['patient_id'] == patient_id]
        ts_features = torch.tensor(ts_data[CONFIG['ts_features']].values.astype(np.float32))
        timesteps = torch.tensor(ts_data['rel_days'].values.astype(np.int64))

        # Value mask for time-series data
        value_mask = self.ts_data_value_mask[self.ts_data_value_mask['patient_id'] == patient_id]
        value_mask = torch.tensor(value_mask[CONFIG['ts_features']].values.astype(np.int64))

        seq_len = ts_features.size(0)

        # Handle notes embeddings and timesteps
        notes_rows = self.notes_df[self.notes_df['patient_id'] == patient_id]
        notes_timesteps = torch.tensor(notes_rows['rel_days'].values.astype(np.int64))
        if 'embeddings' in self.notes_df.columns:
            if notes_rows.empty:
                notes_embeddings = torch.empty((0, CONFIG['notes_embedding_dim']), dtype=torch.float32)
            else:
                notes_embeddings = torch.tensor(np.stack(notes_rows['embeddings'].values), dtype=torch.float32)
        else:
            notes_embeddings = None

        # get labels
        labels = self.labels[self.labels['patient_id'] == patient_id]
        graft_loss_label = torch.tensor(labels['graft_loss_label'].values.astype(np.float32))
        death_label = torch.tensor(labels['death_label'].values.astype(np.float32))
        loss_rel_days = torch.tensor(labels['loss_rel_days'].values.astype(np.float32))
        death_rel_days = torch.tensor(labels['death_rel_days'].values.astype(np.float32))

        # rejection labels
        rej_rel_days_list = self.rej_dict.get(patient_id, [])
        if len(rej_rel_days_list) == 0:
            rej_rel_days_list = None

        sample = {
            'patient_id': patient_id,
            'static_categorical_features': categorical_features,
            'static_numerical_features': numerical_features,
            'ts_features': ts_features,
            'timesteps': timesteps,
            'value_mask': value_mask, # indicates missing values in the time series data
            'graft_loss_label': graft_loss_label,
            'loss_rel_days': loss_rel_days,
            'death_label': death_label,
            'death_rel_days': death_rel_days,
            'rej_rel_days': rej_rel_days_list,  # list of days of rejection
            'seq_len': seq_len,  # sequence length of time series data for this patient
            'notes_embeddings': notes_embeddings,  # variable length per patient around 10-100 embedded notes
            'notes_timesteps': notes_timesteps,  # corresponding to notes relative days since transplant
        }

        return sample


def collate_fn(batch):
    collated_batch = {}

    # Collect sequence lengths
    seq_lengths = torch.tensor([item['seq_len'] for item in batch], dtype=torch.long)
    # Check if note embeddings are available anywhere in the batch. When the
    # embeddings column exists, patients without notes contribute empty tensors.
    has_notes = any(item['notes_embeddings'] is not None for item in batch)
    if has_notes:
        notes_emb_list = [
            item['notes_embeddings']
            if item['notes_embeddings'] is not None
            else torch.empty((0, CONFIG['notes_embedding_dim']), dtype=torch.float32)
            for item in batch
        ]
        notes_timesteps_list = [
            item['notes_timesteps']
            if item['notes_embeddings'] is not None
            else torch.empty((0,), dtype=torch.long)
            for item in batch
        ]
        notes_lengths = torch.tensor([item.shape[0] for item in notes_emb_list], dtype=torch.long)
        max_notes_len = notes_lengths.max().item()
    else:
        notes_emb_list = None
        notes_timesteps_list = None
        notes_lengths = None
        max_notes_len = None

    # Find the maximum sequence length in the batch
    max_seq_len = seq_lengths.max().item()

    for key in batch[0].keys():
        data = [item[key] for item in batch]

        if key in ['ts_features', 'timesteps', 'value_mask']:
            # Pad 'ts_features', 'timsteps', and 'value_mask' to make all sequences the same length
            padding_value = CONFIG['PADDING_VAL'] if key != 'value_mask' else 0
            collated_batch[key] = pad_sequence(data, batch_first=True, padding_value=padding_value)
        elif key == 'notes_embeddings':
            if has_notes:
                collated_batch[key] = pad_sequence(notes_emb_list, batch_first=True, padding_value=0)
            else:
                collated_batch[key] = None
        elif key == 'notes_timesteps':
            if has_notes:
                collated_batch[key] = pad_sequence(notes_timesteps_list, batch_first=True, padding_value=0)
            else:
                collated_batch[key] = None
        elif key == 'seq_len':
            # Already collected sequence lengths
            collated_batch[key] = seq_lengths
        elif all(isinstance(d, torch.Tensor) and d.shape == data[0].shape for d in data):
            # Stack features that have consistent shapes
            collated_batch[key] = torch.stack(data)
        else:
            # Collect non-tensor or variable-length data as-is
            collated_batch[key] = data

    # Create the mask based on sequence lengths
    batch_size = len(seq_lengths)
    mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len) < seq_lengths.unsqueeze(1)
    collated_batch['mask'] = mask  # Shape: (batch_size, max_seq_len)

    # Create the mask for notes embeddings
    if has_notes:
        notes_mask = torch.arange(max_notes_len).expand(batch_size, max_notes_len) < notes_lengths.unsqueeze(1)
        collated_batch['notes_mask'] = notes_mask  # Shape: (batch_size, max_notes_len)
    else:
        collated_batch['notes_embeddings'] = torch.empty((batch_size, 0, CONFIG['notes_embedding_dim']), dtype=torch.float32)
        collated_batch['notes_timesteps'] = torch.empty((batch_size, 0), dtype=torch.long)
        collated_batch['notes_mask'] = torch.zeros((batch_size, 0), dtype=torch.bool)

    return collated_batch
