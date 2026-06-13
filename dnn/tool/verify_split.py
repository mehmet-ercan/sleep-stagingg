#!/usr/bin/env python3
"""Subject-level split leakage verification script."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dnn'))

import config
from dataset import get_subject_id, group_by_subject, subjects_to_recordings
from edf_to_mongo import get_offline_patient_ids
from sklearn.model_selection import KFold
import random

# Load all patients
all_patient_ids = get_offline_patient_ids()
print(f"Total recordings: {len(all_patient_ids)}")

# Subject grouping
subject_map = group_by_subject(all_patient_ids)
subject_list = list(subject_map.keys())
print(f"Total subjects: {len(subject_list)}")
print(f"Subjects with 2 nights: {sum(1 for s,r in subject_map.items() if len(r)==2)}")
print(f"Subjects with 1 night: {sum(1 for s,r in subject_map.items() if len(r)==1)}")

# Simulate K-Fold split
random.seed(config.RANDOM_SEED)
random.shuffle(subject_list)

n_test = config.N_TEST_PATIENTS_KFOLD
test_subjects = subject_list[:n_test]
cv_subjects = subject_list[n_test:]
test_recordings = subjects_to_recordings(test_subjects, subject_map)

print(f"\nTest: {n_test} subjects -> {len(test_recordings)} recordings")
print(f"CV: {len(cv_subjects)} subjects")

kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

total_leaks = 0
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(cv_subjects)):
    train_subjs = [cv_subjects[i] for i in train_idx]
    val_subjs = [cv_subjects[i] for i in val_idx]
    
    train_recs = subjects_to_recordings(train_subjs, subject_map)
    val_recs = subjects_to_recordings(val_subjs, subject_map)
    
    # Check subject-level leakage
    train_s = set(get_subject_id(r) for r in train_recs)
    val_s = set(get_subject_id(r) for r in val_recs)
    test_s = set(get_subject_id(r) for r in test_recordings)
    
    tv_leak = train_s & val_s
    tt_leak = train_s & test_s
    vt_leak = val_s & test_s
    
    leak_count = len(tv_leak) + len(tt_leak) + len(vt_leak)
    total_leaks += leak_count
    
    status = "CLEAN" if leak_count == 0 else f"LEAKS: {leak_count}"
    print(f"Fold {fold_idx}: train={len(train_subjs)}s ({len(train_recs)}r), "
          f"val={len(val_subjs)}s ({len(val_recs)}r) -> {status}")

print(f"\n{'='*50}")
print(f"Total subject leakage across all folds: {total_leaks}")
if total_leaks == 0:
    print("RESULT: ALL CLEAN - NO LEAKAGE!")
else:
    print("RESULT: LEAKAGE DETECTED!")
