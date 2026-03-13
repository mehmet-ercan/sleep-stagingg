#!/usr/bin/env python
# coding: utf-8

# # IMPORTS

# In[2]:


import mne
import numpy as np
import pandas as pd
import shap
import datetime
import uuid
import os
import seaborn as sns
import pyedflib

from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix



# # SOME DEFINES

# In[ ]:


def log(*args, **kwargs):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]", *args, **kwargs)


# In[ ]:


PLOTS_DIR = "plots"
MEMMAP_DATAS = "memmap_datas"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MEMMAP_DATAS, exist_ok=True)


# # REAL PATIENT DATA LOADING

# In[ ]:


edf_annotation_pairs = [
    ("/home/user/sleep-stage/4.uncompressed_data/1.first_phase/2d42885b-ba49-4c21-9e8f-4963349f252c/Traces.edf",
     "/home/user/sleep-stage/4.uncompressed_data/1.first_phase/2d42885b-ba49-4c21-9e8f-4963349f252c/Traces_annotations.edf"
     ),
     ("/home/user/sleep-stage/4.uncompressed_data/1.first_phase/3da2681a-5991-449c-80a4-52cf3257db02/Traces.edf",
     "/home/user/sleep-stage/4.uncompressed_data/1.first_phase/3da2681a-5991-449c-80a4-52cf3257db02/Traces_annotations.edf"
     ),
     ("/home/user/sleep-stage/4.uncompressed_data/1.first_phase/4f680835-27cb-422f-b254-1d2994ff548a/Traces.edf",
     "/home/user/sleep-stage/4.uncompressed_data/1.first_phase/4f680835-27cb-422f-b254-1d2994ff548a/Traces_annotations.edf"
     ),
     ("/home/user/sleep-stage/4.uncompressed_data/1.first_phase/6b19b899-e4b2-4465-93d7-ab2c329b092d/Traces.edf",
     "/home/user/sleep-stage/4.uncompressed_data/1.first_phase/6b19b899-e4b2-4465-93d7-ab2c329b092d/Traces_annotations.edf"
     )
    ]


# # MEMMAP VERSION DATA LOADING

# In[ ]:


def extract_features(epoch, channel_names):
    """Her epoch için özellik çıkaran fonksiyon"""
    features = {}
    stats = {
        "mean": np.mean(epoch, axis=1),
        "std": np.std(epoch, axis=1),
        "skewness": skew(epoch, axis=1),
        "kurtosis": kurtosis(epoch, axis=1),
        "power": np.sum(epoch**2, axis=1),
    }
    
    for stat_name, values in stats.items():
        for ch_idx, value in enumerate(values):
            features[f"{stat_name}_{channel_names[ch_idx]}"] = value
    
    return features


#feature_list = [extract_features(epoch.copy()) for epoch in X_epochs] # .copy() for speed


# In[ ]:


def extract_features_floaat16_approach(epoch, channel_names):
    """Her epoch için özellik çıkaran fonksiyon"""
    features = {}
    # Clip epoch values to avoid extreme values during computation
    epoch = np.clip(epoch, -1e10, 1e10)  # Arbitrary large range, adjust as needed
    
    stats = {
        "mean": np.mean(epoch, axis=1),
        "std": np.std(epoch, axis=1),
        "skewness": skew(epoch, axis=1, nan_policy='omit'),  # Handle NaNs/infs
        "kurtosis": kurtosis(epoch, axis=1, nan_policy='omit'),  # Handle NaNs/infs
        "power": np.sum(epoch**2, axis=1),
    }
    
    for stat_name, values in stats.items():
        # Replace NaNs and infs with a large finite value
        values = np.nan_to_num(values, nan=0.0, posinf=np.finfo(np.float16).max, neginf=-np.finfo(np.float16).max)
        # Clip values to float16 range
        values = np.clip(values, -np.finfo(np.float16).max, np.finfo(np.float16).max)
        for ch_idx, value in enumerate(values):
            features[f"{stat_name}_{channel_names[ch_idx]}"] = np.float16(value)
    
    return features


# In[ ]:


epoch_length = 30  # seconds
all_feature_dicts = []

for person_id, (edf_path, annot_path) in enumerate(edf_annotation_pairs):
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    annotations = mne.read_annotations(annot_path)
    raw.set_annotations(annotations)

    sfreq = raw.info['sfreq']
    samples_per_epoch = int(epoch_length * sfreq)
    n_epochs = len(annotations)
    n_channels = len(raw.ch_names)
    channel_names = raw.info['ch_names']

    # Use per-subject memmap to avoid RAM issues
    mmap_filename = f"epochs_person_{person_id}_{uuid.uuid4().hex}.dat"
    X_epochs = np.memmap(mmap_filename, dtype='float16', mode='w+',
                         shape=(n_epochs, n_channels, samples_per_epoch))

    for i in range(n_epochs):
        start_sample = i * samples_per_epoch
        end_sample = start_sample + samples_per_epoch
        if end_sample <= raw.n_times:
            epoch = raw.get_data(start=start_sample, stop=end_sample)
            X_epochs[i] = epoch
            feat = extract_features(epoch, channel_names)#float32 for more sensitivity
            # feat = extract_features_floaat16_approach(epoch, channel_names) #float16 for less sensitivity but effective ram usage
            feat["sleep_stage"] = annotations[i]['description']
            feat["subject_id"] = person_id
            all_feature_dicts.append(feat)

    X_epochs.flush()  # Save to disk
    


# In[ ]:


df_features = pd.DataFrame(all_feature_dicts)


# In[ ]:


# NaN sayısını al
nan_counts = df_features.isna().sum()

# 50 ve üzeri NaN olan sütunlar → tamamen silinecek
cols_to_drop = nan_counts[nan_counts >= 50].index.tolist()

# 50'den az NaN olan sütunlar → ilgili satırlar silinecek
cols_to_filter = nan_counts[(nan_counts > 0) & (nan_counts < 50)].index.tolist()

log("Tamamen silinecek sütunlar:", cols_to_drop)
log("Satırları silinecek sütunlar:", cols_to_filter)

df_cleaned = df_features.dropna(subset=cols_to_filter) # Önce satırlardan NaN olanları sil
df_cleaned = df_cleaned.drop(columns=cols_to_drop) # Sonra tamamen silinecek sütunları sil

log("\nTemizlenmiş DataFrame:")
log(df_cleaned)

extra_columns = ["sleep_stage", "subject_id"]


# In[ ]:


X = df_cleaned.drop(columns=extra_columns)
y = df_cleaned["sleep_stage"]

# Burada n_splits > 1 olmak zorunda. Dolayısıyla 2 değeri eğitimi uzattığı için 1 değeri için manuel yazdım.
# from sklearn.model_selection import TimeSeriesSplit
# tscv = TimeSeriesSplit(n_splits=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

n = len(X)
train_size = int(n * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]


# # SHAP WITH RANDOM FOREST CLASSIFIER

# In[ ]:


clf_random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
clf_random_forest.fit(X_train, y_train)


# In[ ]:


y_pred = clf_random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
log(f"Test doğruluğu: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred, labels=clf_random_forest.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_random_forest.classes_)
disp.plot(cmap='Blues')
plt.show()

log(classification_report(y_test, y_pred, target_names=clf_random_forest.classes_))

scores = cross_val_score(clf_random_forest, X_train, y_train, cv=5)
log(f"CV Doğruluk Ortalaması: {scores.mean():.2f} ± {scores.std():.2f}")

importances = clf_random_forest.feature_importances_
feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"feat_{i}" for i in range(X_train.shape[1])]
feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

log(feature_importance_df.head(10))


# In[ ]:


explainer = shap.Explainer(clf_random_forest, X_train)
shap_values = explainer(X_test, check_additivity=False)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(f'{PLOTS_DIR}/shap_rfs_features_importance.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:


log("NaNs in X_train:", np.any(np.isnan(X_train)))
log("Infinities in X_train:", np.any(np.isinf(X_train)))
log("Max value in X_train:", np.max(np.abs(X_train)))


# In[ ]:


mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0, 2))  # shape: (features,)

feature_names = X_test.columns
channel_names = [name.split('_', 1)[1] for name in feature_names]  # assuming 'power_C3' → 'C3'

channel_importance = pd.DataFrame({'channel': channel_names, 'shap_value': mean_abs_shap})
channel_summary = channel_importance.groupby('channel').sum().sort_values(by='shap_value', ascending=False)

channel_summary.plot(kind='bar')
plt.title("Channel Importance (Aggregated SHAP Values - Random Forest)")
plt.ylabel("Mean |SHAP Value|")
plt.xlabel("Channel")

plt.savefig(f'{PLOTS_DIR}/shap_rfs_channels_importance.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:


top_channels = channel_summary.head(7)
log(top_channels)


# # SHAP WITH SVM

# In[ ]:


clf_svm = LinearSVC()
clf_svm.fit(X_train, y_train)

explainer = shap.LinearExplainer(clf_svm, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
class_labels = clf_svm.classes_  # ['Wake', 'N1', 'N2', 'N3', 'REM']
plt.legend(handles, class_labels, title="Sleep Stages - Lineer SVM")
plt.savefig(f'{PLOTS_DIR}/shap_svm_features_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# TODO Her bir feature için, stage oranlarının fazla olduğu ikili, channel olarak seçilebilir.


# In[ ]:


mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0, 2))  # shape: (features,)

feature_names = X_test.columns
channel_names = [name.split('_', 1)[1] for name in feature_names]  # assuming 'power_C3' → 'C3'

channel_importance = pd.DataFrame({'channel': channel_names, 'shap_value': mean_abs_shap})
channel_summary = channel_importance.groupby('channel').sum().sort_values(by='shap_value', ascending=False)

channel_summary.plot(kind='bar')
plt.title("Channel Importance (Aggregated SHAP Values - Lineer SVM)")
plt.ylabel("Mean |SHAP Value|")
plt.xlabel("Channel")
plt.figure().set_figwidth(15)
plt.savefig(f'{PLOTS_DIR}/shap_svm_channels_importance.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:


log([name for name in feature_names])


# In[ ]:


top_channels = channel_summary.head(7)
log(top_channels)


# # SEQUENTIAL FORWARD (ÇOK UZUN SÜRDÜĞÜ İÇİN BYPASS EDİLDİ)

# In[ ]:


""" tscv = TimeSeriesSplit(n_splits=5)
clf_random_forest_for_seq = RandomForestClassifier(n_estimators=100, random_state=42)

# Eğer sleep_stage çok sınıflı bir problemse,scoring='f1_weighted' gibi bir metrik de daha uygun olabilir.
sfs = SFS(clf_random_forest_for_seq,
          k_features=15, # 'best'
          forward=True,
          floating=False,
          scoring='accuracy', #'f1_weighted'
          cv=tscv, # <- zaman serisine uygun cv
          n_jobs=-1)

sfs = sfs.fit(X_train, y_train)

selected_features = list(sfs.k_feature_names_)
log("Seçilen Özellikler (TimeSeries SForwardS):", selected_features)

# Performans değerlendirme (X_test için aynı feature subset)
clf_random_forest_for_seq.fit(X_train[selected_features], y_train)
y_pred = clf_random_forest_for_seq.predict(X_test[selected_features])
log("Test Doğruluğu (SForwardS):", accuracy_score(y_test, y_pred))
 """


# # SEQUENTIAL BACKWARD (ÇOK UZUN SÜRDÜĞÜ İÇİN BYPASS EDİLDİ)

# In[ ]:


""" tscv = TimeSeriesSplit(n_splits=5)
clf_random_forest_for_bac = RandomForestClassifier(n_estimators=100, random_state=42)

# Eğer sleep_stage çok sınıflı bir problemse,scoring='f1_weighted' gibi bir metrik de daha uygun olabilir.
sfs = SFS(clf_random_forest_for_bac,
          k_features=15, # 'best'
          forward=False,
          floating=False,
          scoring='accuracy', #'f1_weighted'
          cv=tscv, # <- zaman serisine uygun cv
          n_jobs=-1)

sfs = sfs.fit(X_train, y_train)

selected_features = list(sfs.k_feature_names_)
log("Seçilen Özellikler (TimeSeries SBackwardS):", selected_features)

# Performans değerlendirme (X_test için aynı feature subset)
clf_random_forest_for_bac.fit(X_train[selected_features], y_train)
y_pred = clf_random_forest_for_bac.predict(X_test[selected_features])
log("Test Doğruluğu (SBackwardS):", accuracy_score(y_test, y_pred))
 """


# # PERMUTATION IMPORTANCE

# In[ ]:


log("Top 10 Most Important Features Started:")
clf_random_forest_for_per = RandomForestClassifier(n_estimators=100, random_state=42)
clf_random_forest_for_per.fit(X_train, y_train)

result = permutation_importance(clf_random_forest_for_per, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()[::-1]  # En önemliden en az önemliye
feature_names = np.array(X_test.columns)

log("\n Top 10 Most Important Features (Permutation Importance):\n")
for idx in sorted_idx[:10]:
    log(f"{feature_names[idx]:<25} | Mean: {result.importances_mean[idx]:.4f} | Std: {result.importances_std[idx]:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx[::-1]], align="center")
plt.yticks(np.arange(len(sorted_idx)), feature_names[sorted_idx[::-1]])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance (Permutation)")
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/permutation_importance_features_v1.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:


clf_random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
clf_random_forest.fit(X_train, y_train)

# Calculate permutation importance
result = permutation_importance(
    clf_random_forest, 
    X_test, 
    y_test, 
    n_repeats=10, 
    random_state=42, 
    n_jobs=-1
)

# Get feature names and extract channel names
feature_names = X_test.columns
channel_names = [name.split('_', 1)[1] for name in feature_names]  # 'power_C3' → 'C3'

# Create DataFrame with permutation importance results
perm_importance = pd.DataFrame({
    'feature': feature_names,
    'channel': channel_names,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
})

# Aggregate by channel (sum the importance scores)
channel_importance = perm_importance.groupby('channel').agg({
    'importance_mean': 'sum',
    'importance_std': lambda x: np.sqrt((x**2).sum())  # combined std
}).sort_values('importance_mean', ascending=False)

# Log top channels
log("\nTop Channels by Permutation Importance:")
log(channel_importance.head(10))

# Plot channel importance
plt.figure(figsize=(12, 6))
channel_importance['importance_mean'].plot(kind='bar', 
                                         yerr=channel_importance['importance_std'],
                                         capsize=4)
plt.title("Channel Importance (Permutation Importance - Random Forest)")
plt.ylabel("Permutation Importance Score")
plt.xlabel("Channel")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/permutation_importance_channels.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot feature-level importance
sorted_idx = result.importances_mean.argsort()[::-1]
plt.figure(figsize=(10, 12))
plt.barh(range(len(sorted_idx)), 
        result.importances_mean[sorted_idx[::-1]], 
        xerr=result.importances_std[sorted_idx[::-1]],
        align="center", 
        capsize=4)
plt.yticks(np.arange(len(sorted_idx)), feature_names[sorted_idx[::-1]])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance (Permutation)")
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/permutation_importance_features_v2.png', dpi=300, bbox_inches='tight')
plt.show()


# # GINI IMPORTANCE | MEAN DECREASE IMPURITY
# 
# Korelasyonlu özelliklerde yanıltıcı olabilir, çünkü benzer bilgi taşıyan bir grup özellikten rastgele biri daha çok önem taşıyormuş gibi görünebilir.
# 
# Daha sağlam ve modelden bağımsız analiz için SHAP veya Permutation Importance gibi yöntemler önerilir.

# In[ ]:


importances = clf_random_forest.feature_importances_

# Feature isimleriyle eşleştir importancı eşleştiriyorum
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# En önemli ilk 20 özellik
log(feature_importance_df.head(20))

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df.head(20), x='Importance', y='Feature', palette='viridis')
plt.title('En Önemli 20 Özellik (Random Forest Gini)')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/gini_importance_features.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:


log("Operation completed")

