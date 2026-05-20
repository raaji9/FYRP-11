import pandas as pd
from imblearn.over_sampling import SMOTE

# Load raw data
df = pd.read_csv('data.csv')

# Drop id and Unnamed: 32 if present
if 'Unnamed: 32' in df.columns:
    df = df.drop(['id', 'Unnamed: 32'], axis=1)
else:
    df = df.drop(['id'], axis=1)

# Encode diagnosis: M=1, B=0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print(f"Before SMOTE - Class distribution:\n{y.value_counts().to_dict()}")
print(f"Total samples: {len(df)}")

# Apply SMOTE with sampling_strategy to get 500 each
smote = SMOTE(sampling_strategy={0: 500, 1: 500}, random_state=42)
X_aug, y_aug = smote.fit_resample(X, y)

print(f"\nAfter SMOTE - Class distribution:\n{pd.Series(y_aug).value_counts().to_dict()}")
print(f"Total samples: {len(X_aug)}")

# Reconstruct augmented DataFrame
aug_df = X_aug.copy()
aug_df['diagnosis'] = y_aug

# Save
aug_df.to_csv('augmented_data.csv', index=False)
print(f"\naugmented_data.csv saved with {len(aug_df)} rows and {aug_df.shape[1]} columns.")