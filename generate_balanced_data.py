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

# Apply SMOTE
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

print(f"\nAfter SMOTE - Class distribution:\n{pd.Series(y_bal).value_counts().to_dict()}")
print(f"Total samples: {len(X_bal)}")

# Reconstruct balanced DataFrame
balanced_df = X_bal.copy()
balanced_df['diagnosis'] = y_bal

# Save
balanced_df.to_csv('balanced_data.csv', index=False)
print(f"\nbalanced_data.csv saved with {len(balanced_df)} rows and {balanced_df.shape[1]} columns.")