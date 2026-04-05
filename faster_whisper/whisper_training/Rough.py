import pandas as pd
df = pd.read_parquet("/root/autodl-tmp/whisper_project/whisper_training/Sample01.parquet")
print(df.columns)
print(df.iloc[0])