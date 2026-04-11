from datasets import load_dataset

# Load dataset
dataset = load_dataset("malexandersalazar/mental-health-depression")

# Filter French rows (apply on train split)
french_dataset = dataset["train"].filter(
    lambda x: x["language"].lower() == "french"
)

# Convert to pandas
df_french = french_dataset.to_pandas()

# Save as CSV
df_french.to_csv("french_dataset.csv", index=False, encoding="utf-8")

print("Saved successfully!")

print(df_french.info())