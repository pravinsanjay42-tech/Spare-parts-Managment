import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("data/spare_parts_demand.csv")
    
    print("--- Correlation Check ---")
    
    # 1. Weather Severity
    df['weather_bucket'] = pd.cut(df['weather_severity_index'], bins=[0, 0.4, 0.8, 1.0], labels=['Low', 'Medium', 'High'])
    print("\nMean Demand by Weather Severity:")
    print(df.groupby('weather_bucket', observed=True)['demand'].mean())
    
    # 2. Festival
    print("\nMean Demand by Festival Flag:")
    print(df.groupby('is_festival')['demand'].mean())
    
    # 3. Equipment Age
    df['age_bucket'] = pd.cut(df['equipment_age_years'], bins=[1, 3, 5, 7, 10], labels=['1-3 yrs', '3-5 yrs', '5-7 yrs', '7-10 yrs'])
    print("\nMean Demand by Equipment Age:")
    print(df.groupby('age_bucket', observed=True)['demand'].mean())

if __name__ == "__main__":
    main()
