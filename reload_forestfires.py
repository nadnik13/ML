import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('forestfires.csv')
    month_name = {month: i//6 for i , month in
                  enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}
    print(month_name)
    data['bin_month'] = data['month'].map(month_name)
    data.to_csv('reload_forest_fire.csv')
