
from config import *
import pandas as pd
import os
from scipy.stats import ttest_ind

if __name__ == "__main__":

    all_data = pd.DataFrame()

    # Read Data

    for file in os.listdir(BASE_FOLDER):
        if file[-4:] == "xlsx":
            data = pd.read_excel(os.path.join(BASE_FOLDER, file))
            if INV_CLASS_MAP[0] in file:
                data["class"] = INV_CLASS_MAP[0]
            else:
                data["class"] = INV_CLASS_MAP[1]
            all_data = pd.concat([all_data, data])
    all_data = all_data.reset_index(drop=True)

    # Analysis
    class_grouped = all_data.groupby('class')
    
    patients_count = class_grouped['Folder name'].nunique()

    benign_patient_count = patients_count[INV_CLASS_MAP[0]]
    malignant_patient_count = patients_count[INV_CLASS_MAP[1]]

    age_mean = class_grouped['Age'].mean()
    age_stdv = class_grouped['Age'].std()
    age_medn = class_grouped['Age'].median()

    benign_age_mean = age_mean[INV_CLASS_MAP[0]]
    malignant_age_mean = age_mean[INV_CLASS_MAP[1]]
    benign_age_stdv = age_stdv[INV_CLASS_MAP[0]]
    malignant_age_stdv = age_stdv[INV_CLASS_MAP[1]]
    benign_age_medn = age_medn[INV_CLASS_MAP[0]]
    malignant_age_medn = age_medn[INV_CLASS_MAP[1]]

    tumor_size_mean = class_grouped['Tumor Size/Length (cm)'].mean()
    tumor_size_stdv = class_grouped['Tumor Size/Length (cm)'].std()
    tumor_size_medn = class_grouped['Tumor Size/Length (cm)'].median()

    benign_tumor_size_mean = tumor_size_mean[INV_CLASS_MAP[0]]
    malignant_tumor_size_mean = tumor_size_mean[INV_CLASS_MAP[1]]
    benign_tumor_size_stdv = tumor_size_stdv[INV_CLASS_MAP[0]]
    malignant_tumor_size_stdv = tumor_size_stdv[INV_CLASS_MAP[1]]
    benign_tumor_size_medn = tumor_size_medn[INV_CLASS_MAP[0]]
    malignant_tumor_size_medn = tumor_size_medn[INV_CLASS_MAP[1]]

    bi_rads_grouped = all_data.groupby(['class', 'BI-RADS'])
    bi_rads_group_cnt = bi_rads_grouped['Folder name'].count().reset_index().rename(columns={'Folder name':'count'})
    bi_rads_group_sum = bi_rads_group_cnt.groupby('class')['count'].sum().reset_index().rename(columns={'count':'sum'})
    bi_rads_joined = pd.merge(bi_rads_group_cnt, bi_rads_group_sum, on='class')
    bi_rads_joined['percent'] = bi_rads_joined['count']/bi_rads_joined['sum']*100
    

    table_variables = [
        ["No. of patients", str(benign_patient_count), str(malignant_patient_count)],
        ["\\textbf{Age (y):}", "", ""],
        ["Mean $\pm$ SD", f"{benign_age_mean:.02f} $\pm$ {benign_age_stdv:.02f}", f"{malignant_age_mean:.02f} $\pm$ {malignant_age_stdv:.02f}"],
        ["Median (range)", str(benign_age_medn), str(malignant_age_medn)],
        ["\\textbf{Tumor Size/Length (cm):}", "", ""],
        ["Mean $\pm$ SD", f"{benign_tumor_size_mean:.02f} $\pm$ {benign_tumor_size_stdv:.02f}", f"{malignant_tumor_size_mean:.02f} $\pm$ {malignant_tumor_size_stdv:.02f}"],
        ["Median (range)", str(benign_tumor_size_medn), str(malignant_tumor_size_medn)],
        ["\\textbf{US BI-RADS grade:}", "", ""],
        ["2, no.(\%)", f"{bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[0]) & (bi_rads_joined['BI-RADS']==2)]['count'].values[0]} ({bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[0]) & (bi_rads_joined['BI-RADS']==2)]['percent'].values[0]:.02f}\%)", f"{bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[1]) & (bi_rads_joined['BI-RADS']==2)]['count'].values[0]} ({bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[1]) & (bi_rads_joined['BI-RADS']==2)]['percent'].values[0]:.02f}\%)"],
        ["3, no.(\%)", f"{bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[0]) & (bi_rads_joined['BI-RADS']==3)]['count'].values[0]} ({bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[0]) & (bi_rads_joined['BI-RADS']==3)]['percent'].values[0]:.02f}\%)", f"{bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[1]) & (bi_rads_joined['BI-RADS']==3)]['count'].values[0]} ({bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[1]) & (bi_rads_joined['BI-RADS']==3)]['percent'].values[0]:.02f}\%)"],
        ["4a, no.(\%)", f"{bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[0]) & (bi_rads_joined['BI-RADS']=='4a')]['count'].values[0]} ({bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[0]) & (bi_rads_joined['BI-RADS']=='4a')]['percent'].values[0]:.02f}\%)", f"{bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[1]) & (bi_rads_joined['BI-RADS']=='4a')]['count'].values[0]} ({bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[1]) & (bi_rads_joined['BI-RADS']=='4a')]['percent'].values[0]:.02f}\%)"],
        ["4b, no.(\%)", f"{bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[0]) & (bi_rads_joined['BI-RADS']=='4b')]['count'].values[0]} ({bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[0]) & (bi_rads_joined['BI-RADS']=='4b')]['percent'].values[0]:.02f}\%)", f"{bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[1]) & (bi_rads_joined['BI-RADS']=='4b')]['count'].values[0]} ({bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[1]) & (bi_rads_joined['BI-RADS']=='4b')]['percent'].values[0]:.02f}\%)"],
        ["4c, no.(\%)", f"{bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[0]) & (bi_rads_joined['BI-RADS']=='4c')]['count'].values[0]} ({bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[0]) & (bi_rads_joined['BI-RADS']=='4c')]['percent'].values[0]:.02f}\%)", f"{bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[1]) & (bi_rads_joined['BI-RADS']=='4c')]['count'].values[0]} ({bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[1]) & (bi_rads_joined['BI-RADS']=='4c')]['percent'].values[0]:.02f}\%)"],
        ["5, no.(\%)", "0", f"{bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[1]) & (bi_rads_joined['BI-RADS']==5)]['count'].values[0]} ({bi_rads_joined.loc[(bi_rads_joined['class']==INV_CLASS_MAP[1]) & (bi_rads_joined['BI-RADS']==5)]['percent'].values[0]:.02f}\%)"]
    ]

    print("")
    for var in table_variables:
        print(" & ".join(var), "\\\\ \\hline")

    # Independent T Test
    test_result_age = ttest_ind(all_data.loc[all_data['class']==INV_CLASS_MAP[0]]['Age'], all_data.loc[all_data['class']==INV_CLASS_MAP[1]]['Age'])
    print("\nAge T-Test", test_result_age)
    test_result_tumor = ttest_ind(all_data.loc[all_data['class']==INV_CLASS_MAP[0]]['Tumor Size/Length (cm)'], all_data.loc[all_data['class']==INV_CLASS_MAP[1]]['Tumor Size/Length (cm)'])
    print("\nTumor T-Test", test_result_tumor)