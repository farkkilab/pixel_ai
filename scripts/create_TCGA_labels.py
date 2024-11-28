import pandas as pd
import ipdb

tcga_main_df = pd.read_csv('data/TCGA-CDR-SupplementalTableS1.csv')
tcga_main_df = tcga_main_df[tcga_main_df['type']=='OV']
tcga_aux_df = pd.read_csv('data/TCGA_clinical_PANCAN_patient_with_followup.csv',low_memory=False)
with open("data/TCGA_platin_receivers.txt", "r") as file:
    tcga_platin_receivers = set(line.strip() for line in file)
with open("data/TCGA_tp53mut.txt", "r") as file:
    tcga_tp53_mut = set(line.strip() for line in file)
tcga_main_df["platin_receivers"] = tcga_main_df["bcr_patient_barcode"].apply(lambda x: x in tcga_platin_receivers)
tcga_main_df["tp53mut"] = tcga_main_df["bcr_patient_barcode"].apply(lambda x: x in tcga_tp53_mut)
tcga_main_df = tcga_main_df[(tcga_main_df['platin_receivers']==True)&(tcga_main_df['tp53mut']==True)]
tcga_main_df = tcga_main_df[tcga_main_df['clinical_stage'].isin(['Stage IIIC', 'Stage IIIB', 'Stage IV','Stage IIIA'])]
tcga_main_df = tcga_main_df[~((tcga_main_df['PFI'] == 0) & (tcga_main_df['PFI.time'] < 356))]
tcga_aux_df = tcga_aux_df[tcga_aux_df['bcr_patient_barcode'].isin(tcga_main_df["bcr_patient_barcode"])]
tcga_main_df = tcga_main_df.merge(tcga_aux_df, on='bcr_patient_barcode', how='left')
tcga_main_df['pfs_label'] = tcga_main_df['PFI.time'].apply(lambda x: 'short' if x < 356 else 'long')
export_labels = tcga_main_df[['bcr_patient_barcode','pfs_label']]
# slide,patient,er_status_by_ihc

