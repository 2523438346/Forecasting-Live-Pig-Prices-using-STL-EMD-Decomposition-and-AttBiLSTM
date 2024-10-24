import pandas as pd

merge_tyepe = 'future'
imf0 = pd.read_csv(f"emd_pred/pred_resid_imf0_{merge_tyepe}pred.csv")
imf0 = pd.DataFrame(imf0)
# imf0.set_index("date", inplace=True)
imf1 = pd.read_csv(f"emd_pred/pred_resid_imf1_{merge_tyepe}pred.csv")
imf1 = pd.DataFrame(imf1)
# imf1.set_index("date", inplace=True)
imf2 = pd.read_csv(f"emd_pred/pred_resid_imf2_{merge_tyepe}pred.csv")
imf2 = pd.DataFrame(imf2)
# imf2.set_index("date", inplace=True)
imf3 = pd.read_csv(f"emd_pred/pred_resid_imf3_{merge_tyepe}pred.csv")
imf3 = pd.DataFrame(imf3)
# imf3.set_index("date", inplace=True)
imf4 = pd.read_csv(f"emd_pred/pred_resid_imf4_{merge_tyepe}pred.csv")
imf4 = pd.DataFrame(imf4)
# imf4.set_index("date", inplace=True)
imf5 = pd.read_csv(f"emd_pred/pred_resid_imf5_{merge_tyepe}pred.csv")
imf5 = pd.DataFrame(imf5)
# imf5.set_index("date", inplace=True)
emd_res = imf0+imf1+imf2+imf3+imf4+imf5
emd_res.to_csv(f"EMD_result_{merge_tyepe}.csv")

