from cleanfid import fid

save_path_gt = "$PATH_TO_YOUR_GT_IMAGES/gt_results"
save_path = "$PATH_TO_YOUR_RECON_IMAGES/recon_results"

print ("Calculating FID Score...")
# fid_value_clip = fid.compute_fid(save_path_gt, save_path, model_name="clip_vit_b_32", mode="clean")
fid_value = fid.compute_fid(save_path_gt, save_path, mode="clean")

print(
    # f"rFID-clip: {fid_value_clip:.6f}\t"
    f"rFID: {fid_value:.6f}\t")
