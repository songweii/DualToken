from cleanfid import fid

save_path_gt = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/songwei-240108120100/code/DualToken-new/DualToken-attn/eval-attn-18-fp32-v4/gt_results"
save_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/songwei-240108120100/code/DualToken-new/DualToken-attn/eval-attn-18-fp32-v4/recon_results"

print ("Calculating FID Score...")
# fid_value_clip = fid.compute_fid(save_path_gt, save_path, model_name="clip_vit_b_32", mode="clean")
fid_value = fid.compute_fid(save_path_gt, save_path, mode="clean")

print(
    # f"rFID-clip: {fid_value_clip:.6f}\t"
    f"rFID: {fid_value:.6f}\t")
