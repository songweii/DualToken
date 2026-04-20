from cleanfid import fid

save_path_gt = "/mnt/public/users/songwei/code/DualToken-causal-downdim/eval-siglip2-384-rvq8-32d-9/gt_results"
save_path = "/mnt/public/users/songwei/code/DualToken-causal-downdim/eval-siglip2-384-rvq8-32d-9/recon_results"

print ("Calculating FID Score...")
# fid_value_clip = fid.compute_fid(save_path_gt, save_path, model_name="clip_vit_b_32", mode="clean")
fid_value = fid.compute_fid(save_path_gt, save_path, mode="clean")

print(
    # f"rFID-clip: {fid_value_clip:.6f}\t"
    f"rFID: {fid_value:.6f}\t")
