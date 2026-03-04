# visualisation.py
# -----------------------------------------------------------
# Pedestrian trajectory visualisation for SGAN-style models
# Compatible with Social-GAN / SGAN-modernised repositories
# -----------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

# Safe backend for Colab / headless execution
plt.switch_backend("agg")


# -----------------------------------------------------------
# UTILITY
# -----------------------------------------------------------
def _to_numpy(tensor):
    """
    Converts torch tensor or numpy array to numpy.
    """
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
    except Exception:
        pass
    return np.array(tensor)


# -----------------------------------------------------------
# MAIN VISUALISATION FUNCTION
# -----------------------------------------------------------
def plot_trajectories(
    obs_traj,
    pred_traj_gt=None,
    pred_traj_fake=None,
    ped_id=None,   # None → all pedestrians | int → single pedestrian
    save_path=None,
    show=True,
    title="Pedestrian Trajectories"
):
    """
    Plot trajectories.

    ped_id = None  → plot ALL pedestrians
    ped_id = int   → plot ONE pedestrian
    """

    obs_traj = _to_numpy(obs_traj)

    if pred_traj_gt is not None:
        pred_traj_gt = _to_numpy(pred_traj_gt)

    if pred_traj_fake is not None:
        pred_traj_fake = _to_numpy(pred_traj_fake)

    obs_len, num_peds, _ = obs_traj.shape

    plt.figure(figsize=(7, 7))
    plt.title(title)

    # -------------------------------------------------
    # CASE 1: Plot ONE pedestrian
    # -------------------------------------------------
    if ped_id is not None:

        if ped_id >= num_peds:
            print(f"[Warning] ped_id {ped_id} out of range. Using 0 instead.")
            ped_id = 0

        obs = obs_traj[:, ped_id, :]
        last_point = obs[-1]

        # Observed
        plt.plot(
            obs[:, 0], obs[:, 1],
            "o-", linewidth=3, label="Observed"
        )

        # Ground truth
        if pred_traj_gt is not None:
            gt = pred_traj_gt[:, ped_id, :]
            gt = np.vstack([last_point, gt])

            plt.plot(
                gt[:, 0], gt[:, 1],
                "x--", linewidth=3, label="Ground Truth"
            )

        # Prediction
        if pred_traj_fake is not None:
            pred = pred_traj_fake[:, ped_id, :]
            pred = np.vstack([last_point, pred])

            plt.plot(
                pred[:, 0], pred[:, 1],
                "s-", linewidth=3, label="Predicted"
            )

        # Prediction start marker
        plt.scatter(
            last_point[0],
            last_point[1],
            s=120,
            marker="*",
            label="Prediction Start",
        )

    # -------------------------------------------------
    # CASE 2: Plot ALL pedestrians
    # -------------------------------------------------
    else:
        for ped in range(num_peds):

            obs = obs_traj[:, ped, :]
            plt.plot(
                obs[:, 0], obs[:, 1],
                linestyle="-",
                marker="o",
                label="Observed" if ped == 0 else "",
            )

            last_point = obs[-1]

            if pred_traj_gt is not None:
                gt = pred_traj_gt[:, ped, :]
                gt = np.vstack([last_point, gt])

                plt.plot(
                    gt[:, 0], gt[:, 1],
                    linestyle="--",
                    marker="x",
                    label="Ground Truth" if ped == 0 else "",
                )

            if pred_traj_fake is not None:
                pred = pred_traj_fake[:, ped, :]
                pred = np.vstack([last_point, pred])

                plt.plot(
                    pred[:, 0], pred[:, 1],
                    linestyle="-.",
                    marker="*",
                    label="Predicted" if ped == 0 else "",
                )

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # Save
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    # Display
    if show:
        plt.show()

    plt.close()


# -----------------------------------------------------------
# BATCH VISUALISER (useful during evaluation)
# -----------------------------------------------------------
def visualize_batch(
    obs_traj,
    pred_traj_gt,
    pred_traj_fake,
    output_dir="plots",
    max_scenes=5,
):
    """
    Visualize multiple scenes from a batch.
    """

    obs_traj = _to_numpy(obs_traj)
    pred_traj_gt = _to_numpy(pred_traj_gt)
    pred_traj_fake = _to_numpy(pred_traj_fake)

    os.makedirs(output_dir, exist_ok=True)

    num_scenes = min(max_scenes, obs_traj.shape[1])

    # --- Save scene-wise plots ---
    for i in range(num_scenes):
        plot_trajectories(
            obs_traj[:, i:i+1, :],
            pred_traj_gt[:, i:i+1, :],
            pred_traj_fake[:, i:i+1, :],
            save_path=f"{output_dir}/scene_{i}.png",
            show=False,
            title=f"Scene {i}",
        )

    # --- Also save one highlighted pedestrian ---
    plot_trajectories(
        obs_traj,
        pred_traj_gt,
        pred_traj_fake,
        ped_id=0,
        save_path=f"{output_dir}/highlight_ped.png",
        show=False,
        title="Highlighted Pedestrian",
    )


# -----------------------------------------------------------
# QUICK DEBUG FUNCTION
# -----------------------------------------------------------
def quick_plot(obs_traj, pred_traj_fake):
    """
    Minimal plotting helper.
    """
    plot_trajectories(
        obs_traj,
        pred_traj_fake=pred_traj_fake,
        show=True,
    )