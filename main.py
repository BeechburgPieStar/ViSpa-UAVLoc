import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from scipy.io import savemat

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ============================
# 0. random seed
# ============================
SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{DEVICE}")

IMG_W, IMG_H = 320, 180  # (W, H)

# ============================
# 1. modal network
# ============================
class BiFiLM(nn.Module):
    """
    Bidirectional FiLM conditioning between signal and image branches.

    s_feat: (B, Cs, L)
    i_feat: (B, Ci, H, W)
    s_vec : (B, Ds)  from GAP(s_feat)
    i_vec : (B, Di)  from GAP(i_feat)
    """
    def __init__(self, cs=128, ci=256, ds=128, di=256, hidden=256, use_ln=False):
        super().__init__()
        self.use_ln = use_ln
        if use_ln:
            self.ln_s = nn.LayerNorm(cs)
            self.ln_i = nn.LayerNorm(ci)

        self.i2s = nn.Sequential(
            nn.Linear(di, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * cs)
        )

        self.s2i = nn.Sequential(
            nn.Linear(ds, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * ci)
        )

    def forward(self, s_feat, i_feat, s_vec, i_vec):
        B, Cs, L = s_feat.shape
        Bi, Ci, H, W = i_feat.shape
        if B != Bi:
            raise RuntimeError("Batch size mismatch between s_feat and i_feat")

        if self.use_ln:
            s_feat_ = self.ln_s(s_feat.transpose(1, 2)).transpose(1, 2)
            i_feat_ = self.ln_i(i_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            s_feat_ = s_feat
            i_feat_ = i_feat

        gbs = self.i2s(i_vec)
        gamma_s, beta_s = gbs[:, :Cs], gbs[:, Cs:]
        gamma_s = gamma_s.unsqueeze(-1)
        beta_s = beta_s.unsqueeze(-1)
        s_mod = s_feat_ * (1.0 + gamma_s) + beta_s

        gbi = self.s2i(s_vec)
        gamma_i, beta_i = gbi[:, :Ci], gbi[:, Ci:]
        gamma_i = gamma_i.view(B, Ci, 1, 1)
        beta_i = beta_i.view(B, Ci, 1, 1)
        i_mod = i_feat_ * (1.0 + gamma_i) + beta_i

        return s_mod, i_mod


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim, drop=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop(x)
        x = x.transpose(1, 2)
        return residual + x


class ConvNeXtBlock2D(nn.Module):
    def __init__(self, dim, drop=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


class LowRankBilinearFusion(nn.Module):
    """
    Low rank bilinear fusion on two modality vectors.

    s_vec: (B, ds)
    i_vec: (B, di)
    out : (B, d)
    """
    def __init__(self, ds=128, di=256, d=256, r=32, use_ln=True):
        super().__init__()
        self.r = r
        self.d = d

        self.s_proj = nn.Linear(ds, r * d)
        self.i_proj = nn.Linear(di, r * d)

        if use_ln:
            self.post = nn.Sequential(nn.LayerNorm(d), nn.GELU())
        else:
            self.post = nn.GELU()

    def forward(self, s_vec, i_vec):
        B = s_vec.size(0)
        s = self.s_proj(s_vec).view(B, self.r, self.d)
        i = self.i_proj(i_vec).view(B, self.r, self.d)
        out = (s * i).sum(dim=1)
        out = self.post(out)
        return out


class MultimodalFusionNet(nn.Module):
    def __init__(self, out_dim=3, fuse_d=256, fuse_r=32):
        super().__init__()

        self.sig_stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.sig_stage = nn.Sequential(
            ConvNeXtBlock1D(64, drop=0.0),
            nn.Conv1d(64, 128, kernel_size=2, stride=2),
            ConvNeXtBlock1D(128, drop=0.0),
            ConvNeXtBlock1D(128, drop=0.0)
        )
        self.sig_gap = nn.AdaptiveAvgPool1d(1)
        self.sig_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )

        self.img_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4),
            nn.GroupNorm(1, 64)
        )
        self.img_stage = nn.Sequential(
            ConvNeXtBlock2D(64, drop=0.0),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            ConvNeXtBlock2D(128, drop=0.0),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            ConvNeXtBlock2D(256, drop=0.0)
        )
        self.img_gap = nn.AdaptiveAvgPool2d(1)
        self.img_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        self.bifilm = BiFiLM(cs=128, ci=256, ds=128, di=256, hidden=256, use_ln=False)

        self.fusion = LowRankBilinearFusion(ds=128, di=256, d=fuse_d, r=fuse_r, use_ln=True)

        self.head = nn.Sequential(
            nn.Linear(fuse_d, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, img, sig):
        if sig.dim() == 2:
            sig = sig.unsqueeze(1)

        s_feat = self.sig_stem(sig)
        s_feat = self.sig_stage(s_feat)
        s_vec0 = self.sig_gap(s_feat).flatten(1)

        i_feat = self.img_stem(img)
        i_feat = self.img_stage(i_feat)
        i_vec0 = self.img_gap(i_feat).flatten(1)

        s_feat, i_feat = self.bifilm(s_feat, i_feat, s_vec0, i_vec0)

        s_vec = self.sig_proj(self.sig_gap(s_feat))
        i_vec = self.img_proj(self.img_gap(i_feat))

        fused = self.fusion(s_vec, i_vec)
        return self.head(fused)

# ============================
# 2. dataload
# ============================
class MultimodalDataset(Dataset):
    def __init__(self, image_paths, signal_data, targets, transform=None):
        self.image_paths = image_paths
        self.signal_data = signal_data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            # 如果你担心数据集中存在尺寸不一致，可强制统一到 960x540
            # image = image.resize((IMG_W, IMG_H))
            if self.transform:
                image = self.transform(image)
        except Exception:
            image = torch.zeros((3, IMG_H, IMG_W), dtype=torch.float32)

        signal = torch.tensor(self.signal_data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, signal, target

def retain_topk_per_row(X, k):
    if k >= X.shape[1]:
        return X

    X_out = np.zeros_like(X)
    abs_X = np.abs(X)

    # 每一行找 Top-k 的索引（无序）
    topk_idx = np.argpartition(abs_X, -k, axis=1)[:, -k:]

    row_idx = np.arange(X.shape[0])[:, None]
    X_out[row_idx, topk_idx] = X[row_idx, topk_idx]

    return X_out

def prepare_data(img_folder, csv_input, csv_output, k=8):
    print("loading...")
    all_img_paths = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))

    df_in = pd.read_csv(csv_input, header=0)
    X_signal_raw = df_in.iloc[:, 1:65].values.astype(np.float32)

    # ========= 新增：逐行 Top-k 保留 =========
    X_signal_raw = retain_topk_per_row(X_signal_raw, k)
    # =======================================

    df_out = pd.read_csv(csv_output, header=None)
    y_raw = df_out.values.astype(np.float32)

    min_len = min(len(all_img_paths), len(X_signal_raw), len(y_raw))
    print(f"effective sample: {min_len}")

    all_img_paths = all_img_paths[:min_len]
    X_signal_raw = X_signal_raw[:min_len]
    y_raw = y_raw[:min_len]

    indices = np.arange(min_len)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED
    )

    scaler_sig = StandardScaler()
    X_sig_train = scaler_sig.fit_transform(X_signal_raw[train_idx])
    X_sig_val   = scaler_sig.transform(X_signal_raw[val_idx])
    X_sig_test  = scaler_sig.transform(X_signal_raw[test_idx])

    scaler_y_list = [StandardScaler() for _ in range(3)]
    y_train_norm = np.zeros_like(y_raw[train_idx])
    y_val_norm   = np.zeros_like(y_raw[val_idx])
    y_test_norm  = np.zeros_like(y_raw[test_idx])

    for d in range(3):
        y_train_norm[:, d:d+1] = scaler_y_list[d].fit_transform(
            y_raw[train_idx, d:d+1]
        )
        y_val_norm[:, d:d+1] = scaler_y_list[d].transform(
            y_raw[val_idx, d:d+1]
        )
        y_test_norm[:, d:d+1] = scaler_y_list[d].transform(
            y_raw[test_idx, d:d+1]
        )

    split_paths = {
        "train": [all_img_paths[i] for i in train_idx],
        "val":   [all_img_paths[i] for i in val_idx],
        "test":  [all_img_paths[i] for i in test_idx]
    }

    split_data = {
        "train": (split_paths["train"], X_sig_train, y_train_norm),
        "val":   (split_paths["val"],   X_sig_val,   y_val_norm),
        "test":  (split_paths["test"],  X_sig_test,  y_test_norm)
    }

    return split_data, scaler_y_list


# ============================
# 2.1 training set pre
# ============================
@torch.no_grad()
def compute_train_mean_std(train_image_paths, batch_size=8, num_workers=0):
    tfm = transforms.Compose([
        transforms.ToTensor(),  # -> float32 in [0,1], shape (3,H,W)
    ])

    class ImgOnlyDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            p = self.paths[idx]
            try:
                img = Image.open(p).convert("RGB")
                # 如需强制尺寸一致，可打开下一行
                img = img.resize((IMG_W, IMG_H))
                img = self.transform(img)
            except Exception:
                img = torch.zeros((3, IMG_H, IMG_W), dtype=torch.float32)
            return img

    loader = DataLoader(
        ImgOnlyDataset(train_image_paths, tfm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 累加 sum 与 sumsq
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sumsq = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for imgs in loader:
        # imgs: (B,3,H,W)
        b, c, h, w = imgs.shape
        imgs = imgs.to(dtype=torch.float64)
        channel_sum += imgs.sum(dim=(0, 2, 3))
        channel_sumsq += (imgs ** 2).sum(dim=(0, 2, 3))
        pixel_count += b * h * w

    mean = channel_sum / pixel_count
    var = channel_sumsq / pixel_count - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-12))

    mean = mean.to(dtype=torch.float32).tolist()
    std = std.to(dtype=torch.float32).tolist()
    return mean, std

# ============================
# 3. training and analysis
# ============================
def train_main():
    IMG_DIR = "/mnt/hdd1/6Gssens23/scenario23_dev_w_resources/scenario23_dev/unit1/camera_data/"
    CSV_IN = "input_rss.csv"
    CSV_OUT = "output_3dlocation.csv"
    BATCH_SIZE = 64
    LR = 8e-4
    EPOCHS = 220

    data_pack, scaler_y_list = prepare_data(IMG_DIR, CSV_IN, CSV_OUT, k=64)


    train_img_paths = data_pack["train"][0]
    mean_rgb, std_rgb = compute_train_mean_std(train_img_paths, batch_size=8, num_workers=0)
    print(f"[Train RGB Mean] {mean_rgb}")
    print(f"[Train RGB Std ] {std_rgb}")

    train_tfm = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        # transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean_rgb, std_rgb),
    ])
    test_tfm = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean_rgb, std_rgb),
    ])

    train_loader = DataLoader(
        MultimodalDataset(*data_pack["train"], transform=train_tfm),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        MultimodalDataset(*data_pack["val"], transform=test_tfm),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        MultimodalDataset(*data_pack["test"], transform=test_tfm),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    model = MultimodalFusionNet(out_dim=3).to(DEVICE)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    patience_cnt = 0
    train_losses, val_losses = [], []

    print("\n=== start training ===")
    for epoch in range(EPOCHS):
        model.train()
        r_loss = 0.0
        for imgs, sigs, targets in train_loader:
            imgs, sigs, targets = imgs.to(DEVICE), sigs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            out = model(imgs, sigs)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            r_loss += loss.item()

        avg_train_loss = r_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for imgs, sigs, targets in val_loader:
                imgs, sigs, targets = imgs.to(DEVICE), sigs.to(DEVICE), targets.to(DEVICE)
                out = model(imgs, sigs)
                v_loss += criterion(out, targets).item()

        avg_val_loss = v_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        # Save Best
        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), "best_multimodal_modelv3_topk.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= 30:
                print("Early Stopping Triggered.")
                break

    # ============================
    # 6. 最终评估
    # ============================
    print("\n========================================")
    print("        test and results (Multimodal)        ")
    print("========================================")

    model.load_state_dict(torch.load("best_multimodal_modelv3_topk.pth", map_location=DEVICE))
    model.eval()

    preds_list, targets_list = [], []
    with torch.no_grad():
        for imgs, sigs, targets in test_loader:
            imgs, sigs = imgs.to(DEVICE), sigs.to(DEVICE)
            out = model(imgs, sigs)
            preds_list.append(out.cpu().numpy())
            targets_list.append(targets.numpy())

    pred_scaled = np.vstack(preds_list)
    target_scaled = np.vstack(targets_list)

    # --- 反归一化 ---
    pred_real = np.zeros_like(pred_scaled, dtype=np.float32)
    target_real = np.zeros_like(target_scaled, dtype=np.float32)

    for d in range(3):
        pred_real[:, d:d+1] = scaler_y_list[d].inverse_transform(pred_scaled[:, d:d+1])
        target_real[:, d:d+1] = scaler_y_list[d].inverse_transform(target_scaled[:, d:d+1])

    # --- 计算指标 ---
    mse_real = mean_squared_error(target_real, pred_real)
    rmse_real = np.sqrt(mse_real)
    mae_real = mean_absolute_error(target_real, pred_real)

    dist_errors = np.linalg.norm(target_real - pred_real, axis=1)
    mpe = float(np.mean(dist_errors))

    print("[overall]")
    print(f"  > overall MPE: {mpe:.6f} m")
    print(f"  > overall RMSE         : {rmse_real:.6f}")
    print(f"  > overall MAE          : {mae_real:.6f}")
    print("----------------------------------------")

    print("[axis]")
    axis_names = ["X", "Y", "Z"]
    for i in range(3):
        rmse_axis = np.sqrt(mean_squared_error(target_real[:, i], pred_real[:, i]))
        mae_axis = mean_absolute_error(target_real[:, i], pred_real[:, i])
        print(f"  > {axis_names[i]}: RMSE={rmse_axis:.4f}, MAE={mae_axis:.4f}")
    print("========================================")
    


    # ============================
    # 7.2 XY projection + error CDF
    # ============================
    
    # ---------- (A) XY Projection ----------
    plt.figure(figsize=(7, 6))
    num_points = min(500, len(target_real))
    
    plt.scatter(
        target_real[:num_points, 0], target_real[:num_points, 1],
        marker='o', alpha=0.6, label='True (XY)'
    )
    plt.scatter(
        pred_real[:num_points, 0], pred_real[:num_points, 1],
        marker='^', alpha=0.6, label='Predicted (XY)'
    )
    
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"UAV Positioning: XY Projection (First {num_points} test samples)")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig("uav_xy_projection.png", dpi=300)
    print("XYstored in: uav_xy_projection.png")
    
    
    # ---------- (B) Error CDF (3D Distance Error) ----------
    
    errs = dist_errors.astype(np.float32)
    errs_sorted = np.sort(errs)
    cdf = np.arange(1, len(errs_sorted) + 1) / len(errs_sorted)
    
    # 可选：标注关键分位数（更“paper-friendly”）
    p50 = float(np.percentile(errs, 50))
    p80 = float(np.percentile(errs, 80))
    p90 = float(np.percentile(errs, 90))
    p95 = float(np.percentile(errs, 95))
    
    plt.figure(figsize=(7, 6))
    plt.plot(errs_sorted, cdf)
    
    plt.axvline(p50, linestyle='--', linewidth=1.0)
    plt.axvline(p80, linestyle='--', linewidth=1.0)
    plt.axvline(p90, linestyle='--', linewidth=1.0)
    plt.axvline(p95, linestyle='--', linewidth=1.0)
    
    plt.text(p50, 0.52, f"P50={p50:.2f}m", rotation=90, va='bottom', ha='right')
    plt.text(p80, 0.82, f"P80={p80:.2f}m", rotation=90, va='bottom', ha='right')
    plt.text(p90, 0.92, f"P90={p90:.2f}m", rotation=90, va='bottom', ha='right')
    plt.text(p95, 0.97, f"P95={p95:.2f}m", rotation=90, va='bottom', ha='right')
    
    plt.xlabel("3D Positioning Error ‖e‖₂ (m)")
    plt.ylabel("CDF")
    plt.title("CDF of 3D Positioning Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("uav_error_cdf.png", dpi=300)
    print("CDFstored: uav_error_cdf.png")
    
    print("[error per | 3Dloc error]")
    print(f"  P50: {p50:.4f} m")
    print(f"  P80: {p80:.4f} m")
    print(f"  P90: {p90:.4f} m")
    print(f"  P95: {p95:.4f} m")
    
    plot_data = {
    "target_pos": target_real.astype(np.float32),   # (N,3)
    "pred_pos":   pred_real.astype(np.float32),     # (N,3)
    "dist_error": dist_errors.astype(np.float32),   # (N,)
    }

    mat_file = "uav_positioning_plot_data.mat"
    savemat(mat_file, plot_data)

    print(f"绘图数据已保存至: {mat_file}")

if __name__ == "__main__":
    train_main()
