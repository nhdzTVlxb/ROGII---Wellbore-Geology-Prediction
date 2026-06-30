"""
ROGII 2D CNN+SDF Baseline (B2) — TensorFlow/Keras UNet
P100-safe: uses TF instead of PyTorch (PyTorch 2.10+cu128 dropped sm_60 support).
Input: [B, T, H, C] image (typewell_GR, horiz_GR, GR_diff, history, mask)
Target: SDF = (h_tvt - t_tvt) / 40, clipped [-3, 3]
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import GroupKFold

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

# ── Config ───────────────────────────────────────────────────
H = 512
T = 256
BATCH_SIZE = 8
EPOCHS = 30
LR = 3e-4
N_SPLITS = 5
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")

# ── Data loading ─────────────────────────────────────────────
def load_well_pair(hw_path, tw_path, is_train=True):
    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path).sort_values('TVT')
    if is_train and ('TVT' not in hw.columns or hw['TVT'].isna().all()):
        return None
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0 or len(kn) < 5:
        return None
    tw_tvt = tw['TVT'].values.astype(np.float32)
    tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(np.float32)
    h_gr = hw['GR'].astype(float).interpolate(limit_direction='both').fillna(float(tw_gr.mean())).values.astype(np.float32)
    h_tvt = hw['TVT_input'].values.copy() if 'TVT_input' in hw.columns else np.full(len(hw), np.nan, np.float32)
    if is_train and 'TVT' in hw.columns:
        h_tvt[~np.isnan(hw['TVT'].values)] = hw['TVT'].values[~np.isnan(hw['TVT'].values)]
    h_z = hw['Z'].values.astype(np.float32)
    return {
        'wid': hw_path.stem.replace('__horizontal_well', ''),
        'tw_tvt': tw_tvt, 'tw_gr': tw_gr,
        'h_gr': h_gr, 'h_tvt': h_tvt, 'h_z': h_z,
        'kn_mask': hw['TVT_input'].notna().values,
    }

def build_2d_sample(well, H=512, T=256):
    tw_tvt = well['tw_tvt']
    tw_gr = well['tw_gr']
    h_gr = well['h_gr']
    h_tvt = well['h_tvt']
    h_z = well['h_z']
    kn_mask = well['kn_mask']
    nh = len(h_gr)
    ps_idx = np.where(kn_mask)[0][-1] if kn_mask.any() else 0
    
    if nh > H:
        start = max(0, min(ps_idx - H // 4, nh - H))
        h_gr_seg = h_gr[start:start + H]
        h_tvt_seg = h_tvt[start:start + H]
        h_z_seg = h_z[start:start + H]
        kn_mask_seg = kn_mask[start:start + H]
        ps_idx_seg = ps_idx - start
    else:
        pad_right = H - nh
        h_gr_seg = np.pad(h_gr, (0, pad_right), mode='edge')
        h_tvt_seg = np.pad(h_tvt, (0, pad_right), mode='constant', constant_values=np.nan)
        h_z_seg = np.pad(h_z, (0, pad_right), mode='edge')
        kn_mask_seg = np.pad(kn_mask, (0, pad_right), mode='constant', constant_values=False)
        ps_idx_seg = ps_idx
    
    last_tvt = float(h_tvt[ps_idx]) if not np.isnan(h_tvt[ps_idx]) else float(tw_tvt[len(tw_tvt) // 2])
    center_idx = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    t0 = max(0, center_idx - T // 2)
    t1 = min(len(tw_tvt), t0 + T)
    t_pad_left = max(0, T // 2 - center_idx)
    t_pad_right = T - (t1 - t0) - t_pad_left
    tw_tvt_crop = np.pad(tw_tvt[t0:t1], (t_pad_left, t_pad_right), mode='edge')
    tw_gr_crop = np.pad(tw_gr[t0:t1], (t_pad_left, t_pad_right), mode='edge')
    
    # Channels: [T, H, C]
    ch0 = np.tile(tw_gr_crop[:, None], (1, H))  # typewell GR
    ch1 = np.tile(h_gr_seg[None, :], (T, 1))   # horizontal GR
    ch2 = ch1 - ch0                               # GR diff
    ch3 = np.zeros((T, H), np.float32)            # history mask
    for i in range(H - 1):
        if not kn_mask_seg[i] or not kn_mask_seg[i + 1]:
            continue
        t_idx = int(np.argmin(np.abs(tw_tvt_crop - h_tvt_seg[i])))
        t_idx_next = int(np.argmin(np.abs(tw_tvt_crop - h_tvt_seg[i + 1])))
        cv2.line(ch3, (i, t_idx), (i + 1, t_idx_next), 1.0, 2)
    ch4 = np.ones((T, H), np.float32)             # validity mask
    
    image = np.stack([ch0, ch1, ch2, ch3, ch4], axis=-1)  # [T, H, 5]
    
    sdf = np.zeros((T, H), np.float32)
    for i in range(H):
        if np.isnan(h_tvt_seg[i]):
            continue
        sdf[:, i] = (h_tvt_seg[i] - tw_tvt_crop) / 40.0
    sdf = np.clip(sdf, -3.0, 3.0)
    
    mask = np.zeros((T, H), np.float32)
    for i in range(H):
        if not np.isnan(h_tvt_seg[i]):
            mask[:, i] = 1.0
    
    return image, sdf, mask, ps_idx_seg, well['wid']

# ── Keras UNet ───────────────────────────────────────────────
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def build_unet(input_shape=(T, H, 5)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 128)
    p3 = tf.keras.layers.MaxPooling2D(2)(c3)
    c4 = conv_block(p3, 256)
    p4 = tf.keras.layers.MaxPooling2D(2)(c4)
    # Bottleneck
    bn = conv_block(p4, 512)
    # Decoder
    u4 = tf.keras.layers.UpSampling2D(2)(bn)
    u4 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(u4)
    u4 = tf.keras.layers.concatenate([u4, c4])
    d4 = conv_block(u4, 256)
    
    u3 = tf.keras.layers.UpSampling2D(2)(d4)
    u3 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(u3)
    u3 = tf.keras.layers.concatenate([u3, c3])
    d3 = conv_block(u3, 128)
    
    u2 = tf.keras.layers.UpSampling2D(2)(d3)
    u2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(u2)
    u2 = tf.keras.layers.concatenate([u2, c2])
    d2 = conv_block(u2, 64)
    
    u1 = tf.keras.layers.UpSampling2D(2)(d2)
    u1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(u1)
    u1 = tf.keras.layers.concatenate([u1, c1])
    d1 = conv_block(u1, 32)
    
    outputs = tf.keras.layers.Conv2D(1, 1, activation='tanh')(d1)
    outputs = outputs * 3.0
    model = tf.keras.Model(inputs, outputs)
    return model

# ── Training ─────────────────────────────────────────────────
def create_dataset(images, sdfs, masks, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((images, sdfs, masks))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def masked_mse(y_true, y_pred, mask):
    mse = tf.square(y_true - y_pred)
    return tf.reduce_sum(mse * mask) / tf.reduce_sum(mask)

def train_step(model, optimizer, x, y, m):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = masked_mse(y, pred, m)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def val_step(model, x, y, m):
    pred = model(x, training=False)
    loss = masked_mse(y, pred, m)
    return loss

def train_fold(train_ds, val_ds, fold):
    model = build_unet()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4)
    best_val = float('inf')
    best_weights = None
    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_count = 0
        for x, y, m in train_ds:
            loss = train_step(model, optimizer, x, y, m)
            train_loss += loss.numpy()
            train_count += 1
        val_loss = 0.0
        val_count = 0
        for x, y, m in val_ds:
            loss = val_step(model, x, y, m)
            val_loss += loss.numpy()
            val_count += 1
        train_loss /= max(train_count, 1)
        val_loss /= max(val_count, 1)
        val_rmse = np.sqrt(val_loss)
        print(f"Fold {fold} Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.4f}, val_rmse={val_rmse:.4f}")
        if val_rmse < best_val:
            best_val = val_rmse
            best_weights = model.get_weights()
    print(f"Fold {fold} best val RMSE: {best_val:.4f}")
    return best_weights, best_val

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
print("="*60)
print("ROGII 2D CNN+SDF B2 — TF/Keras UNet")
print("="*60)

train_dir = DATA_ROOT / "train"
test_dir = DATA_ROOT / "test"

# Load all wells
print("\nLoading wells...")
wells = []
for hw_path in sorted(train_dir.glob('*__horizontal_well.csv')):
    tw_path = train_dir / f'{hw_path.stem.replace("__horizontal_well", "")}__typewell.csv'
    if not tw_path.exists():
        continue
    w = load_well_pair(hw_path, tw_path, is_train=True)
    if w is not None:
        wells.append(w)
print(f"Loaded {len(wells)} train wells")

# Build 2D samples
print("Building 2D samples...")
samples = []
for w in wells:
    try:
        img, sdf, mask, ps, wid = build_2d_sample(w, H=H, T=T)
        samples.append((img, sdf, mask, ps, wid))
    except Exception as e:
        print(f"Skip {w['wid']}: {e}")
print(f"Built {len(samples)} samples")

# Prepare arrays
images = np.stack([s[0] for s in samples], axis=0)
sdfs = np.stack([s[1] for s in samples], axis=0)[..., None]  # [B, T, H, 1]
masks = np.stack([s[2] for s in samples], axis=0)[..., None]
wids = [s[4] for s in samples]
groups = pd.factorize(wids)[0]
gkf = GroupKFold(n_splits=N_SPLITS)

fold_weights = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(samples, groups=groups)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    train_ds = create_dataset(images[train_idx], sdfs[train_idx], masks[train_idx], BATCH_SIZE, shuffle=True)
    val_ds = create_dataset(images[val_idx], sdfs[val_idx], masks[val_idx], BATCH_SIZE, shuffle=False)
    weights, val_rmse = train_fold(train_ds, val_ds, fold)
    fold_weights.append(weights)
    # Save weights
    np.savez(f"weights_fold{fold}.npz", *weights)

print("\nTraining complete. Loading test wells...")

# Test inference
test_wells = []
for hw_path in sorted(test_dir.glob('*__horizontal_well.csv')):
    tw_path = test_dir / f'{hw_path.stem.replace("__horizontal_well", "")}__typewell.csv'
    if not tw_path.exists():
        continue
    w = load_well_pair(hw_path, tw_path, is_train=False)
    if w is not None:
        test_wells.append(w)
print(f"Loaded {len(test_wells)} test wells")

# Build test samples
test_samples = []
for w in test_wells:
    try:
        img, sdf, mask, ps, wid = build_2d_sample(w, H=H, T=T)
        test_samples.append((img, sdf, mask, ps, wid))
    except Exception as e:
        print(f"Skip test {w['wid']}: {e}")

if test_samples:
    test_images = np.stack([s[0] for s in test_samples], axis=0)
    test_wids_list = [s[4] for s in test_samples]
    
    # Ensemble inference
    print("Running inference...")
    model = build_unet()
    all_preds = {}
    for fold, weights in enumerate(fold_weights):
        model.set_weights(weights)
        preds = model.predict(test_images, batch_size=BATCH_SIZE, verbose=0)
        for i, wid in enumerate(test_wids_list):
            if wid not in all_preds:
                all_preds[wid] = []
            all_preds[wid].append(preds[i, :, :, 0])
    
    # Extract TVT predictions from SDF zero-crossing
    print("Extracting TVT from SDF...")
    sub_rows = []
    for w in test_wells:
        wid = w['wid']
        if wid not in all_preds:
            continue
        sdf_ens = np.mean(all_preds[wid], axis=0)  # [T, H]
        tw_tvt = w['tw_tvt']
        h_tvt = w['h_tvt']
        kn_mask = w['kn_mask']
        ev_idx = np.where(~kn_mask)[0]
        if len(ev_idx) == 0:
            continue
        
        nh = len(h_tvt)
        ps_idx = np.where(kn_mask)[0][-1] if kn_mask.any() else 0
        if nh > H:
            start = max(0, min(ps_idx - H // 4, nh - H))
            h_tvt_seg = h_tvt[start:start + H]
            ev_idx_seg = ev_idx[(ev_idx >= start) & (ev_idx < start + H)] - start
        else:
            h_tvt_seg = np.pad(h_tvt, (0, H - nh), mode='constant', constant_values=np.nan)
            ev_idx_seg = ev_idx
        
        center_idx = int(np.argmin(np.abs(tw_tvt - float(h_tvt[kn_mask][-1])))) if kn_mask.any() else len(tw_tvt) // 2
        t0 = max(0, center_idx - T // 2)
        tw_tvt_crop = tw_tvt[t0:min(t0 + T, len(tw_tvt))]
        
        for i in ev_idx_seg:
            if i < 0 or i >= sdf_ens.shape[1]:
                continue
            col = sdf_ens[:, i]
            zc = int(np.argmin(np.abs(col)))
            tvt_pred = tw_tvt_crop[min(zc, len(tw_tvt_crop) - 1)]
            actual_idx = i + (start if nh > H else 0)
            sub_rows.append({'id': f'{wid}_{actual_idx}', 'tvt': float(tvt_pred)})
    
    # Also add known TVT points
    for w in test_wells:
        wid = w['wid']
        for i, val in enumerate(w['h_tvt']):
            if not np.isnan(val):
                sub_rows.append({'id': f'{wid}_{i}', 'tvt': float(val)})
    
    sub = pd.DataFrame(sub_rows)
    sub = sub.drop_duplicates('id').sort_values('id')
    sub.to_csv("submission.csv", index=False)
    print(f"Saved submission.csv with {len(sub)} rows")
else:
    print("No test samples generated.")
