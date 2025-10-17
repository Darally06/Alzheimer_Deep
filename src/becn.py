import time
import torch
import numpy as np
import math

# Asume que tienes: model, train_loader, device (ya definidos)
model.eval()  # no backprop, medimos forward+back+opt step below though

# Benchmark parameters
n_warmup = 10     # batches para 'calentar' GPU
n_measure = 50    # batches para medir (ajusta si tu epoch es más corto)
batch_limit = n_warmup + n_measure

# We will perform a full training-like step (forward+loss+backward+opt.step)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# move model to device (if not already)
model.to(device)

times = []
batch_i = 0
print("Iniciando benchmark — this will run ~ (n_warmup+n_measure) batches with training step...")

for vols, labels in train_loader:
    if batch_i >= batch_limit:
        break

    vols = vols.to(device)
    labels = labels.to(device).unsqueeze(1).float()

    # warm-up without timing
    if batch_i < n_warmup:
        optimizer.zero_grad()
        outs = model(vols)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        batch_i += 1
        continue

    # timed batches
    t0 = time.time()
    optimizer.zero_grad()
    outs = model(vols)
    loss = criterion(outs, labels)
    loss.backward()
    optimizer.step()
    t1 = time.time()

    times.append(t1 - t0)
    batch_i += 1
    if (batch_i - n_warmup) % 10 == 0:
        print(f"  medidos {(batch_i - n_warmup)} / {n_measure} batches")

if len(times) == 0:
    raise RuntimeError("El dataloader no tiene suficientes batches para medir.")

avg_batch = np.mean(times)
std_batch = np.std(times)
n_train = len(train_loader.dataset)  # 165 en tu caso
batch_size = train_loader.batch_size
batches_per_epoch = math.ceil(n_train / batch_size)
time_per_epoch = avg_batch * batches_per_epoch
n_epochs = 10
est_total = time_per_epoch * n_epochs

print("\n--- Resultado benchmark ---")
print(f"Avg time / batch (train step): {avg_batch:.3f} s  (std {std_batch:.3f} s) ")
print(f"Batches/epoch: {batches_per_epoch}")
print(f"Estimated time / epoch: {time_per_epoch/60:.2f} min")
print(f"Estimated time for {n_epochs} epochs: {est_total/60:.2f} min")
