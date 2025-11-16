import os
import numpy as np
import torch
import torch.nn as nn
from scipy.signal.windows import hamming
import torch.optim as optim

class EarlyStopping:
    """
    Early stops the training if train loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7,
                 verbose=False,
                 delta=0,
                 save_dir='',
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time train loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each train loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_dir (str): Path for the checkpoint to be saved to.
                            Default: ''
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.save_dir = save_dir
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')

        best_model_path = os.path.join(self.save_dir, 'best_model.pt')
        torch.save(model.state_dict(), best_model_path)
        self.val_loss_min = val_loss


def evaluate_model(dataloader, model, device, fs):
    """
    Evaluate the model on a validation dataloader.
    Computes MAE, STD, and BPM difference, std.
    """
    model.eval()
    mae_list, std_list, bpm_diff_list = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            inputs, targets = batch
            inputs = [i.to(device) for i in inputs]
            targets = targets.to(device)

            with torch.no_grad():
                # predictions, scores = model(inputs[1], inputs[0])  # Expected shape: (batch_size, 100)
                # predictions, scores = model(inputs[0], inputs[1])  # Expected shape: (batch_size, 100)
                # predictions = model(inputs[0], inputs[2], inputs[3], inputs[1])  # Expected shape: (batch_size, 100)
                predictions = model(inputs[0], inputs[1], inputs[2], inputs[3])  # Expected shape: (batch_size, 100)

            # Convert tensors to CPU for visualization
            predictions = predictions.cpu().numpy().squeeze()
            targets = targets.cpu().numpy()

            for i in range(len(predictions)):  # Iterate over batch elements
                # Compute BPM using CZT method
                bpm_output = np.array([time_to_bpm(predictions[i], fs)])
                bpm_gt = np.array([time_to_bpm(targets[i], fs)])

                # print(bpm_output.shape)

                # Ensure no NaN values
                valid_indices = ~np.isnan(bpm_output) & ~np.isnan(bpm_gt)
                if np.sum(valid_indices) == 0:
                    continue  # Skip if no valid BPM values

                bpm_output, bpm_gt = bpm_output[valid_indices], bpm_gt[valid_indices]
                # print(bpm_output)

                # Compute Metrics
                mae = np.mean(np.abs(predictions[i] - targets[i]))
                std = np.std(predictions[i] - targets[i])
                bpm_diff = abs(bpm_output - bpm_gt)

                # Store results
                mae_list.append(mae)
                std_list.append(std)
                bpm_diff_list.append(bpm_diff)

            # print('BPM output:', bpm_output)
            # print('BPM ground:', bpm_gt)

    # Compute overall metrics
    avg_mae = np.mean(mae_list)
    avg_std = np.mean(std_list)
    avg_bpm_diff = np.mean(bpm_diff_list)
    std_bpm_diff = np.std(bpm_diff_list)

    # return {"MAE": avg_mae, "STD": avg_std, "BPM Difference": avg_bpm_diff, 'BPM std': std_bpm_diff, 'Gate Score': scores}
    # return {"MAE": avg_mae, "STD": avg_std, "BPM Difference": avg_bpm_diff, 'BPM std': std_bpm_diff, 'Gate Score': model.get_gate_score(inputs[0], inputs[1])}
    return {"MAE": avg_mae, "STD": avg_std, "BPM Difference": avg_bpm_diff, 'BPM std': std_bpm_diff}


def time_to_bpm(y_valid_time, FS, WIN_SIZE=40, STEP=10):
    """
    Compute BPM using the Chirp z-Transform (CZT).
    
    Parameters:
        y_valid_time: np.array (N, M) - Input time-series signals.
        FS: int - Sampling frequency.
        WIN_SIZE: int - Window size for analysis.
        STEP: int - Step size for sliding window.
    
    Returns:
        bpm_values: np.array (N,) - Estimated BPM values.
    """
    y_valid_time = np.atleast_2d(y_valid_time)  # Ensure it's 2D
    N = len(y_valid_time)
    
    z_valid = np.zeros(y_valid_time.shape)
    fmax_valid = np.zeros((N, 1))

    for i in range(N):
        z_valid[i, :], fz = czt(y_valid_time[i], FS, 0.5, 2.0, y_valid_time.shape[1])
        fmax_valid[i] = fz[np.argmax(z_valid[i, :])]  # Get dominant frequency

    bpm_values = fmax_valid * 60  # Convert Hz to BPM
    return bpm_values

def czt(x, fs, fl, fh, M):
    """Compute the Chirp z-Transform (CZT)."""
    fn = np.arange(M) / M
    fz = (fh - fl) * fn + fl
    W = np.exp(-1j * 2 * np.pi * (fh - fl) / (M * fs))
    A = np.exp(1j * 2 * np.pi * fl / fs)

    x = np.asarray(x, dtype=np.complex64) * hamming(M)
    N = x.size
    L = int(2**np.ceil(np.log2(M + N - 1)))

    n = np.arange(N, dtype=float)
    y = np.power(A, -n) * np.power(W, n**2 / 2) * x
    Y = np.fft.fft(y, L)

    v = np.zeros(L, dtype=np.complex64)
    v[:M] = np.power(W, -n[:M]**2 / 2)
    v[L - N + 1:] = np.power(W, -n[N - 1:0:-1]**2 / 2)
    V = np.fft.fft(v)

    g = np.fft.ifft(V * Y)[:M]
    k = np.arange(M)
    g *= np.power(W, k**2 / 2)

    return abs(g), fz


def move_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [move_to_device(x, device) for x in batch]
    return batch.to(device)


def pearson_loss(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return 1 - (torch.sum(x * y) / (torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2))))


def train_model(model, loss_type, lambda_gate, train_dataloader, valid_dataloader,
                device, lr=1e-3, num_epochs=100, save_dir='./checkpoint', use_early_stop=True):
    # Ensure save directory is valid
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    if use_early_stop:
        early_stopping = EarlyStopping(patience=7, save_dir=save_dir)
    if loss_type == 'mse': criterion = nn.MSELoss()
    elif loss_type == 'mae': criterion = nn.L1Loss()
    elif loss_type == 'pearson': criterion = pearson_loss
    elif loss_type == 'mixed': criterion = nn.L1Loss()
    else: raise ValueError('Support mse, mae, pearson, mixed (mae + pearson) losses only')
    if hasattr(criterion, "to"):
        criterion.to(device)

    train_losses, valid_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch in train_dataloader:
            batch = move_to_device(batch, device)
            inputs, targets = batch
            optimizer.zero_grad()
            inputs = [inp.to(device) for inp in inputs]
            targets = targets.to(device)

            outputs = model(inputs[0], inputs[1], inputs[2], inputs[3])
            if loss_type == 'mixed':
                mae = criterion(outputs.squeeze(), targets.float())
                pearson = pearson_loss(outputs.squeeze(), targets.float())
                # gate = scores.clamp(1e-6, 1 - 1e-6)
                # gate_entropy_loss = -(gate * torch.log(gate) + (1-gate)*torch.log(1-gate)).mean()
                # loss = mse_loss + lambda_gate * gate_entropy_loss
                loss = mae + 0.1 * pearson
            else:
                loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        epoch_valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_dataloader:
                batch = move_to_device(batch, device)
                inputs, targets = batch
                inputs = [inp.to(device) for inp in inputs]
                targets = targets.to(device)

                outputs = model(inputs[0], inputs[1], inputs[2], inputs[3])
                if loss_type == 'mixed':
                    mae = criterion(outputs.squeeze(), targets.float())
                    pearson = pearson_loss(outputs.squeeze(), targets.float())
                    # gate = scores.clamp(1e-6, 1 - 1e-6)
                    # gate_entropy_loss = -(gate * torch.log(gate) + (1-gate)*torch.log(1-gate)).mean()
                    # loss = mse_loss + lambda_gate * gate_entropy_loss
                    loss = mae + 0.1 * pearson
                else:
                    loss = criterion(outputs.squeeze(), targets.float())
                epoch_valid_loss += loss.item()

        avg_valid_loss = epoch_valid_loss / len(valid_dataloader)
        valid_losses.append(avg_valid_loss)

        # Print Epoch Summary
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")

        # Save model every 10 epochs if save_dir is provided
        if save_dir is not None and (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model saved at epoch {epoch+1} to {checkpoint_path}')

        if epoch >= 12:
            scheduler.step(avg_valid_loss)  # Adjust learning rate based on validation loss
            if use_early_stop:
                early_stopping(avg_valid_loss, model)  # Early stopping

        if use_early_stop and early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, valid_losses