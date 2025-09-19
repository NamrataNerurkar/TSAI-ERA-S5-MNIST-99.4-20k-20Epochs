# MNIST_ERA Notebook - Session 5

This notebook implements a Convolutional Neural Network (CNN) for MNIST digit classification, exploring various architectural improvements and optimization techniques.

## Final Architecture (Code Block 7.1)

The final architecture used in the notebook is `NetCustomArch` with the following structure:

```python
class NetCustomArch(nn.Module): # OLD MODEL
    def __init__(self):
        super(NetCustomArch, self).__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)    # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)   # 28x28 -> 28x28 (will pool after)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)   # 7x7 -> 7x7
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)   # 3x3 -> 3x3
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 16, 1)             # 3x3 -> 3x3 bottleneck
        self.bn5 = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(16, 16, 3)             # 3x3 -> 3x3
        self.bn6 = nn.BatchNorm2d(16)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # final output: [batch, 16, 1, 1]

        # Fully connected
        self.fc = nn.Linear(16, 10)
        self.dropout = nn.Dropout(0.5)
```

### Architecture Summary

The model uses:
- **6 Convolutional layers** with Batch Normalization
- **MaxPooling** for spatial dimension reduction
- **Global Average Pooling** instead of fully connected layers
- **1x1 convolution bottleneck** for parameter reduction
- **Dropout** for regularization

### Parameter Count

```
Total params: 19,642
Trainable params: 19,642
Non-trainable params: 0
```

## Training Results (Code Block 10)

The model was trained for 20 epochs with the following configuration:
- **Optimizer**: Adam with learning rate 0.001 and weight decay 1e-4
- **Scheduler**: StepLR with step_size=15 and gamma=0.1
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64

### Final Training Results

| Epoch | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| 1     | 76.20%        | 98.48%        |
| 2     | 85.16%        | 98.77%        |
| 3     | 86.60%        | 98.28%        |
| 4     | 86.84%        | 99.10%        |
| 5     | 87.18%        | 99.26%        |
| 10    | 88.63%        | 98.50%        |
| 15    | 90.44%        | 99.36%        |
| 20    | 90.84%        | 99.41%        |

**Final Performance:**
- **Train Accuracy**: 90.84%
- **Test Accuracy**: 99.41%

### Training Logs

Epoch 1
Train: Loss=0.4579 Batch_id=937 Accuracy=76.20: 100%|██████████| 938/938 [00:20<00:00, 45.02it/s]
Test set: Average loss: 0.0011, Accuracy: 9848/10000 (98.48%)

Epoch 2
Train: Loss=0.4037 Batch_id=937 Accuracy=85.16: 100%|██████████| 938/938 [00:20<00:00, 46.55it/s]
Test set: Average loss: 0.0007, Accuracy: 9877/10000 (98.77%)

Epoch 3
Train: Loss=0.3578 Batch_id=937 Accuracy=86.60: 100%|██████████| 938/938 [00:20<00:00, 46.64it/s]
Test set: Average loss: 0.0009, Accuracy: 9828/10000 (98.28%)

Epoch 4
Train: Loss=0.3325 Batch_id=937 Accuracy=86.84: 100%|██████████| 938/938 [00:20<00:00, 44.80it/s]
Test set: Average loss: 0.0006, Accuracy: 9910/10000 (99.10%)

Epoch 5
Train: Loss=0.3930 Batch_id=937 Accuracy=87.18: 100%|██████████| 938/938 [00:20<00:00, 45.30it/s]
Test set: Average loss: 0.0004, Accuracy: 9926/10000 (99.26%)

Epoch 6
Train: Loss=0.2923 Batch_id=937 Accuracy=87.49: 100%|██████████| 938/938 [00:21<00:00, 44.16it/s]
Test set: Average loss: 0.0007, Accuracy: 9873/10000 (98.73%)

Epoch 7
Train: Loss=0.2983 Batch_id=937 Accuracy=87.84: 100%|██████████| 938/938 [00:20<00:00, 45.03it/s]
Test set: Average loss: 0.0005, Accuracy: 9917/10000 (99.17%)

Epoch 8
Train: Loss=0.3139 Batch_id=937 Accuracy=88.06: 100%|██████████| 938/938 [00:20<00:00, 44.75it/s]
Test set: Average loss: 0.0007, Accuracy: 9898/10000 (98.98%)

Epoch 9
Train: Loss=0.3317 Batch_id=937 Accuracy=88.50: 100%|██████████| 938/938 [00:20<00:00, 46.02it/s]
Test set: Average loss: 0.0009, Accuracy: 9844/10000 (98.44%)

Epoch 10
Train: Loss=0.2829 Batch_id=937 Accuracy=88.63: 100%|██████████| 938/938 [00:19<00:00, 47.41it/s]
Test set: Average loss: 0.0009, Accuracy: 9850/10000 (98.50%)

Epoch 11
Train: Loss=0.2079 Batch_id=937 Accuracy=89.06: 100%|██████████| 938/938 [00:20<00:00, 45.97it/s]
Test set: Average loss: 0.0005, Accuracy: 9914/10000 (99.14%)

Epoch 12
Train: Loss=0.2633 Batch_id=937 Accuracy=89.11: 100%|██████████| 938/938 [00:20<00:00, 45.02it/s]
Test set: Average loss: 0.0005, Accuracy: 9925/10000 (99.25%)

Epoch 13
Train: Loss=0.2786 Batch_id=937 Accuracy=89.29: 100%|██████████| 938/938 [00:20<00:00, 45.75it/s]
Test set: Average loss: 0.0005, Accuracy: 9921/10000 (99.21%)

Epoch 14
Train: Loss=0.2009 Batch_id=937 Accuracy=89.82: 100%|██████████| 938/938 [00:20<00:00, 44.91it/s]
Test set: Average loss: 0.0005, Accuracy: 9933/10000 (99.33%)

Epoch 15
Train: Loss=0.1393 Batch_id=937 Accuracy=89.96: 100%|██████████| 938/938 [00:20<00:00, 45.33it/s]
Test set: Average loss: 0.0005, Accuracy: 9922/10000 (99.22%)

Epoch 16
Train: Loss=0.2770 Batch_id=937 Accuracy=90.29: 100%|██████████| 938/938 [00:20<00:00, 46.04it/s]
Test set: Average loss: 0.0005, Accuracy: 9937/10000 (99.37%)

Epoch 17
Train: Loss=0.1775 Batch_id=937 Accuracy=90.44: 100%|██████████| 938/938 [00:20<00:00, 45.57it/s]
Test set: Average loss: 0.0005, Accuracy: 9936/10000 (99.36%)

Epoch 18
Train: Loss=0.2266 Batch_id=937 Accuracy=90.50: 100%|██████████| 938/938 [00:20<00:00, 44.98it/s]
Test set: Average loss: 0.0004, Accuracy: 9940/10000 (99.40%)

Epoch 19
Train: Loss=0.3962 Batch_id=937 Accuracy=90.68: 100%|██████████| 938/938 [00:20<00:00, 45.41it/s]
Test set: Average loss: 0.0004, Accuracy: 9939/10000 (99.39%)

Epoch 20
Train: Loss=0.2127 Batch_id=937 Accuracy=90.84: 100%|██████████| 938/938 [00:20<00:00, 45.02it/s]
Test set: Average loss: 0.0004, Accuracy: 9941/10000 (99.41%)

## Code Changes

The following modifications were made to the original code:

1. **Device Configuration**: Added explicit device selection for CUDA/CPU
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # changed: added this line
   ```

2. **Optimizer Changes**: Switched from SGD to Adam optimizer with weight decay
   ```python
   # Changed from SGD to Adam with L2 regularization
   optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
   ```

3. **Learning Rate Scheduler**: Modified StepLR scheduler
   ```python
   scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) # changed: removed verbose=True
   ```

4. **Training Epochs**: Increased from 19 to 20 epochs
   ```python
   # for epoch in range(1, 19): # changed # epochs
   num_epochs = 20
   ```

5. **Loss Function**: Updated to use CrossEntropyLoss
   ```python
   criterion = nn.CrossEntropyLoss() # changed: can add reduction='sum'
   ```

6. **Test Function**: Fixed test function call
   ```python
   test(model, device, test_loader, criterion) # changed: train to test_loader
   ```

## Iterations & Experiments

The notebook explores several architectural iterations and experiments:

### 1. **Baseline Architecture (S5)**
- Simple CNN with 7 convolutional layers
- Progressive channel increase: 1→2→4→8→16→32→64→128
- Two MaxPool layers for spatial reduction
- Large fully connected layer (6272→50→10)
- **Total Parameters**: ~6,272 + 50×6272 + 10×50 = ~320K parameters

### 2. **GAP + BatchNorm + Dropout**
- Added Batch Normalization after each conv layer
- Implemented Global Average Pooling (GAP) instead of FC layers
- Added Dropout for regularization
- **Parameter Reduction**: From ~320K to ~128 parameters

### 3. **1x1 Bottleneck Architecture**
- Introduced 1x1 convolutions for channel reduction
- Bottleneck pattern: 64→32→64 channels
- Further parameter reduction while maintaining representational capacity

### 4. **Experiment 1: Small Layers**
- Reduced channel dimensions for efficiency
- Removed one conv layer (conv6)
- Bottleneck: 32→16→32 channels
- **Total Parameters**: ~32 parameters

### 5. **Fixed Size Architecture (Multiple Variants)**
- Explored different pooling strategies
- Varied bottleneck placements
- Tested different channel configurations
- Multiple iterations with different spatial dimensions

### 6. **Final Custom Architecture**
- Optimized 6-layer CNN with strategic pooling
- 1x1 bottleneck for parameter efficiency
- Global Average Pooling for final classification
- **Final Parameters**: 19,642 (significantly reduced from baseline)

### Key Architectural Insights

1. **Global Average Pooling**: Dramatically reduces parameters while maintaining performance
2. **Batch Normalization**: Improves training stability and convergence
3. **1x1 Convolutions**: Effective for channel reduction without losing information
4. **Strategic Pooling**: Careful placement of pooling layers for optimal spatial reduction
5. **Dropout**: Essential for preventing overfitting in smaller networks

The final architecture achieves 99.41% test accuracy with only 19,642 parameters, demonstrating the effectiveness of modern CNN design principles for efficient digit classification.
