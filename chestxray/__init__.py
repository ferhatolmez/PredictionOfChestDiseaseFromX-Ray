"""Chest X-ray classification: data loading, preprocessing, model, training.

Modules:
    config         all paths and hyperparameters in one place
    data           listing and loading images, patient IDs, the grouped split
    preprocessing  white balance
    audit          dataset checks: duplicates and non-medical shortcuts
    pipeline       tf.data input pipeline and data augmentation
    model          MobileNetV2 transfer-learning model
    train          two-phase training (command line entry point)
    evaluate       metrics and plots
    gradcam        Grad-CAM heatmaps
"""
