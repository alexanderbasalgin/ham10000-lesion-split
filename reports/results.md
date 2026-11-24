| model | split | bal_acc | 95% CI | macro F1 | accuracy | mel recall | mel sens @90% spec | malignant AUC |
|---|---|---|---|---|---|---|---|---|
| class prior | lesion | 0.143 | 0.143-0.143 | 0.117 | 0.691 | 0.00 | 0.00 | 0.500 |
| color histograms + LR | lesion | 0.478 | 0.416-0.551 | 0.318 | 0.509 | 0.53 | 0.36 | 0.790 |
| DINOv2-S frozen + LR | lesion | 0.651 | 0.585-0.721 | 0.535 | 0.701 | 0.49 | 0.45 | 0.868 |
| DINOv2-S frozen + metadata + LR | lesion | 0.728 | 0.670-0.781 | 0.599 | 0.732 | 0.60 | 0.52 | 0.886 |
| color histograms + LR | image | 0.472 | 0.422-0.518 | 0.334 | 0.509 | 0.49 | 0.38 | 0.781 |
| DINOv2-S frozen + LR | image | 0.684 | 0.627-0.737 | 0.567 | 0.713 | 0.56 | 0.48 | 0.894 |
| ConvNeXt-T fine-tuned | lesion | 0.798 | 0.736-0.849 | 0.745 | 0.828 | 0.64 | 0.75 | 0.915 |
| ConvNeXt-T fine-tuned + metadata | lesion | 0.797 | 0.745-0.840 | 0.728 | 0.834 | 0.62 | 0.75 | 0.922 |
| ConvNeXt-T fine-tuned, plain CE | lesion | 0.754 | 0.691-0.810 | 0.743 | 0.864 | 0.57 | 0.79 | 0.912 |
| ConvNeXt-T fine-tuned | image | 0.826 | 0.790-0.856 | 0.811 | 0.890 | 0.72 | 0.94 | 0.944 |

Mean ± std over split seeds:

| model | split | seeds | balanced_accuracy | macro_f1 | accuracy | mel_sens_at_90spec | malignant_auc |
|---|---|---|---|---|---|---|---|
| DINOv2-S frozen + LR | lesion | 3 | 0.679 ± 0.028 | 0.550 ± 0.025 | 0.716 ± 0.011 | 0.487 ± 0.030 | 0.880 ± 0.012 |
| DINOv2-S frozen + LR | image | 3 | 0.706 ± 0.017 | 0.584 ± 0.012 | 0.722 ± 0.006 | 0.503 ± 0.027 | 0.900 ± 0.006 |
| DINOv2-S frozen + metadata + LR | lesion | 3 | 0.713 ± 0.018 | 0.581 ± 0.024 | 0.739 ± 0.005 | 0.514 ± 0.011 | 0.893 ± 0.009 |
| ConvNeXt-T fine-tuned | lesion | 3 | 0.789 ± 0.007 | 0.734 ± 0.014 | 0.817 ± 0.011 | 0.728 ± 0.034 | 0.922 ± 0.011 |
| ConvNeXt-T fine-tuned | image | 3 | 0.857 ± 0.023 | 0.853 ± 0.031 | 0.904 ± 0.010 | 0.897 ± 0.029 | 0.949 ± 0.007 |
| ConvNeXt-T fine-tuned + metadata | lesion | 3 | 0.771 ± 0.019 | 0.754 ± 0.033 | 0.849 ± 0.020 | 0.738 ± 0.015 | 0.919 ± 0.002 |
