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
