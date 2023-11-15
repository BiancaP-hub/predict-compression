PARTICIPANTS_FILE = 'data/participants.tsv'
IMG = '_T2w.nii.gz'
SEG_DIR = 'generated_data/seg'
SEG_IMG = '_T2w_seg.nii.gz'
LABELED_SEG_DIR = 'generated_data/labeled_seg'
LABELED_SEG_IMG = '_T2w_seg_labeled.nii.gz'
COMPRESSION_DIR = 'generated_data/compression_labels'
METRICS_DIR = 'generated_data/metrics_clean'
METRICS_FILE = '_T2w_seg_metrics.csv'
PAM50_METRICS_FILE = '_T2w_seg_metrics_PAM50.csv'
MSCC_LABELS_FILE = 'generated_data/ap_ratio_norm_PAM50.csv'

MODEL_TYPES = ['cnn', 'lstm', 'cnn_lstm', 'random_forest', 'gradient_boosting']