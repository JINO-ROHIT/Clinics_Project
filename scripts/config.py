# ====================================================
# CFG
# ====================================================
class CFG:
    apex = False
    debug = False
    print_freq = 10
    num_workers = 4
    size = 224
    model_name='resnet18'
    scheduler='CosineAnnealingLR'
    criterion='CrossEntropyLoss'
    epochs = 40
    T_max = 3 
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 10
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    target_size = meta_data.corrected_labels.nunique()
    target_col = 'corrected_labels'
    train = True
    smoothing = 0.05
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    
if CFG.debug:
    CFG.epochs = 1
    train = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)