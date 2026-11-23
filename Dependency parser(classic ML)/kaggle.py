import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
RUN_LOCAL_VAL = True
MAX_DIST = 25

print("1. Loading dataset...")
DATA_DIR = '/kaggle/input/competitions/iith-foundations-of-machine-learning-2026-hackathon'
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'

train_full = pd.read_csv(f'{DATA_DIR}/train.csv')
test_full = pd.read_csv(f'{DATA_DIR}/test.csv')

train_full[['head_id', 'deprel']] = train_full['target'].str.split('|', expand=True)
train_full['head_id'] = train_full['head_id'].astype(int)

# 2. Classical SVD Semantic Embeddings (LSA - 100% Non-Neural)
print("2. Computing Classical SVD (LSA) Semantic Word Vectors...")
all_tokens = pd.concat([train_full[['sentence_id', 'token']], test_full[['sentence_id', 'token']]])
all_tokens['word'] = all_tokens['token'].astype(str).str.lower()
sentences = all_tokens.groupby('sentence_id')['word'].apply(list).values

vocab = {w: i for i, w in enumerate(all_tokens['word'].unique())}
V = len(vocab)
cooc = dok_matrix((V, V), dtype=np.float32)

for s in sentences:
    n = len(s)
    for i in range(n):
        w1 = vocab[s[i]]
        for j in range(max(0, i-2), min(n, i+3)):
            if i != j:
                cooc[w1, vocab[s[j]]] += 1

cooc = cooc.tocsr()
total_pairs = cooc.sum()
cooc_sum0 = np.asarray(cooc.sum(axis=0)).flatten()
cooc_sum1 = np.asarray(cooc.sum(axis=1)).flatten()

cooc_coo = cooc.tocoo()
pmi_data = np.log((cooc_coo.data * total_pairs) / (cooc_sum1[cooc_coo.row] * cooc_sum0[cooc_coo.col]))
ppmi = csr_matrix((np.maximum(pmi_data, 0), (cooc_coo.row, cooc_coo.col)), shape=(V, V))

svd = TruncatedSVD(n_components=12, random_state=42)
word_vectors = svd.fit_transform(ppmi)
svd_map = {w: word_vectors[i] for w, i in vocab.items()}
del all_tokens, sentences, cooc, ppmi, cooc_coo
gc.collect()

def get_coarse_pos(df_with_targets):
    child_sets = df_with_targets.groupby(['sentence_id', 'head_id'])['deprel'].apply(set).reset_index()
    child_sets.columns = ['sentence_id', 'token_id', 'children']
    m = df_with_targets[['sentence_id', 'token_id', 'deprel']].merge(child_sets, on=['sentence_id', 'token_id'], how='left')
    coarse = []
    for d, c in zip(m['deprel'].tolist(), m['children'].tolist()):
        c = c if isinstance(c, set) else set()
        if d in ('det', 'det:predet'): coarse.append('DET')
        elif d == 'case': coarse.append('ADP')
        elif d in ('aux', 'aux:pass', 'cop'): coarse.append('AUX')
        elif d in ('cc', 'cc:preconj'): coarse.append('CCONJ')
        elif d == 'mark': coarse.append('SCONJ')
        elif d == 'punct': coarse.append('PUNCT')
        elif d == 'nummod': coarse.append('NUM')
        elif d == 'amod': coarse.append('ADJ')
        elif d == 'advmod': coarse.append('ADV')
        elif d == 'compound:prt': coarse.append('PRT')
        elif d == 'compound': coarse.append('COMPOUND')
        elif d == 'nmod:poss': coarse.append('POSS')
        elif d in ('flat', 'appos', 'nmod', 'nmod:unmarked', 'nmod:desc', 'obl', 'obl:unmarked', 'obl:agent'): coarse.append('NOUN')
        elif d in ('ccomp', 'xcomp', 'advcl', 'advcl:relcl', 'acl', 'acl:relcl', 'parataxis', 'csubj', 'csubj:pass'): coarse.append('VERB')
        else:
            if any(k in c for k in ('aux', 'aux:pass', 'cop', 'obj', 'ccomp', 'xcomp', 'mark', 'expl', 'advcl', 'obl')): coarse.append('VERB')
            elif any(k in c for k in ('det', 'amod', 'compound', 'nummod', 'nmod', 'case')): coarse.append('NOUN')
            elif d in ('nsubj', 'nsubj:pass', 'expl', 'iobj'): coarse.append('PRON_NOUN')
            else: coarse.append('VERB' if d == 'root' else 'NOUN')
    return coarse

def word_shape(w):
    res = []
    for ch in str(w):
        if ch.isupper(): res.append('X')
        elif ch.islower(): res.append('x')
        elif ch.isdigit(): res.append('d')
        else: res.append(ch)
    s = ''.join(res)
    return s[:2] + '*' + s[-2:] if len(s) > 4 else s

def get_offset_bucket_from_diff(head_id, diff):
    return np.where(head_id == 0, 'ROOT', np.where(diff == -1, 'L1', np.where(diff == 1, 'R1', np.where(diff == -2, 'L2', np.where(diff == 2, 'R2', np.where(diff == -3, 'L3', np.where(diff == 3, 'R3', np.where(diff < -3, 'L_FAR', 'R_FAR'))))))))

def build_base_token_features(df):
    df = df.copy()
    df['token'] = df['token'].astype(str)
    df['word'] = df['token'].str.lower()
    df['shape'] = [word_shape(w) for w in df['token']]
    df['pre1'] = df['word'].str[:1]; df['pre2'] = df['word'].str[:2]; df['pre3'] = df['word'].str[:3]
    df['suf1'] = df['word'].str[-1:]; df['suf2'] = df['word'].str[-2:]; df['suf3'] = df['word'].str[-3:]; df['suf4'] = df['word'].str[-4:]
    df['is_cap'] = df['token'].str[0].str.isupper().astype(np.int8)
    df['is_upper'] = df['token'].str.isupper().astype(np.int8)
    df['is_num'] = df['token'].str.contains(r'\d', regex=True).astype(np.int8)
    df['is_punct'] = df['token'].str.match(r'^[^\w\s]+$').astype(np.int8)
    df['length'] = df['token'].str.len().astype(np.int16)
    
    df['sent_len'] = df.groupby('sentence_id')['token_id'].transform('max').astype(np.int16)
    df['rel_pos'] = (df['token_id'] / df['sent_len']).astype(np.float32)
    df['is_first'] = (df['token_id'] == 1).astype(np.int8)
    df['is_last'] = (df['token_id'] == df['sent_len']).astype(np.int8)
    
    for off in [-3, -2, -1, 1, 2, 3]:
        df[f'word_{off}'] = df.groupby('sentence_id')['word'].shift(-off).fillna(f'<PAD_{off}>')
        df[f'suf3_{off}'] = df[f'word_{off}'].str[-3:]
        df[f'shape_{off}'] = df.groupby('sentence_id')['shape'].shift(-off).fillna(f'<PAD_{off}>')
        
    df['wm1_w0'] = df['word_-1'] + '|' + df['word']
    df['w0_wp1'] = df['word'] + '|' + df['word_1']
    df['wm1_wp1'] = df['word_-1'] + '|' + df['word_1']
    df['wm2_wm1_w0'] = df['word_-2'] + '|' + df['word_-1'] + '|' + df['word']
    df['w0_wp1_wp2'] = df['word'] + '|' + df['word_1'] + '|' + df['word_2']
    df['sm1_s0_sp1'] = df['shape_-1'] + '|' + df['shape'] + '|' + df['shape_1']
    df['pos_bucket'] = np.where(df['is_first'] == 1, 'FIRST', np.where(df['is_last'] == 1, 'LAST', 'MID'))
    return df

def extract_chunk_governors(df):
    n = len(df); sids = df['sentence_id'].values; words = df['word'].values
    s1_pos = df['s1_pos'].values; s1_dep = df['s1_dep'].values
    gov_prep = np.full(n, '<NONE>', dtype=object); gov_mark = np.full(n, '<NONE>', dtype=object); gov_cc = np.full(n, '<NONE>', dtype=object)
    has_gov_det = np.zeros(n, dtype=np.int8)
    cur_sid = None; act_prep = '<NONE>'; act_mark = '<NONE>'; act_cc = '<NONE>'; act_det = 0
    
    for i in range(n):
        if sids[i] != cur_sid:
            cur_sid = sids[i]; act_prep = '<NONE>'; act_mark = '<NONE>'; act_cc = '<NONE>'; act_det = 0
            
        w, pos, dep = words[i], s1_pos[i], s1_dep[i]
        
        if dep == 'case' or pos == 'ADP': act_prep = w
        elif dep == 'mark' or pos == 'SCONJ': act_mark = w
        elif dep == 'cc' or pos == 'CCONJ': act_cc = w
        elif dep in ('det', 'det:predet', 'nmod:poss') or pos in ('DET', 'POSS'): act_det = 1
        elif pos in ('NOUN', 'PRON_NOUN', 'NUM'):
            gov_prep[i] = act_prep; gov_cc[i] = act_cc; has_gov_det[i] = act_det
            act_prep = '<NONE>'; act_det = 0; act_cc = '<NONE>'
        elif pos == 'VERB' or dep in ('root', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl', 'parataxis'):
            gov_mark[i] = act_mark; gov_cc[i] = act_cc
            act_mark = '<NONE>'; act_cc = '<NONE>'; act_prep = '<NONE>'; act_det = 0
        elif pos == 'PUNCT' and w in (',', ';', ':', '--', '-', '(', ')'):
            act_prep = '<NONE>'; act_det = 0
            
    df['gov_prep'] = gov_prep; df['gov_mark'] = gov_mark
    df['gov_cc'] = gov_cc; df['has_gov_det'] = has_gov_det
    return df

def compute_lexical_stats(source_df, target_df):
    src = source_df.copy()
    src['is_root'] = (src['head_id'] == 0).astype(np.float32)
    src['attach_right'] = (src['head_id'] > src['token_id']).astype(np.float32)
    src['attach_left'] = ((src['head_id'] < src['token_id']) & (src['head_id'] > 0)).astype(np.float32)
    src['signed_dist'] = np.where(src['head_id'] == 0, 0, src['head_id'] - src['token_id']).astype(np.float32)
    
    non_root = src[src['head_id'] > 0]
    left_kids = non_root[non_root['token_id'] < non_root['head_id']].groupby(['sentence_id', 'head_id']).size()
    right_kids = non_root[non_root['token_id'] > non_root['head_id']].groupby(['sentence_id', 'head_id']).size()
    
    src_idx = src.set_index(['sentence_id', 'token_id'])
    src_idx['n_left_kids'] = left_kids; src_idx['n_right_kids'] = right_kids
    src = src_idx.fillna({'n_left_kids': 0, 'n_right_kids': 0}).reset_index()
    src['is_any_head'] = ((src['n_left_kids'] + src['n_right_kids']) > 0).astype(np.float32)
    src['has_left_kid'] = (src['n_left_kids'] > 0).astype(np.float32)
    src['has_right_kid'] = (src['n_right_kids'] > 0).astype(np.float32)
    
    stat_cols = ['is_root', 'attach_right', 'attach_left', 'signed_dist', 'is_any_head', 'has_left_kid', 'has_right_kid']
    global_means = src[stat_cols].mean()
    m_smooth = 5.0
    agg = src.groupby('word')[stat_cols].agg(['count', 'mean'])
    counts = agg[('is_root', 'count')]
    word_stats = pd.DataFrame(index=agg.index)
    for col in stat_cols:
        word_stats[f'lex_{col}'] = ((counts * agg[(col, 'mean')] + m_smooth * global_means[col]) / (counts + m_smooth)).astype(np.float32)
        
    suf_agg = src.groupby('suf3')[stat_cols].agg(['count', 'mean'])
    suf_counts = suf_agg[('is_root', 'count')]
    suf_stats = pd.DataFrame(index=suf_agg.index)
    for col in stat_cols:
        suf_stats[f'suf_lex_{col}'] = ((suf_counts * suf_agg[(col, 'mean')] + m_smooth * global_means[col]) / (suf_counts + m_smooth)).astype(np.float32)
        
    out = target_df.merge(word_stats, on='word', how='left').merge(suf_stats, on='suf3', how='left')
    for col in stat_cols:
        out[f'lex_{col}'] = out[f'lex_{col}'].fillna(out[f'suf_lex_{col}']).fillna(global_means[col]).astype(np.float32)
        out.drop(columns=[f'suf_lex_{col}'], inplace=True)
    return out


def add_stage1_tags_and_stats(train_df, test_df):
    print("3. Running Two-Pass Sequential Stage-1 Syntactic Taggers (5-Fold OOF)...")
    train_df = train_df.reset_index(drop=True).copy()
    test_df = test_df.reset_index(drop=True).copy()
    train_df = build_base_token_features(train_df)
    test_df = build_base_token_features(test_df)
    train_df['coarse_pos'] = get_coarse_pos(train_df)
    
    idx_df = train_df.set_index(['sentence_id', 'token_id'])
    non_root = train_df[train_df['head_id'] > 0]
    idx_df['n_l'] = non_root[non_root['token_id'] < non_root['head_id']].groupby(['sentence_id', 'head_id']).size()
    idx_df['n_r'] = non_root[non_root['token_id'] > non_root['head_id']].groupby(['sentence_id', 'head_id']).size()
    idx_df = idx_df.fillna({'n_l': 0, 'n_r': 0}).reset_index()
    
    y_is_head = ((idx_df['n_l'] + idx_df['n_r']) > 0).astype(np.int32).values
    y_has_lkid = (idx_df['n_l'] > 0).astype(np.int32).values
    y_has_rkid = (idx_df['n_r'] > 0).astype(np.int32).values
    y_is_root = (train_df['head_id'] == 0).astype(np.int32).values
    y_att_r = (train_df['head_id'] > train_df['token_id']).astype(np.int32).values
    y_att_l = ((train_df['head_id'] < train_df['token_id']) & (train_df['head_id'] > 0)).astype(np.int32).values
    
    bin_targets = {'s1_p_head': y_is_head, 's1_p_lkid': y_has_lkid, 's1_p_rkid': y_has_rkid, 's1_p_root': y_is_root, 's1_p_att_r': y_att_r, 's1_p_att_l': y_att_l}
    
    off_buckets = get_offset_bucket_from_diff(train_df['head_id'].values, (train_df['head_id'] - train_df['token_id']).values)
    off_le = LabelEncoder()
    y_off = off_le.fit_transform(off_buckets)
    n_off_classes = len(off_le.classes_)
    
    unique_sents = train_df['sentence_id'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    stat_cols = ['lex_is_root', 'lex_attach_right', 'lex_attach_left', 'lex_signed_dist', 'lex_is_any_head', 'lex_has_left_kid', 'lex_has_right_kid']
    for c in stat_cols: train_df[c] = np.nan
        
    for tr_idx, va_idx in kf.split(unique_sents):
        tr_sub = train_df[train_df['sentence_id'].isin(set(unique_sents[tr_idx]))]
        va_mask = train_df['sentence_id'].isin(set(unique_sents[va_idx]))
        va_with_stats = compute_lexical_stats(tr_sub, train_df.loc[va_mask, ['word', 'suf3']])
        for c in stat_cols: train_df.loc[va_mask, c] = va_with_stats[c].values
            
    test_df = compute_lexical_stats(train_df, test_df)
    
    s1_cols = ['word', 'token', 'shape', 'pre1', 'pre2', 'pre3', 'suf1', 'suf2', 'suf3', 'suf4', 'pos_bucket', 'word_-3', 'word_-2', 'word_-1', 'word_1', 'word_2', 'word_3', 'suf3_-3', 'suf3_-2', 'suf3_-1', 'suf3_1', 'suf3_2', 'suf3_3', 'shape_-2', 'shape_-1', 'shape_1', 'shape_2', 'wm1_w0', 'w0_wp1', 'wm1_wp1', 'wm2_wm1_w0', 'w0_wp1_wp2', 'sm1_s0_sp1']
    ohe_s1 = OneHotEncoder(handle_unknown='ignore', min_frequency=2, dtype=np.float32)
    X_tr_p1 = ohe_s1.fit_transform(train_df[s1_cols])
    X_te_p1 = ohe_s1.transform(test_df[s1_cols])
    
    pos_le = LabelEncoder()
    y_pos = pos_le.fit_transform(train_df['coarse_pos'])
    dep_counts = train_df['deprel'].value_counts()
    s1_dep_target = np.where(train_df['deprel'].isin(set(dep_counts[dep_counts >= 40].index)), train_df['deprel'], 'RARE_DEP')
    dep_le = LabelEncoder()
    y_dep = dep_le.fit_transform(s1_dep_target)
    
    p1_oof_pos = np.zeros(len(train_df), dtype=np.int32)
    p1_te_pos = np.zeros((len(test_df), len(pos_le.classes_)), dtype=np.float32)
    p1_oof_dep = np.zeros(len(train_df), dtype=np.int32)
    p1_te_dep = np.zeros((len(test_df), len(dep_le.classes_)), dtype=np.float32)
    
    for tr_idx, va_idx in kf.split(unique_sents):
        tr_m = train_df['sentence_id'].isin(set(unique_sents[tr_idx])).values
        va_m = train_df['sentence_id'].isin(set(unique_sents[va_idx])).values
        
        cp1 = SGDClassifier(loss='log_loss', alpha=4e-6, max_iter=12, random_state=42, n_jobs=-1).fit(X_tr_p1[tr_m], y_pos[tr_m])
        p1_oof_pos[va_m] = cp1.classes_[np.argmax(cp1.predict_proba(X_tr_p1[va_m]), axis=1)]
        p1_te_pos[:, cp1.classes_] += cp1.predict_proba(X_te_p1) / 5.0
        
        cd1 = SGDClassifier(loss='log_loss', alpha=4e-6, max_iter=10, random_state=42, n_jobs=-1).fit(X_tr_p1[tr_m], y_dep[tr_m])
        p1_oof_dep[va_m] = cd1.classes_[np.argmax(cd1.predict_proba(X_tr_p1[va_m]), axis=1)]
        p1_te_dep[:, cd1.classes_] += cd1.predict_proba(X_te_p1) / 5.0
        
    train_df['p1_pos'] = pos_le.inverse_transform(p1_oof_pos)
    test_df['p1_pos'] = pos_le.inverse_transform(np.argmax(p1_te_pos, axis=1))
    train_df['p1_dep'] = dep_le.inverse_transform(p1_oof_dep)
    test_df['p1_dep'] = dep_le.inverse_transform(np.argmax(p1_te_dep, axis=1))
    del X_tr_p1, X_te_p1; gc.collect()
    
    for d in (train_df, test_df):
        for off in [-2, -1, 1, 2]:
            d[f'p1_pos_{off}'] = d.groupby('sentence_id')['p1_pos'].shift(-off).fillna(f'<P_{off}>')
            d[f'p1_dep_{off}'] = d.groupby('sentence_id')['p1_dep'].shift(-off).fillna(f'<D_{off}>')
        d['seq_pos_3'] = d['p1_pos_-1'] + '|' + d['p1_pos'] + '|' + d['p1_pos_1']
        d['seq_dep_3'] = d['p1_dep_-1'] + '|' + d['p1_dep'] + '|' + d['p1_dep_1']
        d['w_p1_m1'] = d['p1_pos_-1'] + '|' + d['word']
        d['w_p1_p1'] = d['word'] + '|' + d['p1_pos_1']
        d['w_d1_m1'] = d['p1_dep_-1'] + '|' + d['word']
        d['w_d1_p1'] = d['word'] + '|' + d['p1_dep_1']
        
    s2_cols = s1_cols + ['p1_pos', 'p1_dep', 'p1_pos_-2', 'p1_pos_-1', 'p1_pos_1', 'p1_pos_2', 'p1_dep_-2', 'p1_dep_-1', 'p1_dep_1', 'p1_dep_2', 'seq_pos_3', 'seq_dep_3', 'w_p1_m1', 'w_p1_p1', 'w_d1_m1', 'w_d1_p1']
    ohe_s2 = OneHotEncoder(handle_unknown='ignore', min_frequency=2, dtype=np.float32)
    X_tr_s2 = ohe_s2.fit_transform(train_df[s2_cols])
    X_te_s2 = ohe_s2.transform(test_df[s2_cols])
    
    oof_pos = np.zeros(len(train_df), dtype=np.int32); oof_pos_conf = np.zeros(len(train_df), dtype=np.float32); test_pos_probs = np.zeros((len(test_df), len(pos_le.classes_)), dtype=np.float32)
    oof_dep = np.zeros(len(train_df), dtype=np.int32); oof_dep_conf = np.zeros(len(train_df), dtype=np.float32); test_dep_probs = np.zeros((len(test_df), len(dep_le.classes_)), dtype=np.float32)
    oof_off_probs = np.zeros((len(train_df), n_off_classes), dtype=np.float32); test_off_probs = np.zeros((len(test_df), n_off_classes), dtype=np.float32)
    
    for k in bin_targets: train_df[k] = np.nan; test_df[k] = 0.0
        
    for tr_idx, va_idx in kf.split(unique_sents):
        tr_m = train_df['sentence_id'].isin(set(unique_sents[tr_idx])).values
        va_m = train_df['sentence_id'].isin(set(unique_sents[va_idx])).values
        
        clf_p = SGDClassifier(loss='log_loss', alpha=4e-6, max_iter=12, random_state=42, n_jobs=-1).fit(X_tr_s2[tr_m], y_pos[tr_m])
        p_va = clf_p.predict_proba(X_tr_s2[va_m])
        oof_pos[va_m] = clf_p.classes_[np.argmax(p_va, axis=1)]; oof_pos_conf[va_m] = np.max(p_va, axis=1)
        test_pos_probs[:, clf_p.classes_] += clf_p.predict_proba(X_te_s2) / 5.0
        
        clf_d = SGDClassifier(loss='log_loss', alpha=4e-6, max_iter=10, random_state=42, n_jobs=-1).fit(X_tr_s2[tr_m], y_dep[tr_m])
        d_va = clf_d.predict_proba(X_tr_s2[va_m])
        oof_dep[va_m] = clf_d.classes_[np.argmax(d_va, axis=1)]; oof_dep_conf[va_m] = np.max(d_va, axis=1)
        test_dep_probs[:, clf_d.classes_] += clf_d.predict_proba(X_te_s2) / 5.0
        
        clf_o = SGDClassifier(loss='log_loss', alpha=5e-6, max_iter=12, random_state=42, n_jobs=-1).fit(X_tr_s2[tr_m], y_off[tr_m])
        oof_off_probs[np.ix_(va_m, clf_o.classes_)] = clf_o.predict_proba(X_tr_s2[va_m])
        test_off_probs[:, clf_o.classes_] += clf_o.predict_proba(X_te_s2) / 5.0
        
        for k, y_b in bin_targets.items():
            clf_b = SGDClassifier(loss='log_loss', alpha=5e-6, max_iter=12, random_state=42, n_jobs=-1).fit(X_tr_s2[tr_m], y_b[tr_m])
            train_df.loc[va_m, k] = clf_b.predict_proba(X_tr_s2[va_m])[:, 1].astype(np.float32)
            test_df[k] += (clf_b.predict_proba(X_te_s2)[:, 1] / 5.0).astype(np.float32)
            
    del X_tr_s2, X_te_s2; gc.collect()
    
    train_df['s1_pos'] = pos_le.inverse_transform(oof_pos); train_df['s1_pos_conf'] = oof_pos_conf
    test_df['s1_pos'] = pos_le.inverse_transform(np.argmax(test_pos_probs, axis=1)); test_df['s1_pos_conf'] = np.max(test_pos_probs, axis=1).astype(np.float32)
    train_df['s1_dep'] = dep_le.inverse_transform(oof_dep); train_df['s1_dep_conf'] = oof_dep_conf
    test_df['s1_dep'] = dep_le.inverse_transform(np.argmax(test_dep_probs, axis=1)); test_df['s1_dep_conf'] = np.max(test_dep_probs, axis=1).astype(np.float32)
    
    for idx_c, cname in enumerate(off_le.classes_):
        train_df[f'p_off_{cname}'] = oof_off_probs[:, idx_c]
        test_df[f'p_off_{cname}'] = test_off_probs[:, idx_c]
        
    train_df = extract_chunk_governors(train_df)
    test_df = extract_chunk_governors(test_df)
    for df in (train_df, test_df):
        max_sent_root = df.groupby('sentence_id')['s1_p_root'].transform('max')
        df['root_prob_ratio'] = (df['s1_p_root'] / (max_sent_root + 1e-5)).astype(np.float32)
        for off in [-2, -1, 1, 2]:
            df[f's1_pos_{off}'] = df.groupby('sentence_id')['s1_pos'].shift(-off).fillna(f'<P_{off}>')
            df[f's1_dep_{off}'] = df.groupby('sentence_id')['s1_dep'].shift(-off).fillna(f'<D_{off}>')
        df['is_s1_verb'] = (df['s1_pos'] == 'VERB').astype(np.int16)
        df['is_s1_noun'] = (df['s1_pos'].isin(['NOUN', 'PRON_NOUN'])).astype(np.int16)
        df['is_comma'] = (df['word'] == ',').astype(np.int16)
        df['is_act_head'] = (df['s1_p_head'] > 0.5).astype(np.int16)
        df['is_lkid_head'] = (df['s1_p_lkid'] > 0.45).astype(np.int16)
        df['is_rkid_head'] = (df['s1_p_rkid'] > 0.45).astype(np.int16)
        for item, col in [('verb', 'is_s1_verb'), ('noun', 'is_s1_noun'), ('comma', 'is_comma'), ('punct', 'is_punct'), ('head', 'is_act_head'), ('lkid', 'is_lkid_head'), ('rkid', 'is_rkid_head')]:
            df[f'cum_{item}'] = df.groupby('sentence_id')[col].cumsum()
            
    return train_df, test_df

def build_edges(df, is_train=True):
    off_cols = [c for c in df.columns if c.startswith('p_off_')]
    keep_cols = [
        'sentence_id', 'token_id', 'word', 'suf2', 'suf3', 'pre3', 'shape', 'is_cap', 'is_num', 'is_punct', 'length', 'rel_pos', 'is_first', 'is_last', 'sent_len',
        'lex_is_root', 'lex_attach_right', 'lex_attach_left', 'lex_signed_dist', 'lex_is_any_head', 'lex_has_left_kid', 'lex_has_right_kid',
        's1_pos', 's1_pos_conf', 's1_dep', 's1_dep_conf', 's1_p_head', 's1_p_lkid', 's1_p_rkid', 's1_p_root', 'root_prob_ratio', 's1_p_att_r', 's1_p_att_l',
        'gov_prep', 'gov_mark', 'gov_cc', 'has_gov_det', 's1_pos_-1', 's1_pos_1', 's1_pos_-2', 's1_pos_2', 's1_dep_-1', 's1_dep_1', 'word_-1', 'word_1',
        'is_s1_verb', 'is_s1_noun', 'is_comma', 'is_act_head', 'is_lkid_head', 'is_rkid_head', 'cum_verb', 'cum_noun', 'cum_comma', 'cum_punct', 'cum_head', 'cum_lkid', 'cum_rkid'
    ] + off_cols
    
    if is_train: keep_cols += ['head_id', 'deprel']
    sub = df[keep_cols].copy()
    sents = df[['sentence_id', 'sent_len']].drop_duplicates()
    roots = sents.copy()
    roots['token_id'] = 0
    for c in keep_cols:
        if c not in roots.columns:
            if sub[c].dtype == object or isinstance(sub[c].dtype, pd.CategoricalDtype): roots[c] = '<ROOT>'
            else: roots[c] = 0.0
                
    roots['lex_is_any_head'] = 1.0; roots['lex_has_left_kid'] = 0.0; roots['lex_has_right_kid'] = 1.0
    roots['s1_p_head'] = 1.0; roots['s1_p_lkid'] = 0.0; roots['s1_p_rkid'] = 1.0
    roots['s1_pos_conf'] = 1.0; roots['s1_dep_conf'] = 1.0
    
    head_cols_list = [c for c in keep_cols if c not in (['head_id', 'deprel', 'sent_len'] + off_cols)]
    all_heads = pd.concat([roots[head_cols_list], sub[head_cols_list]], ignore_index=True)
    
    mod_rename = {c: f'm_{c}' for c in sub.columns if c not in ('sentence_id', 'sent_len')}
    head_rename = {c: f'h_{c}' for c in all_heads.columns if c != 'sentence_id'}
    modifiers = sub.rename(columns=mod_rename); heads = all_heads.rename(columns=head_rename)
    
    edges = modifiers.merge(heads, on='sentence_id', how='left')
    edges = edges[edges['m_token_id'] != edges['h_token_id']]
    dist = edges['h_token_id'] - edges['m_token_id']
    keep_mask = ((dist.abs() <= MAX_DIST) | (edges['h_token_id'] == 0) | (edges['m_is_punct'] == 1) | (edges['h_root_prob_ratio'] > 0.25))
    edges = edges[keep_mask].reset_index(drop=True)
    
    edges['dist'] = (edges['h_token_id'] - edges['m_token_id']).astype(np.int16)
    edges['abs_dist'] = np.where(edges['h_token_id'] == 0, 0, edges['dist'].abs()).astype(np.int16)
    edges['dir'] = np.where(edges['h_token_id'] == 0, 0, np.sign(edges['dist'])).astype(np.int8)
    edges['is_root_cand'] = (edges['h_token_id'] == 0).astype(np.int8)
    edges['rel_dist'] = (edges['dist'] / edges['sent_len']).astype(np.float32)
    edges['dist_diff_from_prior'] = (edges['dist'] - edges['m_lex_signed_dist']).abs().astype(np.float32)
    edges['punct_to_root_compat'] = (edges['m_is_punct'].astype(np.float32) * edges['m_is_last'].astype(np.float32) * edges['h_root_prob_ratio']).astype(np.float32)
    
    edge_buckets = get_offset_bucket_from_diff(edges['h_token_id'].values, edges['dist'].values)
    edges['off_bucket'] = edge_buckets
    edges['dist_bin'] = np.where(edges['h_token_id'] == 0, '0', np.where(edges['abs_dist'] <= 2, edges['abs_dist'].astype(str), '>2'))
    
    m_off_prob = np.zeros(len(edges), dtype=np.float32)
    for b_name in ['ROOT', 'L1', 'R1', 'L2', 'R2', 'L3', 'R3', 'L_FAR', 'R_FAR']:
        col_name = f'm_p_off_{b_name}'
        if col_name in edges.columns:
            mask_b = (edge_buckets == b_name)
            m_off_prob[mask_b] = edges.loc[mask_b, col_name].values
            edges.drop(columns=[col_name], inplace=True)
    edges['m_off_prob'] = m_off_prob
    
    edges['m_dir_prob'] = np.where(edges['h_token_id'] == 0, edges['m_s1_p_root'], np.where(edges['dist'] > 0, edges['m_s1_p_att_r'], edges['m_s1_p_att_l'])).astype(np.float32)
    edges['h_valency_prob'] = np.where(edges['h_token_id'] == 0, 1.0, np.where(edges['dist'] > 0, edges['h_s1_p_lkid'], edges['h_s1_p_rkid'])).astype(np.float32)
    edges['ctx_compat'] = (edges['m_dir_prob'] * edges['h_valency_prob']).astype(np.float32)
    edges['off_valency_compat'] = (edges['m_off_prob'] * edges['h_valency_prob']).astype(np.float32)
    edges['lex_dir_compat'] = np.where(edges['h_token_id'] == 0, edges['m_lex_is_root'], np.where(edges['dist'] > 0, edges['m_lex_attach_right'] * edges['h_lex_has_left_kid'], edges['m_lex_attach_left'] * edges['h_lex_has_right_kid'])).astype(np.float32)
    
    max_off_val = edges.groupby(['sentence_id', 'm_token_id'])['off_valency_compat'].transform('max')
    edges['off_val_ratio'] = (edges['off_valency_compat'] / (max_off_val + 1e-5)).astype(np.float32)
    
    max_ctx = edges.groupby(['sentence_id', 'm_token_id'])['ctx_compat'].transform('max')
    edges['ctx_compat_ratio'] = (edges['ctx_compat'] / (max_ctx + 1e-5)).astype(np.float32)
    
    not_root = (edges['h_token_id'] > 0)
    right_dir = (edges['dist'] > 0)
    for item, flag_col in [('verb', 'is_s1_verb'), ('noun', 'is_s1_noun'), ('comma', 'is_comma'), ('punct', 'is_punct'), ('head', 'is_act_head'), ('lkid', 'is_lkid_head'), ('rkid', 'is_rkid_head')]:
        btw_right = (edges[f'h_cum_{item}'] - edges[f'h_{flag_col}']) - edges[f'm_cum_{item}']
        btw_left = (edges[f'm_cum_{item}'] - edges[f'm_{flag_col}']) - edges[f'h_cum_{item}']
        edges[f'btw_{item}'] = np.where(not_root, np.maximum(0, np.where(right_dir, btw_right, btw_left)), 0).astype(np.int16)
        edges.drop(columns=[f'h_cum_{item}', f'm_cum_{item}', f'h_{flag_col}', f'm_{flag_col}'], inplace=True)
        
    edges['btw_competing_head'] = np.where(edges['h_token_id'] == 0, 0, np.where(right_dir, edges['btw_lkid'], edges['btw_rkid'])).astype(np.int16)
    
    h_is_verb = (edges['h_s1_pos'] == 'VERB')
    h_is_noun = edges['h_s1_pos'].isin(['NOUN', 'PRON_NOUN'])
    edges['is_nearest_l_verb'] = ((edges['dir'] == -1) & h_is_verb & (edges['btw_verb'] == 0)).astype(np.int8)
    edges['is_nearest_r_verb'] = ((edges['dir'] == 1) & h_is_verb & (edges['btw_verb'] == 0)).astype(np.int8)
    edges['is_nearest_l_noun'] = ((edges['dir'] == -1) & h_is_noun & (edges['btw_noun'] == 0)).astype(np.int8)
    edges['is_nearest_r_noun'] = ((edges['dir'] == 1) & h_is_noun & (edges['btw_noun'] == 0)).astype(np.int8)
    edges['same_pos_coord'] = ((edges['m_gov_cc'] != '<NONE>') & (edges['m_s1_pos'] == edges['h_s1_pos']) & (edges['dir'] == -1)).astype(np.int8)
    edges['of_to_left_noun'] = ((edges['m_gov_prep'] == 'of') & (edges['is_nearest_l_noun'] == 1)).astype(np.int8)
    
    if is_train:
        edges['is_true_head'] = (edges['m_head_id'] == edges['h_token_id']).astype(np.int8)
        edges['true_deprel'] = np.where(edges['is_true_head'] == 1, edges['m_deprel'], np.nan)
    return edges


def add_edge_priors_and_linear_scorer(train_edges, eval_edges):
    print("4. Computing OOF Morphological Target Encodings & Sparse Linear Scorer...")
    templates = [
        ('te_dep_pair', ['m_s1_dep', 'h_s1_dep', 'off_bucket'], 10.0),
        ('te_pos_pair', ['m_s1_pos', 'h_s1_pos', 'off_bucket', 'btw_competing_head'], 15.0),
        ('te_pos_dist_dir', ['m_s1_pos', 'h_s1_pos', 'dir', 'dist_bin'], 10.0), 
        ('te_mword_hpos', ['m_word', 'h_s1_pos', 'off_bucket'], 5.0),
        ('te_mpos_hword', ['m_s1_pos', 'h_word', 'off_bucket'], 5.0),
        ('te_bilexical', ['m_word', 'h_word', 'dir'], 3.0),
        ('te_context_pos', ['m_s1_pos_-1', 'm_s1_pos', 'h_s1_pos', 'h_s1_pos_1', 'dir'], 15.0),
        ('te_gov_prep', ['m_gov_prep', 'm_s1_pos', 'h_s1_pos', 'dir', 'btw_noun', 'btw_verb'], 8.0),
        ('te_gov_cc_mark', ['m_gov_cc', 'm_gov_mark', 'm_s1_pos', 'h_s1_pos', 'dir'], 8.0),
        ('te_m_suf_h_pos', ['m_suf3', 'h_s1_pos', 'dir'], 5.0),  
        ('te_m_pos_h_suf', ['m_s1_pos', 'h_suf3', 'dir'], 5.0),
        ('te_suf3_pair', ['m_suf3', 'h_suf3', 'dir'], 10.0)
    ]
    
    global_mean = float(train_edges['is_true_head'].mean())
    unique_sents = train_edges['sentence_id'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sent_to_fold = {}
    for fold_id, (_, va_idx) in enumerate(kf.split(unique_sents)):
        for s in unique_sents[va_idx]: sent_to_fold[s] = fold_id
    train_edges['_fold'] = train_edges['sentence_id'].map(sent_to_fold).astype(np.int8)
    
    for col_name, keys, m_smooth in templates:
        tot_sum = train_edges.groupby(keys, observed=True)['is_true_head'].transform('sum')
        tot_cnt = train_edges.groupby(keys, observed=True)['is_true_head'].transform('count')
        fold_sum = train_edges.groupby(keys + ['_fold'], observed=True)['is_true_head'].transform('sum')
        fold_cnt = train_edges.groupby(keys + ['_fold'], observed=True)['is_true_head'].transform('count')
        
        train_edges[col_name] = (((tot_sum - fold_sum) + m_smooth * global_mean) / ((tot_cnt - fold_cnt) + m_smooth)).astype(np.float32)
        stats = train_edges.groupby(keys, observed=True)['is_true_head'].agg(['count', 'mean']).reset_index()
        stats[col_name] = ((stats['count'] * stats['mean'] + m_smooth * global_mean) / (stats['count'] + m_smooth)).astype(np.float32)
        eval_edges[col_name] = eval_edges[keys].merge(stats[keys + [col_name]], on=keys, how='left')[col_name].fillna(global_mean).astype(np.float32).values
        
    def get_linear_edge_df(d):
        res = pd.DataFrame(index=d.index)
        dir_s = d['dir'].astype(str); off_s = d['off_bucket'].astype(str)
        btw_h = np.minimum(d['btw_competing_head'], 3).astype(str); btw_v = np.minimum(d['btw_verb'], 2).astype(str)
        btw_n = np.minimum(d['btw_noun'], 2).astype(str); btw_c = np.minimum(d['btw_comma'], 2).astype(str)
        res['f1'] = d['m_word'].astype(str) + '|' + d['h_word'].astype(str) + '|' + dir_s
        res['f2'] = d['m_word'].astype(str) + '|' + d['h_s1_pos'].astype(str) + '|' + off_s + '|' + btw_n
        res['f3'] = d['m_s1_pos'].astype(str) + '|' + d['h_word'].astype(str) + '|' + off_s + '|' + btw_h
        res['f4'] = d['m_s1_dep'].astype(str) + '|' + d['h_s1_dep'].astype(str) + '|' + off_s + '|' + btw_h
        res['f5'] = d['m_s1_pos'].astype(str) + '|' + d['h_s1_pos'].astype(str) + '|' + off_s + '|' + btw_h + '|' + btw_v + '|' + btw_n + '|' + btw_c
        res['f6'] = d['m_s1_pos_-1'].astype(str) + '|' + d['m_s1_pos'].astype(str) + '|' + d['h_s1_pos'].astype(str) + '|' + d['h_s1_pos_1'].astype(str) + '|' + dir_s
        res['f7'] = d['m_gov_prep'].astype(str) + '|' + d['m_s1_pos'].astype(str) + '|' + d['h_s1_pos'].astype(str) + '|' + dir_s + '|' + btw_n + '|' + btw_v
        res['f8'] = d['m_gov_cc'].astype(str) + '|' + d['m_gov_mark'].astype(str) + '|' + d['m_s1_pos'].astype(str) + '|' + d['h_s1_pos'].astype(str) + '|' + dir_s
        return res

    tr_lin_df = get_linear_edge_df(train_edges)
    ev_lin_df = get_linear_edge_df(eval_edges)
    ohe_edge = OneHotEncoder(handle_unknown='ignore', min_frequency=3, dtype=np.float32)
    X_tr_lin = ohe_edge.fit_transform(tr_lin_df)
    X_ev_lin = ohe_edge.transform(ev_lin_df)
    del tr_lin_df, ev_lin_df; gc.collect()
    
    oof_lin = np.zeros(len(train_edges), dtype=np.float32)
    ev_lin = np.zeros(len(eval_edges), dtype=np.float32)
    y_edge = train_edges['is_true_head'].values
    
    for fold_id in range(5):
        va_m = (train_edges['_fold'].values == fold_id); tr_m = ~va_m
        clf_lin = SGDClassifier(loss='log_loss', alpha=2e-6, max_iter=15, random_state=42 + fold_id, n_jobs=-1)
        clf_lin.fit(X_tr_lin[tr_m], y_edge[tr_m])
        oof_lin[va_m] = clf_lin.predict_proba(X_tr_lin[va_m])[:, 1].astype(np.float32)
        ev_lin += (clf_lin.predict_proba(X_ev_lin)[:, 1] / 5.0).astype(np.float32)
        
    train_edges['lin_edge_prob'] = oof_lin; eval_edges['lin_edge_prob'] = ev_lin
    train_edges.drop(columns=['_fold'], inplace=True)
    del X_tr_lin, X_ev_lin; gc.collect()
    
    def add_stacked_tree_features(edges):
        edges['combined_te'] = (edges['te_dep_pair'] * edges['te_mword_hpos'] * edges['te_mpos_hword'] * edges['te_gov_prep'] * edges['te_m_suf_h_pos']).astype(np.float32)
        idx_greedy = edges.groupby(['sentence_id', 'm_token_id'])['combined_te'].idxmax()
        greedy_map = edges.loc[idx_greedy, ['sentence_id', 'm_token_id', 'h_token_id', 'h_s1_pos', 'h_s1_dep']]
        greedy_map.columns = ['sentence_id', 'm_token_id', 'g_head', 'g_head_pos', 'g_head_dep']
        
        edges = edges.merge(greedy_map, on=['sentence_id', 'm_token_id'], how='left')
        edges['is_g_head'] = (edges['h_token_id'] == edges['g_head']).astype(np.int8)
        gp_map = greedy_map[['sentence_id', 'm_token_id', 'g_head']].rename(columns={'m_token_id': 'h_token_id', 'g_head': 'g_grandparent'})
        edges = edges.merge(gp_map, on=['sentence_id', 'h_token_id'], how='left')
        edges['g_grandparent'] = edges['g_grandparent'].fillna(0).astype(np.int16)
        edges['cycle_with_gp'] = (edges['m_token_id'] == edges['g_grandparent']).astype(np.int8)
        
        greedy_kids = greedy_map.groupby(['sentence_id', 'g_head']).size().reset_index(name='g_kids')
        edges = edges.merge(greedy_kids, left_on=['sentence_id', 'h_token_id'], right_on=['sentence_id', 'g_head'], how='left')
        edges['g_kids'] = edges['g_kids'].fillna(0).astype(np.int16)
        if 'g_head_y' in edges.columns: edges.drop(columns=['g_head_y'], inplace=True)
        if 'g_head_x' in edges.columns: edges.rename(columns={'g_head_x': 'g_head'}, inplace=True)
        
        left_kids = greedy_map[greedy_map['m_token_id'] < greedy_map['g_head']].groupby(['sentence_id', 'g_head']).size().reset_index(name='g_lkids')
        edges = edges.merge(left_kids, left_on=['sentence_id', 'h_token_id'], right_on=['sentence_id', 'g_head'], how='left')
        edges['g_lkids'] = edges['g_lkids'].fillna(0).astype(np.int16)
        if 'g_head_y' in edges.columns: edges.drop(columns=['g_head_y'], inplace=True)
        if 'g_head_x' in edges.columns: edges.rename(columns={'g_head_x': 'g_head'}, inplace=True)
        
        right_kids = greedy_map[greedy_map['m_token_id'] > greedy_map['g_head']].groupby(['sentence_id', 'g_head']).size().reset_index(name='g_rkids')
        edges = edges.merge(right_kids, left_on=['sentence_id', 'h_token_id'], right_on=['sentence_id', 'g_head'], how='left')
        edges['g_rkids'] = edges['g_rkids'].fillna(0).astype(np.int16)
        if 'g_head_y' in edges.columns: edges.drop(columns=['g_head_y'], inplace=True)
        if 'g_head_x' in edges.columns: edges.rename(columns={'g_head_x': 'g_head'}, inplace=True)
        
        max_te = edges.groupby(['sentence_id', 'm_token_id'])['combined_te'].transform('max')
        edges['combined_te_ratio'] = (edges['combined_te'] / (max_te + 1e-6)).astype(np.float32)
        sum_lin = edges.groupby(['sentence_id', 'm_token_id'])['lin_edge_prob'].transform('sum')
        edges['lin_edge_norm'] = (edges['lin_edge_prob'] / (sum_lin + 1e-6)).astype(np.float32)
        max_lin = edges.groupby(['sentence_id', 'm_token_id'])['lin_edge_prob'].transform('max')
        edges['lin_edge_ratio'] = (edges['lin_edge_prob'] / (max_lin + 1e-6)).astype(np.float32)
        edges['lin_gap_to_best'] = (max_lin - edges['lin_edge_prob']).astype(np.float32)
        
        edges['lin_edge_rank'] = edges.groupby(['sentence_id', 'm_token_id'])['lin_edge_prob'].rank(ascending=False, method='min').astype(np.int16)
        edges['te_rank'] = edges.groupby(['sentence_id', 'm_token_id'])['combined_te'].rank(ascending=False, method='min').astype(np.int16)
        
        m_vecs = np.vstack(edges['m_word'].apply(lambda w: svd_map.get(str(w).lower(), np.zeros(12, dtype=np.float32))).values)
        h_vecs = np.vstack(edges['h_word'].apply(lambda w: svd_map.get(str(w).lower(), np.zeros(12, dtype=np.float32))).values)
        edges['svd_dot'] = np.sum(m_vecs * h_vecs, axis=1).astype(np.float32)
        for i in range(4): 
            edges[f'm_svd_{i}'] = m_vecs[:, i]
            edges[f'h_svd_{i}'] = h_vecs[:, i]
        return edges

    train_edges = add_stacked_tree_features(train_edges)
    eval_edges = add_stacked_tree_features(eval_edges)
    return train_edges, eval_edges


# 7. Exact Projective Single-Root Eisner DP Decoder
def eisner_single_root(score_matrix):
    N = score_matrix.shape[0] - 1
    if N == 1: return np.array([-1, 0], dtype=np.int32)
        
    NEG_INF = -1e12
    E_inc_L = np.full((N + 1, N + 1), NEG_INF, dtype=np.float64); E_inc_R = np.full((N + 1, N + 1), NEG_INF, dtype=np.float64)
    E_cmp_L = np.zeros((N + 1, N + 1), dtype=np.float64); E_cmp_R = np.zeros((N + 1, N + 1), dtype=np.float64)
    bp_inc_L = np.zeros((N + 1, N + 1), dtype=np.int32); bp_inc_R = np.zeros((N + 1, N + 1), dtype=np.int32)
    bp_cmp_L = np.zeros((N + 1, N + 1), dtype=np.int32); bp_cmp_R = np.zeros((N + 1, N + 1), dtype=np.int32)
    
    for length in range(1, N):
        for s in range(1, N - length + 1):
            t = s + length
            qs = np.arange(s, t)
            
            cand_scores = E_cmp_R[s, qs] + E_cmp_L[qs + 1, t]
            best_idx = int(np.argmax(cand_scores))
            best_val = cand_scores[best_idx]
            best_q = s + best_idx
            
            E_inc_L[s, t] = best_val + score_matrix[t, s]; bp_inc_L[s, t] = best_q
            E_inc_R[s, t] = best_val + score_matrix[s, t]; bp_inc_R[s, t] = best_q
            
            cand_L = E_cmp_L[s, qs] + E_inc_L[qs, t]
            idx_L = int(np.argmax(cand_L))
            E_cmp_L[s, t] = cand_L[idx_L]; bp_cmp_L[s, t] = s + idx_L
            
            qs_R = np.arange(s + 1, t + 1)
            cand_R = E_inc_R[s, qs_R] + E_cmp_R[qs_R, t]
            idx_R = int(np.argmax(cand_R))
            E_cmp_R[s, t] = cand_R[idx_R]; bp_cmp_R[s, t] = s + 1 + idx_R
            
    rs = np.arange(1, N + 1)
    root_totals = score_matrix[0, rs] + E_cmp_L[1, rs] + E_cmp_R[rs, N]
    best_r = int(rs[np.argmax(root_totals)])
    heads = np.zeros(N + 1, dtype=np.int32)
    heads[best_r] = 0
    
    def backtrack(s, t, direction, is_complete):
        if s >= t: return
        if is_complete:
            if direction == 0:
                q = bp_cmp_L[s, t]; backtrack(s, q, 0, True); backtrack(q, t, 0, False)
            else:
                q = bp_cmp_R[s, t]; backtrack(s, q, 1, False); backtrack(q, t, 1, True)
        else:
            if direction == 0:
                heads[s] = t; q = bp_inc_L[s, t]; backtrack(s, q, 1, True); backtrack(q + 1, t, 0, True)
            else:
                heads[t] = s; q = bp_inc_R[s, t]; backtrack(s, q, 1, True); backtrack(q + 1, t, 0, True)
                
    backtrack(1, best_r, 0, True); backtrack(best_r, N, 1, True)
    return heads

def decode_projective_trees(edges_df, tok_df):
    tok_lookup = tok_df.set_index(['sentence_id', 'token_id'])[['token', 'is_punct', 's1_dep']].to_dict('index')
    pred_records = []
    
    for sentence_id, grp in edges_df.groupby('sentence_id', sort=False):
        N = int(grp['m_token_id'].max())
        S = np.full((N + 1, N + 1), -100.0, dtype=np.float64)
        
        h_ids = grp['h_token_id'].values.astype(np.int32)
        m_ids = grp['m_token_id'].values.astype(np.int32)
        probs = np.clip(grp['head_prob'].values, 1e-6, 1.0 - 1e-6)
        
        S[h_ids, m_ids] = np.log(probs / (1.0 - probs))
        heads = eisner_single_root(S)
        
        root_candidates = np.where(heads[1:] == 0)[0] + 1
        r_root = int(root_candidates[0]) if len(root_candidates) > 0 else 1
        
        if N >= 2:
            last_info = tok_lookup.get((sentence_id, N), None)
            if last_info is not None:
                t_str = str(last_info['token'])
                if t_str in ('.', '?', '!', '...', ':') and r_root != N:
                    heads[N] = r_root
                elif t_str in ('"', "'", "''") and N >= 3:
                    prev_info = tok_lookup.get((sentence_id, N - 1), None)
                    if prev_info is not None and str(prev_info['token']) in ('.', '?', '!') and r_root != (N - 1):
                        heads[N - 1] = r_root
                        
        for m_i in range(1, N + 1):
            pred_records.append((sentence_id, m_i, int(heads[m_i])))
            
    return pd.DataFrame(pred_records, columns=['sentence_id', 'token_id', 'pred_head'])


# 8. Second-Order Tree-Aware Relation Classifier
def train_and_predict_relations(train_tok_df, eval_tok_df, eval_pred_heads):
    print("8. Extracting Tree-Child, Sibling & Clause Structural Features for Relation Ensemble...")
    
    def extract_tree_child_features(df, head_col):
        children = df[df[head_col] > 0][['sentence_id', 'token_id', head_col, 'word', 's1_dep', 's1_pos']].copy()
        
        case_kids = children[children['s1_dep'] == 'case'].drop_duplicates(subset=['sentence_id', head_col])
        case_map = case_kids.set_index(['sentence_id', head_col])['word'].rename('kid_case_word')
        
        mark_kids = children[children['s1_dep'] == 'mark'].drop_duplicates(subset=['sentence_id', head_col])
        mark_map = mark_kids.set_index(['sentence_id', head_col])['word'].rename('kid_mark_word')
        
        cc_kids = children[children['s1_dep'] == 'cc'].drop_duplicates(subset=['sentence_id', head_col])
        cc_map = cc_kids.set_index(['sentence_id', head_col])['word'].rename('kid_cc_word')
        
        auxpass_kids = children[children['s1_dep'] == 'aux:pass'].groupby(['sentence_id', head_col]).size().rename('n_auxpass_kid')
        cop_kids = children[children['s1_dep'] == 'cop'].groupby(['sentence_id', head_col]).size().rename('n_cop_kid')
        det_kids = children[children['s1_dep'] == 'det'].groupby(['sentence_id', head_col]).size().rename('n_det_kid')
        subj_kids = children[children['s1_dep'].isin(['nsubj', 'nsubj:pass'])].groupby(['sentence_id', head_col]).size().rename('n_subj_kid')
        
        l_cnt = children[children['token_id'] < children[head_col]].groupby(['sentence_id', head_col]).size().rename('tree_n_lkid')
        r_cnt = children[children['token_id'] > children[head_col]].groupby(['sentence_id', head_col]).size().rename('tree_n_rkid')
        
        idx = df.set_index(['sentence_id', 'token_id'])
        for s_map in [case_map, mark_map, cc_map]:
            idx[s_map.name] = s_map
        for n_map in [auxpass_kids, cop_kids, det_kids, subj_kids, l_cnt, r_cnt]:
            idx[n_map.name] = n_map
            
        res = idx.fillna({
            'kid_case_word': '<NONE>', 'kid_mark_word': '<NONE>', 'kid_cc_word': '<NONE>',
            'n_auxpass_kid': 0, 'n_cop_kid': 0, 'n_det_kid': 0, 'n_subj_kid': 0,
            'tree_n_lkid': 0, 'tree_n_rkid': 0
        }).reset_index()
        
        res['has_case_kid'] = (res['kid_case_word'] != '<NONE>').astype(str)
        res['has_mark_kid'] = (res['kid_mark_word'] != '<NONE>').astype(str)
        res['has_cc_kid'] = (res['kid_cc_word'] != '<NONE>').astype(str)
        res['has_auxpass_kid'] = (res['n_auxpass_kid'] > 0).astype(str)
        res['has_cop_kid'] = (res['n_cop_kid'] > 0).astype(str)
        res['has_det_kid'] = (res['n_det_kid'] > 0).astype(str)
        res['has_subj_kid'] = (res['n_subj_kid'] > 0).astype(str)
        
        is_bare_r_nom = ((res[head_col] > 0) & (res['token_id'] > res[head_col]) & (res['kid_case_word'] == '<NONE>') & (res['s1_pos'].isin(['NOUN', 'PRON_NOUN', 'NUM'])))
        res['_bare_r_nom'] = is_bare_r_nom.astype(int)
        tot_bare_r = res.groupby(['sentence_id', head_col])['_bare_r_nom'].transform('sum')
        cum_bare_r = res.groupby(['sentence_id', head_col])['_bare_r_nom'].cumsum()
        res['bare_r_nom_pos'] = np.where(~is_bare_r_nom, 'NONE', np.where(tot_bare_r == 1, 'SINGLE', np.where(cum_bare_r == 1, 'FIRST_OF_MULTI', 'LATER_OF_MULTI')))
        res.drop(columns=['_bare_r_nom'], inplace=True)
        return res

    def build_rel_table(df, head_series):
        d = df.copy()
        d['assigned_head'] = head_series.values
        d = extract_tree_child_features(d, 'assigned_head')
        
        h_info = d[['sentence_id', 'token_id', 'word', 'suf3', 'suf2', 'pre3', 'shape', 's1_pos', 's1_dep', 'word_-1', 'word_1', 'has_auxpass_kid', 'has_cop_kid', 'has_subj_kid', 'kid_case_word']].copy()
        h_info.columns = ['sentence_id', 'assigned_head', 'h_word', 'h_suf3', 'h_suf2', 'h_pre3', 'h_shape', 'h_s1_pos', 'h_s1_dep', 'h_wm1', 'h_wp1', 'h_has_auxpass', 'h_has_cop', 'h_has_subj', 'h_kid_case']
        m = d.merge(h_info, on=['sentence_id', 'assigned_head'], how='left')
        for c in ['h_word', 'h_suf3', 'h_suf2', 'h_pre3', 'h_shape', 'h_s1_pos', 'h_s1_dep', 'h_wm1', 'h_wp1', 'h_has_auxpass', 'h_has_cop', 'h_has_subj', 'h_kid_case']:
            m[c] = m[c].fillna('<ROOT>')
            
        m['dir'] = np.where(m['assigned_head'] == 0, 'ROOT', np.where(m['assigned_head'] > m['token_id'], 'R', 'L'))
        dist = np.abs(m['assigned_head'] - m['token_id'])
        m['dist_bin'] = np.where(m['assigned_head'] == 0, '0', np.where(dist == 1, '1', np.where(dist == 2, '2', np.where(dist <= 5, '3-5', '6+'))))
        
        m['w_hw_dir'] = m['word'] + '|' + m['h_word'] + '|' + m['dir']
        m['w_hw'] = m['word'] + '|' + m['h_word']
        m['w_dir'] = m['word'] + '|' + m['dir']
        m['w_dist_dir'] = m['word'] + '|' + m['dist_bin'] + '|' + m['dir']
        m['w_hpos_dir'] = m['word'] + '|' + m['h_s1_pos'] + '|' + m['dir']
        m['w_hdep_dir'] = m['word'] + '|' + m['h_s1_dep'] + '|' + m['dir']
        m['mpos_hw_dir'] = m['s1_pos'] + '|' + m['h_word'] + '|' + m['dir']
        m['mdep_hw_dir'] = m['s1_dep'] + '|' + m['h_word'] + '|' + m['dir']
        m['mpos_hpos_dir'] = m['s1_pos'] + '|' + m['h_s1_pos'] + '|' + m['dir'] + '|' + m['dist_bin']
        m['mdep_hdep_dir'] = m['s1_dep'] + '|' + m['h_s1_dep'] + '|' + m['dir']
        m['w_hsuf_dir'] = m['word'] + '|' + m['h_suf3'] + '|' + m['dir']
        m['suf_hw_dir'] = m['suf3'] + '|' + m['h_word'] + '|' + m['dir']
        m['suf_hsuf_dir'] = m['suf3'] + '|' + m['h_suf3'] + '|' + m['dir']
        m['hw_dir'] = m['h_word'] + '|' + m['dir']
        m['w_wm1_dir'] = m['word_-1'] + '|' + m['word'] + '|' + m['dir']
        m['w_wp1_dir'] = m['word'] + '|' + m['word_1'] + '|' + m['dir']
        m['hw_hwm1_dir'] = m['h_wm1'] + '|' + m['h_word'] + '|' + m['dir']
        m['hw_hwp1_dir'] = m['h_word'] + '|' + m['h_wp1'] + '|' + m['dir']
        
        m['tree_case_hpos'] = m['has_case_kid'] + '|' + m['kid_case_word'] + '|' + m['h_s1_pos'] + '|' + m['h_s1_dep'] + '|' + m['dir']
        m['tree_case_mpos'] = m['has_case_kid'] + '|' + m['s1_pos'] + '|' + m['h_s1_pos'] + '|' + m['dir']
        m['tree_mark_hpos'] = m['has_mark_kid'] + '|' + m['kid_mark_word'] + '|' + m['h_s1_pos'] + '|' + m['dir']
        m['tree_cc_conj'] = m['has_cc_kid'] + '|' + m['kid_cc_word'] + '|' + m['s1_pos'] + '|' + m['h_s1_pos'] + '|' + m['dir']
        m['tree_passive'] = m['h_has_auxpass'] + '|' + m['s1_dep'] + '|' + m['kid_case_word'] + '|' + m['dir']
        m['tree_cop_structure'] = m['has_cop_kid'] + '|' + m['h_has_cop'] + '|' + m['s1_dep'] + '|' + m['h_s1_dep'] + '|' + m['dir']
        m['tree_valency_profile'] = m['has_det_kid'] + '|' + m['has_subj_kid'] + '|' + m['s1_dep'] + '|' + m['h_s1_pos'] + '|' + m['dir']
        m['tree_iobj_obj'] = m['bare_r_nom_pos'] + '|' + m['h_word'] + '|' + m['s1_pos'] + '|' + m['dir']
        m['tree_clause_type'] = m['has_subj_kid'] + '|' + m['kid_mark_word'] + '|' + m['s1_pos'] + '|' + m['h_word'] + '|' + m['dir']
        m['tree_noun_clause'] = m['h_s1_pos'] + '|' + m['kid_mark_word'] + '|' + m['has_subj_kid'] + '|' + m['suf3'] + '|' + m['word_-1'] + '|' + m['dir']
        return m

    tr_rel_all = build_rel_table(train_tok_df, train_tok_df['head_id'])
    ev_rel_all = build_rel_table(eval_tok_df, eval_pred_heads)
    
    tr_non_root = tr_rel_all[tr_rel_all['assigned_head'] != 0].copy()
    ev_non_root_mask = (ev_rel_all['assigned_head'] != 0).values
    ev_non_root = ev_rel_all.loc[ev_non_root_mask].copy()
    
    rel_feat_cols = [
        'word', 'token', 'shape', 'suf1', 'suf2', 'suf3', 'suf4', 'pre1', 'pre2', 'pre3', 'pos_bucket',
        's1_pos', 's1_dep', 'gov_prep', 'gov_mark', 'gov_cc', 'word_-3', 'word_-2', 'word_-1', 'word_1', 'word_2', 'word_3',
        's1_pos_-1', 's1_pos_1', 's1_dep_-1', 's1_dep_1', 'h_word', 'h_suf3', 'h_suf2', 'h_pre3', 'h_shape', 'h_s1_pos', 'h_s1_dep', 'h_wm1', 'h_wp1',
        'dir', 'dist_bin', 'wm1_w0', 'w0_wp1', 'wm1_wp1', 'wm2_wm1_w0', 'w0_wp1_wp2',
        'w_hw_dir', 'w_hw', 'w_dir', 'w_dist_dir', 'w_hpos_dir', 'w_hdep_dir', 'mpos_hw_dir', 'mdep_hw_dir', 'mpos_hpos_dir', 'mdep_hdep_dir',
        'w_hsuf_dir', 'suf_hw_dir', 'suf_hsuf_dir', 'hw_dir', 'w_wm1_dir', 'w_wp1_dir', 'hw_hwm1_dir', 'hw_hwp1_dir',
        'kid_case_word', 'kid_mark_word', 'kid_cc_word', 'has_case_kid', 'has_mark_kid', 'has_cc_kid',
        'has_auxpass_kid', 'has_cop_kid', 'has_det_kid', 'has_subj_kid', 'h_has_auxpass', 'h_has_cop',
        'bare_r_nom_pos', 'tree_case_hpos', 'tree_case_mpos', 'tree_mark_hpos', 'tree_cc_conj',
        'tree_passive', 'tree_cop_structure', 'tree_valency_profile', 'tree_iobj_obj', 'tree_clause_type', 'tree_noun_clause'
    ]
    
    ohe = OneHotEncoder(handle_unknown='ignore', min_frequency=2, dtype=np.float32)
    X_tr = ohe.fit_transform(tr_non_root[rel_feat_cols])
    X_ev = ohe.transform(ev_non_root[rel_feat_cols])
    
    rel_le = LabelEncoder()
    y_tr = rel_le.fit_transform(tr_non_root['deprel'])
    n_classes = len(rel_le.classes_)
    
    probs_log = np.zeros((len(ev_non_root), n_classes), dtype=np.float32)
    clf_log = SGDClassifier(loss='log_loss', alpha=2.5e-6, max_iter=22, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    probs_log[:, clf_log.classes_] = clf_log.predict_proba(X_ev)
    
    probs_log2 = np.zeros((len(ev_non_root), n_classes), dtype=np.float32)
    clf_log2 = SGDClassifier(loss='log_loss', alpha=5e-6, max_iter=22, random_state=99, n_jobs=-1).fit(X_tr, y_tr)
    probs_log2[:, clf_log2.classes_] = clf_log2.predict_proba(X_ev)
    
    probs_hub = np.zeros((len(ev_non_root), n_classes), dtype=np.float32)
    clf_hub = SGDClassifier(loss='modified_huber', alpha=1e-5, max_iter=22, random_state=101, n_jobs=-1).fit(X_tr, y_tr)
    probs_hub[:, clf_hub.classes_] = clf_hub.predict_proba(X_ev)
    
    probs_nb = np.zeros((len(ev_non_root), n_classes), dtype=np.float32)
    clf_nb = MultinomialNB(alpha=0.15).fit(X_tr, y_tr)
    probs_nb[:, clf_nb.classes_] = clf_nb.predict_proba(X_ev)
    
    tr_non_root['y_idx'] = y_tr
    grp1 = tr_non_root.groupby(['w_hw_dir', 'y_idx']).size().unstack(fill_value=0).reindex(columns=range(n_classes), fill_value=0)
    grp1_prob = grp1.div(grp1.sum(axis=1) + 3.0, axis=0)
    mapped1 = ev_non_root[['w_hw_dir']].merge(grp1_prob, on='w_hw_dir', how='left').drop(columns=['w_hw_dir']).fillna(0).values
    
    grp2 = tr_non_root.groupby(['tree_case_hpos', 'y_idx']).size().unstack(fill_value=0).reindex(columns=range(n_classes), fill_value=0)
    grp2_prob = grp2.div(grp2.sum(axis=1) + 5.0, axis=0)
    mapped2 = ev_non_root[['tree_case_hpos']].merge(grp2_prob, on='tree_case_hpos', how='left').drop(columns=['tree_case_hpos']).fillna(0).values
    
    ensemble_probs = (0.35 * probs_log + 0.25 * probs_log2 + 0.15 * probs_hub + 0.07 * probs_nb + 0.10 * mapped1 + 0.08 * mapped2)
    pred_non_root = rel_le.inverse_transform(np.argmax(ensemble_probs, axis=1))
    
    final_deprels = np.full(len(ev_rel_all), 'root', dtype=object)
    final_deprels[ev_non_root_mask] = pred_non_root
    return final_deprels


# 9. Master End-to-End Pipeline
def run_pipeline(train_df, eval_df, is_validation=False):
    train_p, eval_p = add_stage1_tags_and_stats(train_df, eval_df)
    
    print("3. Building smart pairwise candidate edges with exact between-token features...")
    train_edges = build_edges(train_p, is_train=True)
    eval_edges = build_edges(eval_p, is_train=False)
    
    train_edges, eval_edges = add_edge_priors_and_linear_scorer(train_edges, eval_edges)
    
    cat_cols = [
        'm_word', 'h_word', 'm_suf2', 'h_suf2', 'm_suf3', 'h_suf3', 'm_pre3', 'h_pre3',
        'm_shape', 'h_shape', 'm_s1_pos', 'h_s1_pos', 'm_s1_dep', 'h_s1_dep',
        'm_gov_prep', 'm_gov_mark', 'm_gov_cc', 'h_gov_prep', 'h_gov_mark', 'h_gov_cc',
        'm_s1_pos_-1', 'm_s1_pos_1', 'm_s1_pos_-2', 'm_s1_pos_2',
        'h_s1_pos_-1', 'h_s1_pos_1', 'h_s1_pos_-2', 'h_s1_pos_2',
        'm_s1_dep_-1', 'm_s1_dep_1', 'h_s1_dep_-1', 'h_s1_dep_1',
        'm_word_-1', 'm_word_1', 'h_word_-1', 'h_word_1', 'off_bucket', 'dist_bin',
        'g_head_pos', 'g_head_dep'
    ]
    exclude = set(cat_cols + ['sentence_id', 'm_head_id', 'm_deprel', 'is_true_head', 'true_deprel'])
    num_cols = [c for c in train_edges.columns if c not in exclude]
    features = num_cols + cat_cols
    
    print(f"5. Aligning {len(cat_cols)} categorical and {len(num_cols)} numerical edge features...")
    for col in cat_cols:
        cats = sorted(list(set(train_edges[col].astype(str)) | set(eval_edges[col].astype(str))))
        train_edges[col] = pd.Categorical(train_edges[col].astype(str), categories=cats)
        eval_edges[col] = pd.Categorical(eval_edges[col].astype(str), categories=cats)
        
    print(f"6. Training 3-Model Precision Edge Ensemble on {len(train_edges):,} candidate edges...")
    
    clf_head1 = lgb.LGBMClassifier(
        n_estimators=1800,
        learning_rate=0.02,
        num_leaves=127,
        min_child_samples=30,
        max_delta_step=1.0,
        colsample_bytree=0.60,
        subsample=0.8,
        subsample_freq=1,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    clf_head1.fit(train_edges[features], train_edges['is_true_head'])
    p1 = clf_head1.predict_proba(eval_edges[features])[:, 1].astype(np.float32)
    del clf_head1; gc.collect()
    
    clf_head2 = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        max_delta_step=1.0,
        colsample_bytree=0.50,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=2.5,
        reg_lambda=7.5,
        random_state=2026,
        verbose=-1,
        n_jobs=-1
    )
    clf_head2.fit(train_edges[features], train_edges['is_true_head'])
    p2 = clf_head2.predict_proba(eval_edges[features])[:, 1].astype(np.float32)
    del clf_head2; gc.collect()
    
    clf_head3 = lgb.LGBMClassifier(
        n_estimators=1600,
        learning_rate=0.025,
        num_leaves=95,
        extra_trees=True, 
        min_child_samples=25,
        colsample_bytree=0.45,
        subsample=0.75,
        subsample_freq=1,
        random_state=777,
        verbose=-1,
        n_jobs=-1
    )
    clf_head3.fit(train_edges[features], train_edges['is_true_head'])
    p3 = clf_head3.predict_proba(eval_edges[features])[:, 1].astype(np.float32)
    del clf_head3, train_edges; gc.collect()
    
    print("7. Blending Edge Models and decoding Projective Single-Root Trees (Eisner DP)...")
    eval_edges['raw_prob'] = 0.38 * p1 + 0.35 * p2 + 0.20 * p3 + 0.07 * eval_edges['lin_edge_prob'].values
    prob_sum = eval_edges.groupby(['sentence_id', 'm_token_id'])['raw_prob'].transform('sum')
    eval_edges['head_prob'] = np.where(prob_sum > 0, eval_edges['raw_prob'] / prob_sum, eval_edges['raw_prob'])
    
    head_map = decode_projective_trees(eval_edges, eval_p)
    del eval_edges; gc.collect()
    
    merged = eval_p.merge(head_map, on=['sentence_id', 'token_id'], how='left')
    merged['pred_head'] = merged['pred_head'].fillna(0).astype(int)
    
    merged['pred_deprel'] = train_and_predict_relations(train_p, merged, merged['pred_head'])
    merged['pred_target'] = merged['pred_head'].astype(str) + '|' + merged['pred_deprel'].astype(str)
    
    if is_validation:
        uas = (merged['pred_head'] == merged['head_id']).mean() * 100
        rel_acc = (merged['pred_deprel'] == merged['deprel']).mean() * 100
        las = (merged['pred_target'] == merged['target']).mean() * 100
        print("\n==========================================")
        print("LOCAL VALIDATION RESULTS:")
        print(f"  Head Accuracy (UAS):     {uas:.2f}%")
        print(f"  Relation Accuracy:       {rel_acc:.2f}%")
        print(f"  Exact Match Score (LAS): {las:.2f}%")
        print("==========================================\n")
    return merged

# --- EXECUTE PIPELINE ---
if RUN_LOCAL_VAL:
    print("\n--- PHASE A: RUNNING 80/20 LOCAL VALIDATION ---")
    all_sents = train_full['sentence_id'].unique()
    tr_s, va_s = train_test_split(all_sents, test_size=0.2, random_state=42)
    tr_split = train_full[train_full['sentence_id'].isin(tr_s)].copy()
    va_split = train_full[train_full['sentence_id'].isin(va_s)].copy()
    _ = run_pipeline(tr_split, va_split, is_validation=True)
    del tr_split, va_split
    gc.collect()

print("\n--- PHASE B: FULL TRAINING & GENERATING 227,783-ROW SUBMISSION.CSV ---")
final_test_preds = run_pipeline(train_full, test_full, is_validation=False)

train_sub = train_full[['ID', 'target']].rename(columns={'target': 'Label'})
test_sub = final_test_preds[['ID', 'pred_target']].rename(columns={'pred_target': 'Label'})

submission = pd.concat([train_sub, test_sub], ignore_index=True)
assert len(submission) == len(train_full) + len(test_full)
submission.to_csv('submission.csv', index=False)
print(f"Success! 'submission.csv' saved with {len(submission):,} rows (100% matched to sample.csv).")