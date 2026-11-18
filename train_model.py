"""
train_model.py  -  Neural Network-Based Hand Gesture Recognition for Human Verification
Sumair Hejib | 20241CAI0092 | Neural Networks And Fuzzy Logic (CAI 2512)

Run once:   python train_model.py
Does: dataset download -> MediaPipe landmarks -> finger counting -> preprocessing -> train Keras MLP
      -> evaluation -> save gesture_model.keras -> verification demos -> figs/ and results.json
Then run:   python app.py   (live webcam verification)
"""
import matplotlib
matplotlib.use('Agg')
import os, json, glob, random, time
import numpy as np, cv2
import matplotlib.pyplot as plt, seaborn as sns
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
os.makedirs('figs', exist_ok=True)
RESULTS = {}          # every number reported later is stored here

print('TensorFlow', tf.__version__, '| MediaPipe', mp.__version__)

def ensure_dataset():
    """Download the public dataset: git clone if git exists, otherwise download the zip from GitHub."""
    for d in ('Sign-Language-Digits-Dataset/Dataset', 'Sign-Language-Digits-Dataset-master/Dataset'):
        if os.path.exists(d):
            return d
    os.system('git clone --depth 1 -q https://github.com/ardamavi/Sign-Language-Digits-Dataset.git')
    if os.path.exists('Sign-Language-Digits-Dataset/Dataset'):
        return 'Sign-Language-Digits-Dataset/Dataset'
    import urllib.request, zipfile
    print('git not available - downloading the dataset zip instead ...')
    urllib.request.urlretrieve('https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/refs/heads/master.zip', 'dataset.zip')
    zipfile.ZipFile('dataset.zip').extractall('.')
    return 'Sign-Language-Digits-Dataset-master/Dataset'

DATA_DIR = ensure_dataset()

CLASSES = [0, 1, 2, 3, 4, 5]
NAMES = {0: 'Zero (closed)', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five (open palm)'}
paths, labels = [], []
for c in CLASSES:
    fs = sorted(f for f in glob.glob(f'{DATA_DIR}/{c}/*') if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    paths += fs; labels += [c] * len(fs)
labels = np.array(labels)
print('Total images:', len(paths))
print({c: int((labels == c).sum()) for c in CLASSES})
RESULTS['n_images'] = len(paths)
RESULTS['per_class_images'] = {int(c): int((labels == c).sum()) for c in CLASSES}

def read_img(path, size=400):
    return cv2.resize(cv2.imread(path), (size, size))

fig, ax = plt.subplots(len(CLASSES), 5, figsize=(10, 12))
rng = np.random.RandomState(1)
for r, c in enumerate(CLASSES):
    idx = rng.choice(np.where(labels == c)[0], 5, replace=False)
    for k, i in enumerate(idx):
        ax[r, k].imshow(cv2.cvtColor(read_img(paths[i], 200), cv2.COLOR_BGR2RGB)); ax[r, k].axis('off')
    ax[r, 0].set_title(f'Class {c}', loc='left', fontsize=11)
plt.suptitle('Dataset samples (5 random images per class)', y=0.995)
plt.tight_layout(); plt.savefig('figs/fig_samples.png', dpi=130, bbox_inches='tight'); plt.show()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

def get_landmarks(img_bgr):
    res = detector.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None, None
    raw = res.multi_hand_landmarks[0]
    return np.array([[p.x, p.y, p.z] for p in raw.landmark], dtype=np.float32), raw

CACHE = 'landmarks_cache.npz'
if os.path.exists(CACHE):
    d = np.load(CACHE); LM, LAB, PATHS = d['lm'], d['lab'], d['paths']
else:
    lm_list, lab_list, path_list = [], [], []
    t0 = time.time()
    for p, y in zip(paths, labels):
        lm, _ = get_landmarks(read_img(p))
        if lm is not None:
            lm_list.append(lm); lab_list.append(y); path_list.append(p)
    LM, LAB, PATHS = np.array(lm_list), np.array(lab_list), np.array(path_list)
    np.savez(CACHE, lm=LM, lab=LAB, paths=PATHS)
    print(f'Extraction took {time.time()-t0:.0f}s')

rate = len(LM) / len(paths)
print(f'Hand detected in {len(LM)} of {len(paths)} images ({rate*100:.1f}%)')
RESULTS['n_detected'] = int(len(LM)); RESULTS['detection_rate'] = float(rate)
RESULTS['per_class_detected'] = {int(c): int((LAB == c).sum()) for c in CLASSES}

# Visual output: landmarks drawn on one image per class
fig, ax = plt.subplots(1, 6, figsize=(18, 3.6))
for k, c in enumerate(CLASSES):
    i = np.where(LAB == c)[0][3]
    img = read_img(PATHS[i]); lm, raw = get_landmarks(img)
    mp_draw.draw_landmarks(img, raw, mp_hands.HAND_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(0, 0, 255), thickness=6, circle_radius=5),
                           mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3))
    ax[k].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[k].set_title(f'Class {c}: 21 landmarks'); ax[k].axis('off')
plt.tight_layout(); plt.savefig('figs/fig_landmarks.png', dpi=130, bbox_inches='tight'); plt.show()

FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
TIPS, PIPS = [8, 12, 16, 20], [6, 10, 14, 18]

def finger_states(lm, margin=1.0, thumb_margin=1.05):
    xy = lm[:, :2]
    wrist = xy[0]
    states = []
    # thumb: tip (4) farther from pinky base (17) than the thumb IP joint (3)
    states.append(int(np.linalg.norm(xy[4] - xy[17]) > thumb_margin * np.linalg.norm(xy[3] - xy[17])))
    for t, p in zip(TIPS, PIPS):
        states.append(int(np.linalg.norm(xy[t] - wrist) > margin * np.linalg.norm(xy[p] - wrist)))
    return states            # [thumb, index, middle, ring, pinky] 1 = raised

def count_fingers(lm):
    return int(sum(finger_states(lm)))

rule_pred = np.array([count_fingers(l) for l in LM])
rule_acc = accuracy_score(LAB, rule_pred)
print(f'Rule-based finger count matches the true 0-5 label in {rule_acc*100:.1f}% of images')
for c in CLASSES:
    m = LAB == c
    print(f'  class {c}: {np.mean(rule_pred[m] == c)*100:.1f}%')
RESULTS['rule_count_acc'] = float(rule_acc)
RESULTS['rule_count_acc_per_class'] = {int(c): float(np.mean(rule_pred[LAB == c] == c)) for c in CLASSES}

fig, ax = plt.subplots(1, 6, figsize=(18, 4))
for k, c in enumerate(CLASSES):
    i = np.where(LAB == c)[0][8]
    img = read_img(PATHS[i]); lm, raw = get_landmarks(img)
    mp_draw.draw_landmarks(img, raw, mp_hands.HAND_CONNECTIONS)
    st = finger_states(lm)
    ax[k].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[k].axis('off')
    ax[k].set_title(f'Fingers raised: {sum(st)}\n' + ' '.join(n[0] + ('^' if s else 'v') for n, s in zip(FINGER_NAMES, st)),
                    color='green' if sum(st) == c else 'red')
plt.suptitle('Finger counting output (T=Thumb I=Index M=Middle R=Ring P=Pinky, ^ raised, v lowered)')
plt.tight_layout(); plt.savefig('figs/fig_finger_counting.png', dpi=130, bbox_inches='tight'); plt.show()

def normalize(lm):
    p = lm - lm[0]
    s = np.max(np.linalg.norm(p[:, :2], axis=1)) + 1e-6
    return (p / s).reshape(-1)

def augment(lm, rng):
    out = lm.copy()
    if rng.rand() < 0.5: out[:, 0] = 1 - out[:, 0]                     # mirror
    a = np.deg2rad(rng.uniform(-15, 15))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = out[:, :2].mean(0); out[:, :2] = (out[:, :2] - c) @ R.T + c   # rotate
    out *= rng.uniform(0.9, 1.1)                                       # scale
    out += rng.normal(0, 0.004, out.shape)                             # noise
    return out

idx = np.arange(len(LM))
i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=LAB, random_state=SEED)
i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=LAB[i_tmp], random_state=SEED)

rng = np.random.RandomState(SEED)
AUG = 4
X_tr = [normalize(LM[i]) for i in i_tr] + [normalize(augment(LM[i], rng)) for i in i_tr for _ in range(AUG)]
y_tr = list(LAB[i_tr]) + [LAB[i] for i in i_tr for _ in range(AUG)]
X_tr, y_tr = np.array(X_tr, dtype=np.float32), np.array(y_tr)
X_va = np.array([normalize(LM[i]) for i in i_va], dtype=np.float32); y_va = LAB[i_va]
X_te = np.array([normalize(LM[i]) for i in i_te], dtype=np.float32); y_te = LAB[i_te]
print('train (with augmentation):', X_tr.shape, '| validation:', X_va.shape, '| test:', X_te.shape)
RESULTS.update(n_train_orig=int(len(i_tr)), n_train_aug=int(len(X_tr)), n_val=int(len(X_va)), n_test=int(len(X_te)))

model = keras.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.3),
    layers.Dense(64, activation='relu'), layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(CLASSES), activation='softmax'),
], name='gesture_mlp')
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
RESULTS['n_params'] = int(model.count_params())

# Architecture diagram
fig, ax = plt.subplots(figsize=(13, 2.8)); ax.axis('off')
blocks = [('Input\n63 values\n(21 x 3)', '#dbeafe'), ('Dense 128\nReLU + BatchNorm\nDropout 0.3', '#bfdbfe'),
          ('Dense 64\nReLU\nDropout 0.3', '#93c5fd'), ('Dense 32\nReLU', '#60a5fa'), ('Output 6\nSoftmax\n(gestures 0-5)', '#fde68a')]
for k, (t, col) in enumerate(blocks):
    ax.add_patch(plt.Rectangle((k * 2.6, 0.2), 2.0, 1.6, fc=col, ec='#1e3a8a', lw=1.5))
    ax.text(k * 2.6 + 1.0, 1.0, t, ha='center', va='center', fontsize=10)
    if k < len(blocks) - 1: ax.annotate('', xy=((k + 1) * 2.6, 1.0), xytext=(k * 2.6 + 2.0, 1.0), arrowprops=dict(arrowstyle='->', lw=1.8))
ax.set_xlim(-0.2, 13.2); ax.set_ylim(0, 2)
plt.savefig('figs/fig_architecture.png', dpi=140, bbox_inches='tight'); plt.show()

cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=150, batch_size=32, callbacks=cb, verbose=0)
ep = len(hist.history['loss'])
print(f'Stopped after {ep} epochs')
print(f"Final train acc {hist.history['accuracy'][-1]:.4f} | val acc {hist.history['val_accuracy'][-1]:.4f}")
print(f"Final train loss {hist.history['loss'][-1]:.4f} | val loss {hist.history['val_loss'][-1]:.4f}")
RESULTS.update(epochs=int(ep), train_acc=float(hist.history['accuracy'][-1]), val_acc=float(hist.history['val_accuracy'][-1]),
               train_loss=float(hist.history['loss'][-1]), val_loss=float(hist.history['val_loss'][-1]))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(hist.history['accuracy'], label='Training'); ax[0].plot(hist.history['val_accuracy'], label='Validation')
ax[0].set_title('Model accuracy'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Accuracy'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(hist.history['loss'], label='Training'); ax[1].plot(hist.history['val_loss'], label='Validation')
ax[1].set_title('Model loss'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Loss'); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig('figs/fig_training.png', dpi=140, bbox_inches='tight'); plt.show()

probs = model.predict(X_te, verbose=0)
y_hat = probs.argmax(1)
test_acc = accuracy_score(y_te, y_hat)
test_loss = model.evaluate(X_te, y_te, verbose=0)[0]
print(f'Test accuracy: {test_acc*100:.2f}%  |  Test loss: {test_loss:.4f}  ({len(y_te)} unseen images)\n')
print(classification_report(y_te, y_hat, target_names=[f'Gesture {c}' for c in CLASSES], digits=3))
RESULTS['test_acc'] = float(test_acc); RESULTS['test_loss'] = float(test_loss)
rep = classification_report(y_te, y_hat, output_dict=True, digits=3)
RESULTS['per_class_f1'] = {int(c): float(rep[str(c)]['f1-score']) for c in CLASSES}
RESULTS['macro_f1'] = float(rep['macro avg']['f1-score'])

cm = confusion_matrix(y_te, y_hat)
RESULTS['confusion_matrix'] = cm.tolist()
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted gesture'); plt.ylabel('True gesture'); plt.title(f'Confusion matrix (test accuracy {test_acc*100:.1f}%)')
plt.savefig('figs/fig_confusion.png', dpi=140, bbox_inches='tight'); plt.show()

# Save and reload test
model.save('gesture_model.keras')
reloaded = keras.models.load_model('gesture_model.keras')
same = np.allclose(reloaded.predict(X_te, verbose=0), probs, atol=1e-5)
print('Model saved to gesture_model.keras and reloaded; predictions identical:', same)
RESULTS['save_load_ok'] = bool(same)

def classify_frame(img_bgr, conf_threshold=0.0):
    lm, raw = get_landmarks(img_bgr)
    out = img_bgr.copy()
    if lm is None:
        cv2.putText(out, 'No hand detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return dict(found=False, pred=None, conf=0.0, fingers=None, image=out)
    p = reloaded.predict(normalize(lm)[None], verbose=0)[0]
    pred, conf = int(p.argmax()), float(p.max())
    mp_draw.draw_landmarks(out, raw, mp_hands.HAND_CONNECTIONS)
    fingers = count_fingers(lm)
    cv2.rectangle(out, (0, 0), (out.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(out, f'Gesture: {pred}  ({conf*100:.0f}%)', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(out, f'Fingers raised: {fingers}', (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return dict(found=True, pred=pred, conf=conf, fingers=fingers, image=out)

test_paths = PATHS[i_te]; test_labels = LAB[i_te]
sel = [np.where(test_labels == c)[0][1] for c in CLASSES]
fig, ax = plt.subplots(1, 6, figsize=(18, 4))
for k, i in enumerate(sel):
    r = classify_frame(read_img(test_paths[i]))
    ax[k].imshow(cv2.cvtColor(r['image'], cv2.COLOR_BGR2RGB)); ax[k].axis('off')
    ax[k].set_title(f"True {test_labels[i]} -> Predicted {r['pred']}", color='green' if r['pred'] == test_labels[i] else 'red')
plt.suptitle('Gesture recognition output on unseen test photos')
plt.tight_layout(); plt.savefig('figs/fig_recognition.png', dpi=130, bbox_inches='tight'); plt.show()

def new_challenge(n=4):
    seq = [random.choice(CLASSES)]
    while len(seq) < n:
        c = random.choice(CLASSES)
        if c != seq[-1]: seq.append(c)
    return seq

def verify_sequence(challenge, frame_provider, min_conf=0.5):
    steps = []
    for k, target in enumerate(challenge):
        frame = frame_provider(k, target)
        r = classify_frame(frame)
        ok = bool(r['found'] and r['pred'] == target and r['conf'] >= min_conf)
        steps.append(dict(step=k + 1, target=target, pred=r['pred'], conf=r['conf'], fingers=r['fingers'], ok=ok, image=r['image']))
    return dict(challenge=challenge, steps=steps, verified=all(s['ok'] for s in steps))

def show_challenge(challenge, save=None):
    fig, ax = plt.subplots(1, len(challenge), figsize=(3.2 * len(challenge), 2.6))
    for k, c in enumerate(challenge):
        ax[k].set_facecolor('#eff6ff'); ax[k].set_xticks([]); ax[k].set_yticks([])
        ax[k].text(0.5, 0.72, f'STEP {k+1}', ha='center', fontsize=12, color='#1e3a8a', transform=ax[k].transAxes)
        ax[k].text(0.5, 0.40, f'Show {c}', ha='center', fontsize=22, weight='bold', transform=ax[k].transAxes)
        ax[k].text(0.5, 0.15, 'finger' + ('' if c == 1 else 's'), ha='center', fontsize=12, transform=ax[k].transAxes)
    plt.suptitle('Random verification challenge: perform these gestures in order', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    if save: plt.savefig(save, dpi=140, bbox_inches='tight')
    plt.show()

def show_result(res, title, save=None):
    n = len(res['steps'])
    fig, ax = plt.subplots(1, n, figsize=(3.6 * n, 4.4))
    for k, s in enumerate(res['steps']):
        ax[k].imshow(cv2.cvtColor(s['image'], cv2.COLOR_BGR2RGB)); ax[k].set_xticks([]); ax[k].set_yticks([])
        col = 'green' if s['ok'] else 'red'
        for sp in ax[k].spines.values(): sp.set_edgecolor(col); sp.set_linewidth(4)
        ax[k].set_title(f"Step {s['step']}: asked {s['target']} | got {s['pred']}\n{'MATCH' if s['ok'] else 'MISMATCH'}", color=col, fontsize=10)
    verdict = 'VERIFICATION SUCCESS - HUMAN VERIFIED' if res['verified'] else 'VERIFICATION FAILED'
    plt.suptitle(f"{title}\nChallenge: {res['challenge']}  ->  {verdict}", fontsize=13, color='green' if res['verified'] else 'red', weight='bold', y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    if save: plt.savefig(save, dpi=130, bbox_inches='tight')
    plt.show()

# ---- Random challenge generation demo (3 different challenges to show randomness) ----
random.seed(7)
demo_challenges = [new_challenge() for _ in range(3)]
print('Three random challenges generated:', demo_challenges)
RESULTS['random_challenges_demo'] = demo_challenges
show_challenge(demo_challenges[0], save='figs/fig_challenge.png')

def sample_provider_factory(performed):
    # returns a function giving an unseen test photo showing the gesture the "user" performs at each step
    r = np.random.RandomState(11)
    def provider(k, target):
        shown = performed[k]
        cand = np.where(test_labels == shown)[0]
        return read_img(test_paths[r.choice(cand)])
    return provider

challenge = demo_challenges[0]
# Demo A: genuine user performs exactly the requested gestures
resA = verify_sequence(challenge, sample_provider_factory(list(challenge)))
show_result(resA, 'Demo A - correct gestures performed', save='figs/fig_verify_success.png')

# Demo B: one gesture is wrong (step 3)
wrong = list(challenge); wrong[2] = [c for c in CLASSES if c != challenge[2]][0 if challenge[2] != 0 else 0]
resB = verify_sequence(challenge, sample_provider_factory(wrong))
show_result(resB, 'Demo B - one wrong gesture performed (step 3)', save='figs/fig_verify_fail.png')
RESULTS['demoA'] = dict(challenge=challenge, verified=resA['verified'], steps=[(s['target'], s['pred'], round(s['conf'], 3)) for s in resA['steps']])
RESULTS['demoB'] = dict(challenge=challenge, performed=wrong, verified=resB['verified'], steps=[(s['target'], s['pred'], round(s['conf'], 3)) for s in resB['steps']])
print('Demo A verified:', resA['verified'], '| Demo B verified:', resB['verified'])

def run_sessions(n_sessions, genuine, seed=123):
    rr = np.random.RandomState(seed); passed = 0
    by_cls = {c: np.where(test_labels == c)[0] for c in CLASSES}
    for _ in range(n_sessions):
        ch = [int(rr.choice(CLASSES))]
        while len(ch) < 4:
            c = int(rr.choice(CLASSES))
            if c != ch[-1]: ch.append(c)
        perf = ch if genuine else [int(rr.choice(CLASSES)) for _ in ch]
        ok = True
        for t, s in zip(ch, perf):
            i = rr.choice(by_cls[s])
            p = reloaded.predict(normalize(LM[i_te][i])[None], verbose=0)[0]
            ok &= (p.argmax() == t and p.max() >= 0.5)
        passed += ok
    return passed / n_sessions

N_SESS = 300
gen_rate = run_sessions(N_SESS, True)
imp_rate = run_sessions(N_SESS, False)
print(f'Genuine users accepted : {gen_rate*100:.1f}% ({N_SESS} simulated sessions)')
print(f'Random impostors accepted: {imp_rate*100:.1f}% ({N_SESS} simulated sessions)')
RESULTS.update(sim_sessions=N_SESS, sim_genuine_accept=float(gen_rate), sim_impostor_accept=float(imp_rate))

checks = [
    ('1. Detect hand landmarks (MediaPipe)', f"Landmarks found in {RESULTS['n_detected']}/{RESULTS['n_images']} images ({RESULTS['detection_rate']*100:.1f}%)",
     'figs/fig_landmarks.png', RESULTS['detection_rate'] >= 0.85),
    ('2. Identify raised/lowered fingers and count 0-5', f"Rule-based count correct on {RESULTS['rule_count_acc']*100:.1f}% of images",
     'figs/fig_finger_counting.png', RESULTS['rule_count_acc'] >= 0.70),
    ('3. Train a neural network gesture classifier', f"Trained {RESULTS['epochs']} epochs; train acc {RESULTS['train_acc']*100:.1f}%, val acc {RESULTS['val_acc']*100:.1f}%",
     'figs/fig_training.png', RESULTS['val_acc'] >= 0.80),
    ('4. Evaluate the model (accuracy, loss, confusion matrix)', f"Test accuracy {RESULTS['test_acc']*100:.1f}%, test loss {RESULTS['test_loss']:.3f}",
     'figs/fig_confusion.png', RESULTS['test_acc'] >= 0.80),
    ('5. Recognise gestures from new images', 'Predicted gesture shown for unseen test photos',
     'figs/fig_recognition.png', True),
    ('6. Generate random gesture challenges', f"Challenges generated: {RESULTS['random_challenges_demo']}",
     'figs/fig_challenge.png', len({tuple(c) for c in RESULTS['random_challenges_demo']}) > 1),
    ('7. Verify the sequence and show success/failure', f"Demo A verified={RESULTS['demoA']['verified']}, Demo B verified={RESULTS['demoB']['verified']}; genuine accept {RESULTS['sim_genuine_accept']*100:.1f}%, impostor accept {RESULTS['sim_impostor_accept']*100:.1f}%",
     'figs/fig_verify_success.png, figs/fig_verify_fail.png', RESULTS['demoA']['verified'] and not RESULTS['demoB']['verified']),
]
print(f"{'Objective':58s} | Result | Evidence")
for o, r, f, ok in checks:
    print(f"{o:58s} | {'ACHIEVED' if ok else 'NOT MET'} | {f}\n     -> {r}")
RESULTS['objectives'] = [dict(objective=o, result=r, evidence=f, achieved=bool(ok)) for o, r, f, ok in checks]
json.dump(RESULTS, open('results.json', 'w'), indent=2)
print('\nAll numbers saved to results.json')
