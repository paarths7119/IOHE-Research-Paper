import gradio as gr
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# ── load model once at startup ────────────────────────
model = joblib.load('model.pkl')

N_BITS = 2048

def get_fingerprint(smi):
    """Return a 2048-bit Morgan fingerprint as float32, or None."""
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return None
    return np.array(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=N_BITS),
        dtype=np.float32
    )

def get_pair_features(smi_a, smi_b):
    """
    Build the same 8193-dim feature vector used during training:
      fp_A (2048) + fp_B (2048) + shared (2048) + diff (2048) + cosine (1)
    """
    fp_a = get_fingerprint(smi_a)
    fp_b = get_fingerprint(smi_b)
    if fp_a is None or fp_b is None:
        return None

    shared = fp_a * fp_b                # element-wise AND on binary FPs
    diff   = np.abs(fp_a - fp_b)        # element-wise absolute difference

    # cosine similarity
    norm_a = np.linalg.norm(fp_a)
    norm_b = np.linalg.norm(fp_b)
    if norm_a == 0 or norm_b == 0:
        cosine = np.array([0.0], dtype=np.float32)
    else:
        cosine = np.array(
            [float(np.dot(fp_a, fp_b) / (norm_a * norm_b))],
            dtype=np.float32
        )

    return np.concatenate([fp_a, fp_b, shared, diff, cosine])

def predict(smi_a, smi_b):
    feat = get_pair_features(smi_a, smi_b)
    if feat is None:
        return "❌ One or both SMILES strings are invalid"

    feat  = feat.reshape(1, -1)
    prob  = model.predict_proba(feat)[0][1]
    label = model.predict(feat)[0]

    if label == 1:
        return (f"⚠️  Likely INTERACTION detected\n"
                f"Confidence: {prob:.1%}\n\n"
                "Note: This is a research model — always verify clinically.")
    else:
        return (f"✅  No interaction predicted\n"
                f"Confidence: {1-prob:.1%}\n\n"
                "Note: Absence of prediction ≠ absence of interaction.")

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Drug A — SMILES",
                   placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"),
        gr.Textbox(label="Drug B — SMILES",
                   placeholder="e.g. c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"),
    ],
    outputs=gr.Textbox(label="Result"),
    title="DDI Predictor — TwoSides + Morgan FP + ML",
    description="Enter canonical SMILES for two drugs. Get them from pubchem.ncbi.nlm.nih.gov",
    examples=[
        ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"],  # Aspirin + Ibuprofen
        ["CC(=O)Oc1ccccc1C(=O)O", "CC1=C2CC(CC2=CC(=C1)OC)c3ccc(cc3)OC"],
    ]
)

demo.launch(share=True)   # share=True → public URL, no deployment needed