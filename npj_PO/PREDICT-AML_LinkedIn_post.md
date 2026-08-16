## PREDICT-AML — LinkedIn post (draft)

---

🧬 **New work out from our lab: PREDICT-AML** - an AI framework that looks at a leukemia patient's molecular profile and predicts which drugs will actually work for *them*.

Accepted in the **Journal of Translational Medicine**. 📄

In acute myeloid leukemia, two patients can walk into the same clinic with the same diagnosis - and the drug that saves one does nothing for the other. Right now, we mostly find out the hard way.

---

**What we built:**

🧪 **Patient side:** gene expression, somatic mutations, oncogenic pathway activity, and cell-state programs, drawn from 942 BeatAML specimens across 805 patients.

💊 **Drug side:** four complementary molecular representations - physicochemical descriptors, our knowledge-distilled KD-Embed vectors (trained on ~2.5M compounds), plus ChemBERTa and MolFormer chemical language models.

🕸️ **The bridge:** a patient-specific *drug–pathway proximity* score - each drug's targets are dropped onto the STRING protein network, propagated by random walk, and weighted by that patient's own RNA-seq to ask how close a drug sits to the pathways actually active in that person.

🤖 **The engine: TabPFN** - a tabular foundation model that adapts in a single forward pass, no gradient updates. It beat 10 other architectures (trees, CNNs, LSTMs, GNNs) across 132 configurations, with the smallest CV-to-test gap of any top model.

---

**What's genuinely new here:**

🚨 **We benchmarked ourselves honestly.** Standard random cross-validation inflates apparent performance by ~13%, since bits of the same patient land on both sides of the split. **Patient-stratified CV** - the split that mimics a truly new patient - is the benchmark that matters, and most published AML models skip it.

🩺 **You may not need a full multi-omics workup.** Gene expression alone recovers 98.8% of full-model performance. Separately, mutations + cell-state enrichment alone - no expression data at all - nearly match it too (Spearman r = 0.685). Two independent, cheap, reduced-panel options for resource-limited settings.

🧪 **Chemical language models earn their keep on novel drugs.** Under drug-stratified CV (compounds withheld entirely from training), ChemBERTa/MolFormer embeddings lift correlation from 0.31 to 0.38 over physicochemical descriptors alone.

🌍 **Validated cold on two independent cohorts.** Trained on BeatAML (US) only, tested on **LeeAML** (r = 0.597, beating published MDREAM) and **FIMM-AML** in Finland (|r| = 0.557, on a different sensitivity metric entirely). Different countries, labs, and assays - it held.

🥊 **Benchmarked against the latest models, not just the classics.** Beyond ElasticNet (r = 0.36) and MDREAM (r = 0.68), we compare against 2025's NetAML, which trains 87 separate drug-specific models with leakage-prone per-drug splits and one external cohort. PREDICT-AML is one unified model spanning all 150 test drugs (14 never seen in training), across four omics modalities, validated on two external cohorts at once.

🔍 **It explains itself in two tiers.** SHAP shows drug identity sets the prediction's *direction*, patient biology *tunes* it - independently rediscovering IDH2/BCL-2 synthetic lethality and monocytic Venetoclax resistance, with no pathway supervision.

🎯 **And it flagged things we didn't expect:** *POU4F1* as a candidate sensitivity marker, *BCL6* significant across all ten top drugs, and immunogenic cell death signalling as a possible multi-drug biomarker axis. All model-derived hypotheses - now they need a wet lab. 🧫

---

**Where it matters most:** Venetoclax and Panobinostat show the widest patient-to-patient variability in predicted response - exactly where one-size-fits-all prescribing does the most damage. Conversely, PREDICT-AML correctly flags drugs like Vismodegib as uniformly ineffective. Knowing when *not* to bother is its own kind of useful.

---

**Credit:** **Mohammed Al-Ani**, **Siddhi P. Jani**, and **Halima Bensmail** - an in-house **QCRI/HBKU** effort, supported by the **Qatar National Library**. And none of it exists without the **BeatAML consortium** publishing a decade of hard-won patient data openly.

🔗 Code, data and workflows: github.com/raghvendra5688/PT-AML2.0

Next up: extending PREDICT-AML to patient data from the **Qatar Precision Health Institute (QPHI)**.

---

*#AML #Leukemia #PrecisionMedicine #MachineLearning #MultiOmics #TabPFN #FoundationModels #Bioinformatics #DrugResponse #BeatAML #ExplainableAI #SHAP #Oncology #OpenScience #QCRI #HBKU*

