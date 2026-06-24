# SageMaker Serverless SFT Workshop

Hands-on workshop: fine-tune **Qwen3-4B** with **LoRA** on **SageMaker Serverless SFT**, using a real-world phishing-detection dataset.

## Goal

By the end attendees will have:

1. Pulled a curated phishing dataset from HuggingFace into SageMaker
2. Launched a serverless SFT-LoRA training job on Qwen3-4B without picking instance types, with live loss curves in MLflow
3. Deployed the fine-tuned adapter as a real-time endpoint
4. Measured a **base-model baseline** and the **fine-tuned model** with the same Custom Scorer, and quantified the lift
5. (Optional) Repeated the loop with the **vision** variant of Qwen3.5-4B on rendered page screenshots

## Source dataset

[`phreshphish/phreshphish`](https://huggingface.co/datasets/phreshphish/phreshphish) — 666k labeled phishing/benign webpage HTML+URL pairs (CC-BY-4.0, anti-phishing research only). The full dataset is **36.6 GB**, so we never download it whole — we stream it and slice 2k samples for the workshop.

## Architecture

```
PhreshPhish (HF, 36.6 GB, streaming)
     │
     ▼
[ prep/ ]  ─── Bill runs ONCE before workshop day  ──────────────────┐
                                                                     │
  prep/01_prep_text_dataset.ipynb                                    │
    └─► HF: gt2026workshop/phreshphish-2k  (config: text)            │
        1k phish + 1k benign, prompt+completion JSONL                │
                                                                     │
  prep/02_prep_image_dataset.ipynb                                   │
    └─► HF: gt2026workshop/phreshphish-2k  (config: image)           │
        rendered PNG screenshots, multimodal                         │
        ⚠️ ~30% renders fail or look degraded — see notebook caveat   │
                                                                     │
─────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
[ workshop/ ]  ─── attendees run on workshop day ──────────────────┐
                                                                   │
  workshop/01_sft_text_qwen3_4b.ipynb                              │
    ├─► HF dataset → S3 JSONL → SageMaker AI Registry              │
    ├─► SFTTrainer (LoRA) on huggingface-reasoning-qwen3-4b        │
    └─► (opt) temporary SageMaker endpoint for a smoke test        │
                                                                   │
  workshop/02_sft_vision_qwen3_5_4b.ipynb  (optional)              │
    ├─► HF image dataset → S3 + JSONL → AI Registry                │
    ├─► SFTTrainer (LoRA) on huggingface-vlm-qwen3-5-4b            │
    └─► Deploy → vision-text inference → eval                      │
                                                                   │
  workshop/03_evaluate_base_model.ipynb                            │
    ├─► Register validation set + reward fn in AI Registry         │
    ├─► Custom Scorer on BASE Qwen3-4B (no fine-tune needed)       │
    │     → baseline accuracy; can run before 01 finishes          │
    └─► (opt) base-model MMLU + LLM-as-a-Judge                     │
                                                                   │
  workshop/04_evaluate_finetuned_model.ipynb                       │
    ├─► Resolve fine-tuned model package from 01                   │
    ├─► Reuse the SAME dataset + reward fn from 03                 │
    ├─► Custom Scorer: fine-tuned vs base side-by-side             │
    ├─► (opt) MMLU — catastrophic forgetting check                 │
    └─► (opt) LLM-as-a-Judge with Claude on Bedrock                │
                                                                   │
  workshop/05_deploy_bedrock_custom_model_import.ipynb             │
    ⚠️ TAKE-HOME — run in your OWN account, not the workshop one   │
    ├─► Point at SFT job's pre-merged weights (hf_merged/) — no    │
    │     merge step needed                                        │
    ├─► Reuse the SageMaker execution role (Bedrock-assumable)     │
    ├─► create_model_import_job → poll → Completed                 │
    └─► Invoke via bedrock-runtime (serverless, pay-per-token)     │
                                                                   │
───────────────────────────────────────────────────────────────────┘
```

## Repository layout

```
.
├── README.md                              ← you are here
├── prep/
│   ├── requirements.txt
│   ├── 01_prep_text_dataset.ipynb         ← run once before workshop
│   └── 02_prep_image_dataset.ipynb        ← run once before workshop
└── workshop/
    ├── requirements.txt
    ├── 01_sft_text_qwen3_4b.ipynb         ← attendees run live
    ├── 02_sft_vision_qwen3_5_4b.ipynb     ← attendees run live (optional)
    ├── 03_evaluate_base_model.ipynb       ← baseline; can run before/while 01 trains
    ├── 04_evaluate_finetuned_model.ipynb  ← after 01; reuses 03's dataset + scorer
    └── 05_deploy_bedrock_custom_model_import.ipynb  ← take-home: Bedrock CMI in your own account
```

## Workshop-day flow & timing

| Phase | Time | Notebook | What happens |
|---|---|---|---|
| Welcome + concept intro | 10 min | — | What is SFT, what is LoRA, what does "serverless training" mean here |
| Walk through dataset | 5 min | 01 | `load_dataset(...)` from HF, peek at a phish row vs a benign row |
| Stage data into SageMaker | 10 min | 01 | Upload to S3, register `DataSet`, create `ModelPackageGroup` |
| Launch SFT job | 5 min | 01 | Configure `SFTTrainer` (incl. `mlflow_resource_arn`), kick off |
| **Baseline eval (while training runs)** | 10 min | **03** | Custom Scorer on the BASE model — runs in parallel, needs no fine-tune |
| ⏳ Training runs | 20–30 min | 01 | Watch `train_loss`/`eval_loss` curves in MLflow while 03 runs |
| **Fine-tuned eval + compare** | 10–15 min | **04** | Custom Scorer: fine-tuned vs base side-by-side (+ optional MMLU/LLMAJ) |
| Discussion + wrap-up | 5 min | — | Talk through results, what to fine-tune next, point to the take-home Bedrock deploy |
| **Total** | **~80–95 min** | | |

> **Take-home (notebook 05):** deploying to **Bedrock Custom Model Import** is a post-workshop exercise — the shared workshop account blocks `bedrock:CreateModelImportJob`, so run it in your own AWS account (~15–25 min). It points straight at the SFT job's pre-merged `hf_merged/` weights and reuses your SageMaker execution role.

> **Tip:** notebook 03 (baseline eval) needs no fine-tuned model, so run it *during* the 20–30 min training wait instead of after. It also registers the shared dataset + reward function that notebook 04 reuses.

> **Deployment options:** during the workshop, notebook 01 has an *optional* quick SageMaker endpoint (set `RUN_SMOKE_TEST = True`) for a fast in-session check — billed per hour, tear it down after. For a serverless production deployment, **notebook 05 → Amazon Bedrock Custom Model Import** is a take-home exercise to run in your own AWS account (the shared workshop account blocks it).

The vision variant (notebook 02) is a stretch goal — same template, different model ID, more interesting visuals.

> **Heads-up on the image variant:** PhreshPhish stores HTML only, not original screenshots. Re-rendering today produces visually degraded pages because (a) most modern sites are JS-rendered SPAs and we disable JS for safety, (b) referenced assets are often offline. About 70-80% of attempts succeed; of those, maybe half look reasonably page-like. **For the workshop's pedagogical goal — demonstrating SFT on a VLM — this is sufficient.** For production phishing classification you'd want screenshots captured at crawl time. See `prep/02_prep_image_dataset.ipynb` for full details.

## Prerequisites

### For Bill (workshop author) — before the workshop

- HuggingFace account with **write** access to `gt2026workshop`
- SageMaker Studio access OR an EC2/Cloud Desktop with:
  - Python 3.10+
  - ≥10 GB free disk for dataset cache
  - ≥3 GB extra for Chromium (image variant only)
- Run `prep/01_prep_text_dataset.ipynb` → publishes config `text`
- Run `prep/02_prep_image_dataset.ipynb` → publishes config `image`
- Verify the published HF dataset (`gt2026workshop/phreshphish-2k`) is public and both configs are visible

### For workshop attendees — on workshop day

- AWS account with a SageMaker AI domain
- Studio access with execution role having `AmazonSageMakerModelCustomizationCoreAccess`
- An S3 bucket for training output (any `sagemaker-*` named bucket works)
- A few minutes accepting the Qwen3 EULA via JumpStart UI before the session
- **An MLflow app** to reuse (for loss curves + eval metrics). The account is capped at 3 MLflow apps; the notebooks auto-discover an existing one. If none exists, create it once:
  `aws sagemaker create-mlflow-app --mlflow-app-name workshop-mlflow` — then the notebooks will find it.
- **A Lambda-assumable role** for the Custom Scorer reward function (eval notebooks 03/04). Create once:
  ```bash
  ACCT=$(aws sts get-caller-identity --query Account --output text)
  cat > lambda-trust.json <<EOF
  {"Version":"2012-10-17","Statement":[{"Effect":"Allow",
    "Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
  EOF
  aws iam create-role --role-name SageMakerForLambda-workshop \
    --assume-role-policy-document file://lambda-trust.json
  aws iam attach-role-policy --role-name SageMakerForLambda-workshop \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "arn:aws:iam::${ACCT}:role/SageMakerForLambda-workshop"
  ```
  Put the printed ARN into `LAMBDA_ROLE_ARN` in notebooks 03 and 04. The execution role must also allow `lambda:CreateFunction` (covered by `AmazonSageMakerModelCustomizationCoreAccess`) and `iam:PassRole` to `lambda.amazonaws.com`.
- **Bedrock Custom Model Import** is **not available in the shared workshop account** — it blocks `bedrock:CreateModelImportJob` via an account guardrail/SCP. Notebook 05 is therefore a **take-home exercise** for your own AWS account, where the execution role needs `bedrock:CreateModelImportJob`, `bedrock:GetModelImportJob`, `bedrock:ListImportedModels`, `bedrock:InvokeModel`, `iam:PassRole` to `bedrock.amazonaws.com`, and the role must be trusted by `bedrock.amazonaws.com`. Run in `us-east-1`/`us-east-2`/`us-west-2`/`eu-central-1`.

## Troubleshooting: Studio "Deploy to Bedrock" → "missing permission `bedrock:CreateModelImportJob`"

In the **workshop account this is expected** — the account blocks Bedrock custom-model creation by design. Use notebook 05 in your own account instead.

If you hit it in your **own** account: the error is about your **notebook/Studio execution role**, not your console SSO identity — they're different principals. Even if your inline policy looks correct, check:

1. **Right role?** `aws sts get-caller-identity` to see the role you're actually running as, then simulate: `aws iam simulate-principal-policy --policy-source-arn <execution-role-arn> --action-names bedrock:CreateModelImportJob`.
2. **`implicitDeny`** → no policy grants it; attach the Bedrock policy above to the **execution role**.
3. **`explicitDeny`** → an SCP or permission boundary blocks it (this is what the workshop account does). Adding Allows won't help — the account admin must adjust the SCP/boundary.
4. Custom Model Import is **region-gated** — calling it outside the 4 supported regions also fails.

## Troubleshooting: "no loss curves" / MLflow `ResourceLimitExceeded`

The SageMaker SDK auto-creates an MLflow app when you don't pass one. If the account is already at its 3-app quota, that create **fails silently** and training/eval run with tracking disabled — so you see no curves, and evaluation jobs can fail with a malformed-ARN error (`MLflow Resource ARN must match format...`).

Fix: pass an existing app's ARN. All notebooks now auto-discover one via `list_mlflow_apps` and set `MLFLOW_RESOURCE_ARN`; you can also hardcode it. To free a slot or list apps:

```bash
aws sagemaker list-mlflow-apps --query 'Summaries[*].[Name,Status,Arn]' --output table
aws sagemaker delete-mlflow-app --mlflow-app-name <old-name>   # frees a quota slot
```

## Why Qwen3-4B?

- **Small enough** to LoRA-tune end-to-end inside a 75-minute workshop session
- **Capable enough** that LoRA on 1.8k examples produces a visibly improved classifier
- **Available in JumpStart** as `huggingface-reasoning-qwen3-4b` — supports SFT, DPO, RLAIF, RLVR
- **Has a sibling VLM** (`huggingface-vlm-qwen3-5-4b`) so the workshop has a natural "now let's try vision" stretch goal

Full list of supported open-weight models for SageMaker customization: <https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-open-weight.html>

## Why label-only completion (`phish` / `benign`)?

- Trivial to evaluate exactly — accuracy = `pred == truth`
- Single-token completions train fast and surface LoRA's effect cleanly
- Lets attendees focus on the SDK + workflow rather than parsing model output
- Easy to extend later: brand-target prediction, reasoning chains, JSON output

## Cost estimate (rough)

| Item | Cost |
|---|---|
| SageMaker SFT-LoRA on Qwen3-4B, ~1.8k samples, 1 epoch | ~$5–15 per attendee |
| Real-time endpoint while live (~30 min) | ~$1–3 per attendee |
| S3 storage for artifacts | <$0.10 |
| **Per-attendee total** | **~$6–20** |

Costs vary by region and current SageMaker serverless training pricing — confirm with your AWS account team before scaling out.

## Key references

- [PhreshPhish dataset card](https://huggingface.co/datasets/phreshphish/phreshphish) — schema, license, paper
- [SageMaker open-weight customization docs](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-open-weight.html) — supported models, IAM, samples
- [SDK example notebook](https://sagemaker.readthedocs.io/en/stable/v3-examples/model-customization-examples/sft_finetuning_example_notebook_pysdk_prod_v3.html) — end-to-end Python SDK walkthrough
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-4B) — license, capabilities

## License

Workshop scaffolding: MIT.

Underlying assets carry their own licenses:
- PhreshPhish dataset: CC-BY-4.0, anti-phishing research only
- Qwen3 / Qwen3.5 models: Qwen license (acceptable for fine-tuning; check current terms)
