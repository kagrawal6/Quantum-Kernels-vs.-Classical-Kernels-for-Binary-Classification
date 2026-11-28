# Running the QSVM on real quantum hardware (AWS Braket)

This runs one small quantum-kernel experiment on a real quantum computer and
compares it with the simulator. **Planned cost: about $29** of the AWS credit.
Nothing is ever charged without you typing the dollar amount to confirm.

What it does: Breast Cancer data, 2 qubits, 8 training + 6 test points.
That is 76 kernel entries = 76 circuits, 200 shots each, on Rigetti (cheapest device).

---

## 1. One-time setup (~15 min)

**Code**

```bash
git checkout steve-branch && git pull
pip install -r requirements.txt          # includes qiskit-braket-provider
```

**AWS account (the one with the credit)**

1. In the AWS console, open **Amazon Braket** once and accept its terms
   ("Get started" / enable). This also creates the S3 bucket Braket needs.
2. **IAM → Users → Create user** (e.g. `braket-runner`), attach the policy
   **AmazonBraketFullAccess**. Then **Security credentials → Create access key**
   ("Command Line Interface"). Copy the key ID and secret.
3. On your laptop:

   ```bash
   pip install awscli        # if `aws` isn't installed
   aws configure
   #   Access key ID:      <paste>
   #   Secret access key:  <paste>
   #   Default region:     us-west-1     (Rigetti devices live here)
   #   Output format:      json
   ```

   Never paste the keys into chat, Slack or a git commit.
4. Recommended: **Billing → Budgets → Create budget**, monthly, $40, with an
   email alert. The script has its own caps, but this catches anything else.

---

## 2. Run it (in this order)

All commands from the repo root.

| Step | Command | Costs |
|---|---|---|
| 0. Which devices are online | `python scripts/run_hardware.py --list-devices` | free |
| 1. Cost plan | `python scripts/run_hardware.py` | free |
| 2. Full dry run on a local simulator | `python scripts/run_hardware.py --local braket_local` | free |
| 3. One circuit on the real device | `python scripts/run_hardware.py --smoke-test` | ~$0.39 |
| 4. The real run | `python scripts/run_hardware.py --submit` | ~$29 |

**Step 0:** the device in `configs/default.yaml` is `Cepheus-1-108Q` (Rigetti).
If the list shows a different Rigetti name, or it is OFFLINE, see Troubleshooting.

**Step 1:** must end with `budget check: OK`.

**Step 2:** takes a minute, spends nothing, and proves the whole pipeline works
on your machine. It writes `results/hardware/braket_local_.../`.

**Step 3:** you will be asked to type `0.39` to confirm. Real devices have a
queue, so this can take **minutes to hours**. It prints the device's kernel value
next to the exact one; they should be roughly close (within ~0.1).

**Step 4:** you will be asked to type `29.26`. Circuits go out in batches of 10.
Progress and job IDs are saved after every batch, so if your laptop sleeps or
the script crashes, **just run the same command again**: it re-attaches to
submitted jobs and only pays for circuits that were never sent.

**Note:** Rigetti devices only run during certain hours (check the availability window in the Braket console). Outside them, jobs wait in the queue.

---

## 3. What to send back

The folder `results/hardware/Cepheus-1-108Q_breast_cancer_f2_n8_t6_s200_zz1l_bw0.25/`
(name may differ slightly), which contains:

- `summary.csv` — the numbers (exact vs simulated shot noise vs device)
- `kernel_vs_exact.png` — the graph for the slides
- `kernels.npz`, `entries.json` — raw data

Plus `results/hardware/spend_ledger.csv` (what was spent). Either push them to
`steve-branch` (`git add -f results/hardware/` — the folder is gitignored) or zip and send.

**How to read the result:** look at `kernel_mae_vs_exact` (average error of the
kernel values). `shot_sim_200` shows the error from shot noise alone; anything
the device adds on top of that is **hardware noise**. Accuracy on 6 test points
means little.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `NoRegionError` / credentials error | Run `aws configure` again; region `us-west-1`. |
| Device name not found | Run step 0. Put the listed Rigetti name in `configs/default.yaml` as `device:`, **and** rename the matching key under `prices:` to the same name. Check the per-shot price at aws.amazon.com/braket/pricing. |
| Device OFFLINE or long queue | Wait for its availability window, or ask Steve about switching to IQM Garnet (~$45, region `eu-north-1`). |
| `budget check` fails | Don't override it. Something changed (price, settings or past spend); ask Steve. |
| Script interrupted | Run the same command again; it resumes. |
| Access denied | The IAM user needs `AmazonBraketFullAccess`; Braket must be enabled in the console. |

Safety caps (in `configs/default.yaml`): one run ≤ $50, all runs together ≤ $80,
estimates padded by 20%. Every charge is logged to `results/hardware/spend_ledger.csv`
**before** it is submitted.
