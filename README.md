# RA-LMM Risk Advisor

RA-LMM Risk Advisor is a research-use decision-support application for low muscle mass risk assessment in patients with rheumatoid arthritis. It combines a local logistic regression model, SHAP-based feature explanations, and DeepSeek-assisted interpretation and interactive discussion.

Logistic regression generates the predicted probability, and the application assigns the risk category using fixed cutoffs. SHAP describes feature contributions to the prediction. The LLM explains the active patient or cohort result.

## Features

- Single-patient assessment from manual input
- Batch assessment from uploaded CSV files
- Top three SHAP key drivers and contribution directions
- Downloadable batch template and prediction results
- LLM-generated summaries and interactive discussion of the current results
- Docker deployment and Windows start/stop scripts

## Quick Start

Install and start Docker Desktop. On Windows, double-click `run.bat` to pull and launch the published Clinical + Blood image. Use `stop.bat` to stop it.

To launch the published image from the command line:

```powershell
docker run --pull=always --rm --name ra-lmm-clinical-blood-check -p 127.0.0.1:3840:7860 blythe0827/ra-lmm-risk-advisor-variants:clinical-blood-v1.0.0
```

Open [http://127.0.0.1:3840](http://127.0.0.1:3840). The application uses port 7860 inside the container. Stop any existing container that uses local port 3840 before launching another one.

To build and run the source in this directory:

```powershell
docker build -t ra-lmm-risk-advisor-clinical-blood .
docker run --rm -d -p 127.0.0.1:3840:7860 --name ra-lmm-clinical-blood-check ra-lmm-risk-advisor-clinical-blood
```

View logs:

```powershell
docker logs -f ra-lmm-clinical-blood-check
```

Stop the application:

```powershell
docker stop ra-lmm-clinical-blood-check
```

The `--rm` option removes the container when it stops; it does not delete the image. Download any results you need before stopping the application.

## Model Inputs

Both assessment workflows use these six predictors. CSV column names are case-sensitive.

| Column | Description | Unit or encoding |
| --- | --- | --- |
| `ANC` | Absolute neutrophil count | 10^9/L |
| `ALT` | Alanine aminotransferase | U/L |
| `Cr` | Creatinine | μmol/L |
| `Sex` | Biological sex | 1 = Male; 2 = Female |
| `Age` | Age | Years |
| `BMI` | Body mass index | kg/m² |

Use values in their original clinical units; do not standardize or normalize inputs before upload. The application converts continuous inputs to numeric values and encodes `Sex` as a factor.

### Batch CSV example

The following synthetic example illustrates the input format and does not contain patient records:

```csv
PatientID,ANC,ALT,Cr,Sex,Age,BMI
Example001,3.5,25,60,1,65,22.5
Example002,4.0,30,70,2,70,24.0
Example003,2.8,40,80,1,75,21.0
```

`PatientID` is optional. If omitted, the application generates identifiers. Use non-identifying study codes. A downloadable CSV template is also available in the batch assessment interface.

The application checks for all six required columns and converts numeric fields before prediction. This is not comprehensive input validation: numeric conversion can produce missing values, and unrecognized character values for `Sex` can be mapped to Male by the current encoding logic. Check missing values, units, numeric entries, and sex codes before uploading. The manual input interface also specifies minimum and maximum values for numeric controls.

## Risk Stratification

Single-patient and batch assessments use the same logistic regression probability cutoffs:

| Risk category | Predicted probability P |
| --- | --- |
| Low | P < 0.0438 |
| Intermediate | 0.0438 ≤ P < 0.1442 |
| High | P ≥ 0.1442 |

Categories are assigned using the model probability before display rounding. SHAP arrows indicate contributions toward higher or lower predicted probability; they do not establish causality.

## DeepSeek Interpretation and Discussion

The application calls `deepseek-v4-pro` through `ellmer::chat_deepseek()`. Result interpretation uses a temperature of 0.20 and interactive discussion uses 0.35. The application uses keywords in the user's question to select a predefined discussion task within the application. The API request includes that task and the active result summary.

Both modules request JSON containing only an `answer` string and validate the response against a JSON schema. Prompts restrict answers to the supplied model results and prohibit diagnostic, treatment, medication, testing, and management recommendations. The interpretation module additionally checks prohibited text patterns; the discussion module's response check currently checks only emptiness and minimum length. Responses failing these checks are retried once, with predefined fallback text used after repeated failure or an API error.

These controls do not automatically verify every number, risk label, or SHAP direction in generated text and do not establish clinical safety. The LLM component has not undergone an independent systematic evaluation of numerical fidelity, hallucinations, reproducibility, usability, or clinical impact. Prediction-model validation does not validate the LLM or the entire application.

### API key configuration

The application first reads `DEEPSEEK_API_KEY` from its runtime environment. If it is absent, a password input dialog asks the user to enter a key and select **Save & Start App**. Manually entered keys are held in the current Shiny session and are not written to a file by the application. The interface currently has no separate key-change control.

For environment-based configuration, set `DEEPSEEK_API_KEY` in your PowerShell session, then pass it when creating the container:

```powershell
docker run --pull=always --rm -d -p 127.0.0.1:3840:7860 --name ra-lmm-clinical-blood-check -e DEEPSEEK_API_KEY blythe0827/ra-lmm-risk-advisor-variants:clinical-blood-v1.0.0
```

Environment-based credentials are managed by your host/container configuration. Do not commit keys to source control. A valid API account with available credit is required for LLM responses. HTTP 402 indicates a payment or credit issue; logistic regression scoring is independent of the API, and failed LLM calls return fallback text. Fallback answers may be cached within the session; start a new session after correcting API configuration.

### Data privacy

Logistic regression prediction and SHAP computation run in the application environment. Single-patient prompts include the predicted probability, risk category, cutoffs, and up to three SHAP feature names and contribution directions. Batch prompts include cohort size, mean predicted probability, risk-category counts and percentages, and cutoffs. The application excludes patient identifiers, raw clinical input values, individual CSV records, free-text questions, and discussion history when constructing DeepSeek prompts.

Single-patient prompts still contain patient-level model outputs. The application does not implement automatic de-identification or user authentication. Use non-identifying study codes and avoid entering directly identifiable patient information. External API retention and processing depend on the provider's policies and account arrangements. Some application logs include prediction probabilities and request metadata.

## Software Environment

The Dockerfile specifies `rocker/shiny-verse:4.4.3` and mlr3 1.3.0. Other dependencies, including Shiny, kernelshap, shapviz, ellmer, jsonvalidate, and logger, are installed during the build and are not all pinned to exact versions. Rebuilding the Dockerfile later may therefore produce a different dependency environment.

To inspect package versions in a running container:

```powershell
docker exec ra-lmm-clinical-blood-check Rscript -e "sessionInfo()"
```

To record the local image ID:

```powershell
docker image inspect blythe0827/ra-lmm-risk-advisor-variants:clinical-blood-v1.0.0 --format '{{.Id}}'
```

## Repository Structure

```text
RA-LMM-Risk-Advisor-Clinical+Blood/
|-- .dockerignore
|-- .gitattributes
|-- .gitignore
|-- app.R
|-- Dockerfile
|-- models/
|   |-- background_data.rds
|   |-- best_params.json
|   |-- logreg_model.rds
|   |-- feature_cols.rds
|   |-- target_levels.rds
|   |-- test_data.rds
|   `-- train_data.rds
|-- run.bat
|-- stop.bat
`-- README.md
```

The application loads the trained model, feature list, target levels, and SHAP background data from `models/`. `MODEL_DIR` can override this location; the Docker image sets it to `/home/hfuser/app/models`.

## Intended Use

RA-LMM Risk Advisor is a decision-support tool for research use. Generated explanations are not definitive diagnoses or treatment recommendations. Independent evaluation of the LLM component and prospective assessment of the complete system are required before clinical deployment.
