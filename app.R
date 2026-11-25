library(shiny)
library(data.table)
library(mlr3verse)
library(mlr3)
library(mlr3tuningspaces)
library(kernelshap)
library(shapviz)
library(ggplot2)
library(DT)
library(jsonlite)
library(shinythemes)
library(shinyWidgets)
library(jsonvalidate)
library(logger)
options(ellmer_echo = "none")

# Project Configuration
LOW_RISK_THRESHOLD <- 0.0438
HIGH_RISK_THRESHOLD <- 0.1442
RISK_THRESHOLDS <- c(LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD)

get_app_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  file_path <- sub(file_arg, "", args[grep(file_arg, args)])

  if (length(file_path) > 0) {
    return(dirname(normalizePath(file_path[1], winslash = "/", mustWork = TRUE)))
  }

  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

MODEL_DIR <- {
  model_dir_env <- trimws(Sys.getenv("MODEL_DIR", unset = ""))
  if (nzchar(model_dir_env)) {
    normalizePath(model_dir_env, winslash = "/", mustWork = TRUE)
  } else {
    normalizePath(file.path(get_app_dir(), "models"), winslash = "/", mustWork = TRUE)
  }
}

# Key Drivers Computation Function
compute_key_drivers <- function(learner, X_new, bg_X, feature_cols, target_levels) {
  n <- nrow(X_new)
  empty_drivers <- data.table(
    driver1 = rep(NA_character_, n),
    driver2 = rep(NA_character_, n),
    driver3 = rep(NA_character_, n)
  )

  tryCatch({
    bg_X_features <- bg_X[, feature_cols, with = FALSE]

    ps <- permshap(
      object = learner,
      X = X_new,
      bg_X = bg_X_features,
      predict_type = "prob"
    )

    ps_result <- shapviz(ps)
    sv_pos <- ps_result[["Low"]]
    shap_mat <- as.matrix(sv_pos$S)

    feature_names <- colnames(shap_mat)

    drivers_mat <- matrix(NA_character_, nrow = n, ncol = 3)

    for (i in seq_len(n)) {
      shap_values <- shap_mat[i, ]
      ord <- order(abs(shap_values), decreasing = TRUE)
      k <- min(3, length(ord))
      idx <- ord[seq_len(k)]
      shap_vals <- shap_values[idx]
      feature_names_sel <- feature_names[idx]

      descriptions <- character(k)

      for (j in seq_len(k)) {
        feat_name <- feature_names_sel[j]
        shap_val <- shap_vals[j]
        feat_val <- X_new[[feat_name]][i]

        if (feat_name == "Sex") {
          sex_label <- ifelse(as.character(feat_val) == "1", "Male", "Female")
          direction <- ifelse(shap_val >= 0, "\u2191", "\u2193")
          descriptions[j] <- paste0("Sex(", sex_label, ") ", direction)
        } else if (is.numeric(feat_val)) {
          direction <- ifelse(shap_val >= 0, "\u2191", "\u2193")
          descriptions[j] <- paste0(feat_name, "=", round(feat_val, 1), " ", direction)
        } else {
          direction <- ifelse(shap_val >= 0, "\u2191", "\u2193")
          descriptions[j] <- paste0(feat_name, " ", direction)
        }
      }

      drivers_mat[i, seq_len(k)] <- descriptions
    }

    return(data.table(
      driver1 = drivers_mat[, 1],
      driver2 = drivers_mat[, 2],
      driver3 = drivers_mat[, 3]
    ))

  }, error = function(e) {
    message("SHAP calculation error: ", e$message)
    return(empty_drivers)
  })
}

generate_ai_interpretation <- function(risk_level, probability, key_drivers, api_key, cache_env = NULL) {
  driver1 <- if(!is.na(key_drivers$driver1[1])) key_drivers$driver1[1] else "None"
  driver2 <- if(!is.na(key_drivers$driver2[1])) key_drivers$driver2[1] else "None"
  driver3 <- if(!is.na(key_drivers$driver3[1])) key_drivers$driver3[1] else "None"
  prompt_drivers <- Advisor_prompt_drivers_text(key_drivers)
  cache_key <- NULL

  tryCatch({
    context <- sprintf(
      paste(
        "Current RA-LMM Risk Advisor result:",
        "Risk level: %s.",
        "Predicted probability of low muscle mass: %.1f%%.",
        "Risk cutoffs: Low: P < %.4f; Intermediate: %.4f <= P < %.4f; High: P >= %.4f.",
        "Top SHAP key drivers (feature and direction only): %s.",
        sep = "\n"
      ),
      risk_level, probability * 100,
      LOW_RISK_THRESHOLD, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
      prompt_drivers
    )
    task_type <- "risk summary"
    question_id <- "single_overview"
    question <- "Summarize the current result in a natural research-facing interpretation."
    cache_key <- Advisor_cache_key("single_overview", context, question_id, question, task_type)
    cached_response <- Advisor_cache_get(cache_env, cache_key)

    if (!is.null(cached_response)) {
      log_info("Single AI interpretation cache hit: question_id={question_id}, task_type={task_type}, probability={probability}")
      return(cached_response)
    }

    log_info("Single AI interpretation cache miss: question_id={question_id}, task_type={task_type}, probability={probability}")

    result <- Advisor_request_with_retry(
      context = context,
      question_id = question_id,
      question = question,
      task_type = task_type,
      api_key = api_key,
      log_prefix = "Single AI interpretation"
    )

    if (!is.null(result$answer)) {
      Advisor_cache_set(cache_env, cache_key, result$answer)
      return(result$answer)
    }

    fallback_text <- sprintf(
      "The model-derived probability is %.1f%% and is classified as %s risk under the current cutoffs. The strongest displayed SHAP contributors are %s. Upward arrows indicate higher model-derived probability, while downward arrows indicate lower model-derived probability.",
      probability * 100, risk_level, paste(c(driver1, driver2, driver3), collapse = "; ")
    )
    Advisor_cache_set(cache_env, cache_key, fallback_text)
    return(fallback_text)
  }, error = function(e) {
    log_warn("Single AI interpretation fell back to deterministic text: {e$message}")
    fallback_text <- sprintf(
      "The model-derived probability is %.1f%% and is classified as %s risk under the current cutoffs. The strongest displayed SHAP contributors are %s. Upward arrows indicate higher model-derived probability, while downward arrows indicate lower model-derived probability.",
      probability * 100, risk_level, paste(c(driver1, driver2, driver3), collapse = "; ")
    )
    Advisor_cache_set(cache_env, cache_key, fallback_text)
    return(fallback_text)
  })
}

generate_batch_ai_interpretation <- function(summary_stats, api_key, cache_env = NULL) {
  cache_key <- NULL
  tryCatch({
    context <- sprintf(
      paste(
        "Current RA-LMM Risk Advisor batch result:",
        "Total patients: %d.",
        "Mean predicted probability of low muscle mass: %.1f%%.",
        "Risk distribution: High=%d (%.1f%%); Intermediate=%d (%.1f%%); Low=%d (%.1f%%).",
        "Risk cutoffs: Low: P < %.4f; Intermediate: %.4f <= P < %.4f; High: P >= %.4f.",
        sep = "\n"
      ),
      summary_stats$total_patients,
      summary_stats$mean_probability * 100,
      summary_stats$high_risk, (summary_stats$high_risk / summary_stats$total_patients) * 100,
      summary_stats$intermediate_risk, (summary_stats$intermediate_risk / summary_stats$total_patients) * 100,
      summary_stats$low_risk, (summary_stats$low_risk / summary_stats$total_patients) * 100,
      summary_stats$thresholds[1], summary_stats$thresholds[1], summary_stats$thresholds[2], summary_stats$thresholds[2]
    )
    task_type <- "cohort summary"
    question_id <- "batch_overview"
    question <- "Summarize the cohort result in a natural research-facing interpretation."
    cache_key <- Advisor_cache_key("batch_overview", context, question_id, question, task_type)
    cached_response <- Advisor_cache_get(cache_env, cache_key)

    if (!is.null(cached_response)) {
      log_info("Batch AI interpretation cache hit: question_id={question_id}, task_type={task_type}, total_patients={summary_stats$total_patients}")
      return(cached_response)
    }

    log_info("Batch AI interpretation cache miss: question_id={question_id}, task_type={task_type}, total_patients={summary_stats$total_patients}")

    result <- Advisor_request_with_retry(
      context = context,
      question_id = question_id,
      question = question,
      task_type = task_type,
      api_key = api_key,
      log_prefix = "Batch AI interpretation"
    )

    if (!is.null(result$answer)) {
      Advisor_cache_set(cache_env, cache_key, result$answer)
      return(result$answer)
    }

    fallback_text <- sprintf(
      "This batch contains %d patients. The mean model-derived probability is %.1f%%, and the current distribution is High=%d, Intermediate=%d, and Low=%d under the current numeric cutoffs.",
      summary_stats$total_patients, summary_stats$mean_probability * 100,
      summary_stats$high_risk, summary_stats$intermediate_risk, summary_stats$low_risk
    )
    Advisor_cache_set(cache_env, cache_key, fallback_text)
    return(fallback_text)
  }, error = function(e) {
    log_warn("Batch AI interpretation fell back to deterministic text: {e$message}")
    fallback_text <- sprintf(
      "This batch contains %d patients. The mean model-derived probability is %.1f%%, and the current distribution is High=%d, Intermediate=%d, and Low=%d under the current numeric cutoffs.",
      summary_stats$total_patients, summary_stats$mean_probability * 100,
      summary_stats$high_risk, summary_stats$intermediate_risk, summary_stats$low_risk
    )
    Advisor_cache_set(cache_env, cache_key, fallback_text)
    return(fallback_text)
  })
}

Advisor_agent_schema <- '{
  "type": "object",
  "required": ["answer"],
  "properties": {
    "answer": {"type": "string"}
  },
  "additionalProperties": false
}'

Advisor_forbidden_output_pattern <- paste(c(
  "sarcopenia", "cachexia", "muscle wasting",
  "diagnos", "prescription", "prescribe", "drug", "dose", "dosage", "treat", "therapy",
  "dxa", "bia", "functional measure", "physical examination",
  "inflammation", "nutritional", "causal", "modifiable", "progression",
  "threshold 1", "threshold 2"
), collapse = "|")

Advisor_min_answer_chars <- 45

Advisor_system_rules <- "
You are the RA-LMM Risk Advisor, an interpretation agent for model-derived low muscle mass risk outputs.
Answer the current interpretation task clearly and naturally.

### Core Constraints
1. Use only the supplied logistic regression probability, risk category, numeric cutoffs, cohort summary statistics, and SHAP feature names and directions, as applicable.
2. Preserve the supplied probabilities, numeric cutoffs, risk categories, SHAP rankings, and SHAP directions.
3. Use the exact term \"low muscle mass\". Do not mention other disease labels, even in negative statements.
4. Do not provide diagnostic statements, treatment advice, medication advice, testing advice, or management instructions.
5. Do not infer biological mechanisms, causality, progression, or modifiability from model outputs.
6. Refer to the explicit numeric cutoffs when relevant. The fixed cutoffs are 0.0438 and 0.1442.
7. Use the numeric cutoffs in place of generic labels such as threshold 1 or threshold 2.
8. Do not infer or discuss patient identity or reconstruct raw clinical input values.

### SHAP Interpretation Rules
- Upward arrow (\u2191) means the feature contributed toward a higher predicted probability.
- Downward arrow (\u2193) means the feature contributed toward a lower predicted probability.
- Describe model contribution only.

### Output Discipline
- Frame risk categories as classifications under the current model cutoffs.
- Keep batch summaries descriptive. Use labels such as bimodal, notable, or unusual only if the supplied result supports them.
- Do not recommend what should be reviewed, investigated, or done next.

### Style Rules
- Use natural, direct prose in 2 to 4 sentences unless the task requires otherwise.
- Refer to the supplied probability, risk category, numeric cutoffs, counts, or SHAP drivers when relevant.

### Output Format
Return valid JSON only. The JSON object must contain exactly one string field named \"answer\".
Do not include Markdown, self-talk, chain-of-thought, or text outside the JSON object.
"

Advisor_scope_text <- "This explanation is limited to the current RA-LMM Risk Advisor output: probability, risk category, numeric cutoffs, input variables, and SHAP key drivers."

Advisor_safety_text <- "This is an auxiliary risk interpretation for research use. It is not a standalone basis for patient-level decisions."

extract_Advisor_json <- function(text) {
  starts <- gregexpr("\\{", text, perl = TRUE)[[1]]
  ends <- gregexpr("\\}", text, perl = TRUE)[[1]]
  if (starts[1] > 0 && ends[1] > 0 && tail(ends, 1) > starts[1]) {
    return(substr(text, starts[1], tail(ends, 1)))
  }
  text
}

parse_Advisor_answer <- function(raw) {
  raw_json <- extract_Advisor_json(raw)
  if (!json_validate(raw_json, Advisor_agent_schema, verbose = FALSE)) return(NULL)
  parsed <- fromJSON(raw_json)
  if (is.null(parsed$answer) || !is.character(parsed$answer)) return(NULL)
  parsed$answer
}

format_Advisor_output <- function(answer) {
  answer
}

Advisor_cache_key <- function(scope, context, question_id, question, task_type) {
  paste(scope, question_id, task_type, question, context, sep = "\r")
}

Advisor_cache_get <- function(cache_env, key) {
  if (is.null(cache_env) || !exists(key, envir = cache_env, inherits = FALSE)) {
    return(NULL)
  }
  get(key, envir = cache_env, inherits = FALSE)
}

Advisor_cache_set <- function(cache_env, key, value) {
  if (!is.null(cache_env) && !is.null(key) && nzchar(key)) {
    assign(key, value, envir = cache_env)
  }
  invisible(value)
}

format_Advisor_notes_ui <- function() {
  div(
    style = "margin-top: 16px; padding-top: 12px; border-top: 1px solid #e5e7eb; color: #5f6b7a; line-height: 1.6;",
    tags$div(tags$strong("Scope: "), Advisor_scope_text),
    tags$div(tags$strong("Safety note: "), Advisor_safety_text)
  )
}

Advisor_task_type <- function(question_id) {
  task_map <- list(
    single_overview = "risk summary",
    batch_overview = "cohort summary",
    single_risk_explanation = "risk clarification",
    single_shap_drivers = "SHAP interpretation",
    single_thresholds = "cutoff explanation",
    single_monitoring_focus = "follow-up focus",
    batch_distribution = "cohort distribution",
    batch_mean_probability = "cohort mean probability",
    batch_monitoring_focus = "cohort follow-up focus",
    batch_limitations = "limitations review"
  )
  if (!is.null(task_map[[question_id]])) task_map[[question_id]] else "interpretation"
}

Advisor_answer_check <- function(answer) {
  text <- trimws(answer)

  if (!nzchar(text)) {
    return(list(ok = FALSE, reason = "empty answer"))
  }

  if (nchar(text) < Advisor_min_answer_chars) {
    return(list(ok = FALSE, reason = "answer too short"))
  }

  if (grepl(Advisor_forbidden_output_pattern, tolower(text), perl = TRUE)) {
    return(list(ok = FALSE, reason = "answer hit forbidden pattern"))
  }

  list(ok = TRUE, reason = "ok")
}

Advisor_build_prompt <- function(context, question_id, question, task_type, retry_reason = NULL) {
  retry_block <- ""
  if (!is.null(retry_reason)) {
    retry_block <- paste(
      "Retry instruction:",
      sprintf("Your previous answer could not be used because %s.", retry_reason),
      "Rewrite it in a cleaner and more natural way while staying within policy.",
      "Keep the wording specific to the current task and avoid sounding formulaic.",
      sep = "\n"
    )
  }

  paste(
    context,
    "\n\nTask:", task_type,
    "\nQuestion:", question,
    "\nRule:", Advisor_question_rules(question_id),
    if (nzchar(retry_block)) paste0("\n", retry_block) else "",
    "\n\nReturn concise JSON with only the answer field."
  )
}

Advisor_request_once <- function(context, question_id, question, task_type, api_key, retry_reason = NULL) {
  if (!requireNamespace("ellmer", quietly = TRUE)) stop("ellmer is not installed")

  chat <- ellmer::chat_deepseek(
    credentials = function() api_key,
    model = "deepseek-v4-pro",
    system_prompt = Advisor_system_rules,
    api_args = list(
      temperature = 0.2,
      response_format = list(type = "json_object")
    )
  )

  raw <- chat$chat(
    Advisor_build_prompt(
      context = context,
      question_id = question_id,
      question = question,
      task_type = task_type,
      retry_reason = retry_reason
    ),
    echo = "none"
  )

  answer <- parse_Advisor_answer(raw)
  if (is.null(answer)) {
    return(list(ok = FALSE, reason = "non-schema output", answer = NULL))
  }

  check <- Advisor_answer_check(answer)
  list(ok = check$ok, reason = check$reason, answer = answer)
}

Advisor_request_with_retry <- function(context, question_id, question, task_type, api_key, log_prefix) {
  log_info("{log_prefix} attempt 1 started: question_id={question_id}, task_type={task_type}")
  first_attempt <- Advisor_request_once(
    context = context,
    question_id = question_id,
    question = question,
    task_type = task_type,
    api_key = api_key
  )

  if (isTRUE(first_attempt$ok)) {
    return(first_attempt)
  }

  log_warn("{log_prefix} first pass unusable: {first_attempt$reason}")

  log_info("{log_prefix} attempt 2 started: question_id={question_id}, task_type={task_type}, retry_reason={first_attempt$reason}")
  second_attempt <- Advisor_request_once(
    context = context,
    question_id = question_id,
    question = question,
    task_type = task_type,
    api_key = api_key,
    retry_reason = first_attempt$reason
  )

  if (isTRUE(second_attempt$ok)) {
    log_info("{log_prefix} recovered after soft retry")
    return(second_attempt)
  }

  log_warn("{log_prefix} fallback triggered after retry: {second_attempt$reason}")
  list(ok = FALSE, reason = second_attempt$reason, answer = NULL)
}

safe_Advisor_drivers_text <- function(result) {
  key_drivers <- result$key_drivers
  drivers <- c(key_drivers$driver1[1], key_drivers$driver2[1], key_drivers$driver3[1])
  drivers <- drivers[!is.na(drivers) & nzchar(drivers)]
  if (length(drivers) == 0) return("No SHAP key drivers are available for this result.")
  paste(drivers, collapse = "; ")
}

Advisor_prompt_drivers_text <- function(key_drivers) {
  drivers <- c(key_drivers$driver1[1], key_drivers$driver2[1], key_drivers$driver3[1])
  drivers <- drivers[!is.na(drivers) & nzchar(drivers)]
  if (length(drivers) == 0) return("None")

  drivers <- vapply(drivers, function(driver) {
    feature <- sub("[=( ].*$", "", driver)
    direction <- if (grepl("↑", driver, fixed = TRUE)) "↑" else if (grepl("↓", driver, fixed = TRUE)) "↓" else ""
    trimws(paste(feature, direction))
  }, character(1))

  paste(drivers, collapse = "; ")
}

Advisor_fallback_answer <- function(question_id, result = NULL, summary_stats = NULL) {
  if (!is.null(result)) {
    probability <- sprintf("%.1f%%", result$probability * 100)
    thresholds <- result$thresholds
    interval <- sprintf(
      "Low: P < %.4f; Intermediate: %.4f <= P < %.4f; High: P >= %.4f",
      thresholds[1], thresholds[1], thresholds[2], thresholds[2]
    )
    drivers <- safe_Advisor_drivers_text(result)

    if (question_id == "single_risk_explanation") {
      return(sprintf(
        "The current model-derived probability is %s, which places this result in the %s risk category under the current numeric cutoffs (%s).",
        probability, result$risk_level, interval
      ))
    }
    if (question_id == "single_shap_drivers") {
      return(sprintf(
        "The strongest SHAP key drivers are %s. An upward arrow means that the feature value pushed the model-derived probability higher, while a downward arrow means that it pushed the probability lower.",
        drivers
      ))
    }
    if (question_id == "single_thresholds") {
      return(sprintf(
        "The risk category is determined by comparing the current model-derived probability (%s) with the fixed numeric cutoffs: %s.",
        probability, interval
      ))
    }
    if (question_id == "single_monitoring_focus") {
      return(sprintf(
        "For this result, the most relevant follow-up interpretation focus is the displayed SHAP driver set: %s. These variables made the largest model-derived contributions in the current prediction. Upward arrows indicate higher model-derived probability, and downward arrows indicate lower model-derived probability.",
        drivers
      ))
    }
  }

  if (!is.null(summary_stats)) {
    thresholds <- summary_stats$thresholds
    interval <- sprintf(
      "Low: P < %.4f; Intermediate: %.4f <= P < %.4f; High: P >= %.4f",
      thresholds[1], thresholds[1], thresholds[2], thresholds[2]
    )
    distribution <- sprintf(
      "High=%d (%.1f%%), Intermediate=%d (%.1f%%), Low=%d (%.1f%%)",
      summary_stats$high_risk, (summary_stats$high_risk / summary_stats$total_patients) * 100,
      summary_stats$intermediate_risk, (summary_stats$intermediate_risk / summary_stats$total_patients) * 100,
      summary_stats$low_risk, (summary_stats$low_risk / summary_stats$total_patients) * 100
    )

    if (question_id == "batch_distribution") {
      return(sprintf(
        "This batch contains %d patients, with the current model-derived risk distribution summarized as %s.",
        summary_stats$total_patients, distribution
      ))
    }
    if (question_id == "batch_mean_probability") {
      return(sprintf(
        "The mean model-derived probability across this batch is %.1f%%. This cohort-level average is most useful when read together with the overall risk distribution rather than as an individual-level result.",
        summary_stats$mean_probability * 100
      ))
    }
    if (question_id == "batch_monitoring_focus") {
      return(sprintf(
        "For cohort-level follow-up interpretation, the High and Intermediate groups deserve the closest attention because they are nearest to, or already beyond, the higher cutoff range. The current batch distribution is %s.",
        distribution
      ))
    }
    if (question_id == "batch_limitations") {
      return(sprintf(
        "This batch summary is bounded by the cohort composition and the fixed numeric cutoffs used in the current model setup (%s). It describes the uploaded dataset only and should not be generalized beyond this batch without further validation.",
        interval
      ))
    }
  }

  "This response is limited to the current RA-LMM Risk Advisor output and the current interpretation scope."
}

Agent_task_catalog <- list(
  single_overview = list(
    mode = "single",
    task_rule = "Provide a natural summary of the current patient-level result using the probability, risk category, fixed cutoffs, and SHAP key drivers. Make clear that the category is defined under the current cutoffs. Describe SHAP as model-derived contribution only."
  ),
  batch_overview = list(
    mode = "batch",
    task_rule = "Provide a natural summary of the current cohort-level result using the mean probability, risk distribution, and explicit numeric cutoffs 0.0438 and 0.1442. Stay descriptive and result-centered. Do not use threshold 1 or threshold 2 wording. Do not describe the distribution as bimodal, notable, unusual, or action-guiding unless directly supported by the displayed result."
  ),
  single_risk_explanation = list(
    mode = "single",
    task_rule = "Explain only why the probability falls into the displayed risk category. Do not mention SHAP unless needed.",
    followup_prompt = "Explain this patient's risk level using only the current probability and the fixed cutoffs 0.0438 and 0.1442."
  ),
  single_shap_drivers = list(
    mode = "single",
    task_rule = "Explain only the SHAP arrow directions and listed key drivers. Do not provide clinical mechanisms or biological explanations.",
    followup_prompt = "Explain the SHAP key drivers and arrow directions as model contribution directions only."
  ),
  single_thresholds = list(
    mode = "single",
    task_rule = "Explain only the fixed numeric cutoff intervals and how the probability maps to the category. If useful, mention the 0.0438 cutoff and the 0.1442 cutoff.",
    followup_prompt = "Explain how the fixed cutoffs 0.0438 and 0.1442 define this patient's risk category."
  ),
  single_monitoring_focus = list(
    mode = "single",
    task_rule = "Identify the displayed model variables that should be emphasized in follow-up interpretation. Write a complete explanation using the SHAP key drivers and arrow directions. Do not return only a variable list. Do not recommend tests, treatment, or clinical actions.",
    followup_prompt = "Identify which model variables may deserve attention in follow-up interpretation, based only on this result."
  ),
  batch_distribution = list(
    mode = "batch",
    task_rule = "Explain only the counts and percentages in the cohort risk distribution.",
    followup_prompt = "Explain the cohort risk distribution in the current RA-LMM Risk Advisor batch result."
  ),
  batch_mean_probability = list(
    mode = "batch",
    task_rule = "Explain only the cohort mean probability and why it differs from individual probabilities.",
    followup_prompt = "Explain the mean predicted probability in the current RA-LMM Risk Advisor batch result."
  ),
  batch_monitoring_focus = list(
    mode = "batch",
    task_rule = "Identify which risk groups should be emphasized in cohort-level follow-up interpretation. Write a complete explanation using only the current risk distribution and numeric cutoffs. Do not recommend tests, treatment, or clinical actions.",
    followup_prompt = "Identify which risk groups may deserve attention in cohort-level follow-up interpretation, based only on this batch result."
  ),
  batch_limitations = list(
    mode = "batch",
    task_rule = "Explain only limitations of interpreting this batch model output, including numeric cutoffs, cohort composition, and probabilistic output. Do not mention tests or clinical management.",
    followup_prompt = "Explain the limitations of interpreting this RA-LMM Risk Advisor batch result."
  )
)

Agent_task_value <- function(question_id, field, default = NULL) {
  task <- Agent_task_catalog[[question_id]]
  if (is.null(task) || is.null(task[[field]])) return(default)
  task[[field]]
}

Advisor_question_rules <- function(question_id) {
  Agent_task_value(question_id, "task_rule", "Answer within the RA-LMM Risk Advisor scope only.")
}

Discussion_prompt_registry <- list(
  single = list(
    explain = "Can you explain this active single-patient result in a clinician-facing way?",
    summarize = "Can you summarize this case using the current probability, risk category, cutoffs, and SHAP drivers?",
    limitations = "What are the main limitations of interpreting this single-patient result too strongly?",
    focus = "Which displayed factors stand out most in this case, and why?"
  ),
  batch = list(
    explain = "Can you explain this active batch result in a clinician-facing way?",
    summarize = "Can you summarize what stands out in this cohort using the current mean probability and risk distribution?",
    limitations = "What are the main limitations of reading this batch result in a strong way?",
    focus = "Which cohort-level patterns deserve the most attention in this active batch result?"
  ),
  none = list(
    explain = "Run a single or batch prediction to start discussion.",
    summarize = "Run a single or batch prediction to start discussion.",
    limitations = "Run a single or batch prediction to start discussion.",
    focus = "Run a single or batch prediction to start discussion."
  )
)

Agent_tool_descriptions <- list(
  explain_risk_level = "Use the active result probability, category, and fixed cutoffs to explain the displayed risk level.",
  explain_cutoffs = "Use the fixed numeric cutoffs to explain category mapping without changing thresholds.",
  explain_shap_drivers = "Use only the displayed SHAP key drivers and arrow directions.",
  explain_limitations = "Stay within the current result scope and explain interpretation limits.",
  explain_batch_distribution = "Use only the current cohort counts and percentages.",
  explain_batch_probability = "Use only the current cohort-level mean probability and its interpretation."
)

Discussion_intent_registry <- list(
  single = list(
    list(
      intent_id = "single_shap_drivers",
      task_type = "SHAP interpretation",
      tool_name = "explain_shap_drivers",
      question_rule_id = "single_shap_drivers",
      pattern = "shap|driver|drivers|arrow|arrows|feature|features|contribution|contributions"
    ),
    list(
      intent_id = "single_thresholds",
      task_type = "cutoff explanation",
      tool_name = "explain_cutoffs",
      question_rule_id = "single_thresholds",
      pattern = "cutoff|cutoffs|threshold|thresholds|category|categories|risk level|why high|why intermediate|why low"
    ),
    list(
      intent_id = "single_limitations",
      task_type = "limitations review",
      tool_name = "explain_limitations",
      instruction = "Explain only the interpretation limits of the active single-patient result using the current probability, category, cutoffs, and SHAP drivers.",
      pattern = "limit|limits|limitation|limitations|caution|overread|too strong|uncertain|uncertainty|scope"
    ),
    list(
      intent_id = "single_monitoring_focus",
      task_type = "follow-up focus",
      tool_name = "explain_shap_drivers",
      question_rule_id = "single_monitoring_focus",
      pattern = "focus|stand out|stands out|important|attention|follow-up|follow up|emphasize|emphasis"
    ),
    list(
      intent_id = "single_overview",
      task_type = "risk summary",
      tool_name = "explain_risk_level",
      question_rule_id = "single_overview",
      pattern = "summary|summarize|overview|recap"
    )
  ),
  batch = list(
    list(
      intent_id = "batch_distribution",
      task_type = "cohort distribution",
      tool_name = "explain_batch_distribution",
      question_rule_id = "batch_distribution",
      pattern = "distribution|distribute|count|counts|percentage|percentages|proportion|proportions"
    ),
    list(
      intent_id = "batch_mean_probability",
      task_type = "cohort mean probability",
      tool_name = "explain_batch_probability",
      question_rule_id = "batch_mean_probability",
      pattern = "mean|average|probability|probabilities"
    ),
    list(
      intent_id = "batch_limitations",
      task_type = "limitations review",
      tool_name = "explain_limitations",
      question_rule_id = "batch_limitations",
      pattern = "limit|limits|limitation|limitations|caution|overread|too strong|uncertain|uncertainty|scope"
    ),
    list(
      intent_id = "batch_monitoring_focus",
      task_type = "cohort follow-up focus",
      tool_name = "explain_batch_distribution",
      question_rule_id = "batch_monitoring_focus",
      pattern = "focus|stand out|stands out|important|attention|follow-up|follow up|group|groups|emphasize|emphasis"
    ),
    list(
      intent_id = "batch_overview",
      task_type = "cohort summary",
      tool_name = "explain_batch_distribution",
      question_rule_id = "batch_overview",
      pattern = "summary|summarize|overview|recap"
    )
  )
)

Discussion_default_routes <- list(
  single = list(
    intent_id = "single_risk_explanation",
    task_type = "risk clarification",
    tool_name = "explain_risk_level",
    question_rule_id = "single_risk_explanation"
  ),
  batch = list(
    intent_id = "batch_overview",
    task_type = "cohort summary",
    tool_name = "explain_batch_distribution",
    question_rule_id = "batch_overview"
  ),
  none = list(
    intent_id = "general",
    task_type = "general discussion",
    tool_name = "explain_limitations",
    instruction = "Answer only within the current RA-LMM result context."
  )
)

Agent_get_followup_prompt <- function(mode, question_id) {
  prompt <- Agent_task_value(question_id, "followup_prompt", "")
  task_mode <- Agent_task_value(question_id, "mode", NULL)
  if (identical(mode, task_mode)) return(prompt)
  ""
}

Agent_get_discussion_prompt <- function(mode, prompt_type) {
  prompts <- Discussion_prompt_registry[[mode]]
  if (!is.null(prompts[[prompt_type]])) return(prompts[[prompt_type]])
  Discussion_prompt_registry$none[[prompt_type]]
}

Agent_mode_tools <- function(mode) {
  base_tools <- switch(
    mode,
    single = c("predict_single_patient", "get_single_result_context"),
    batch = c("predict_batch_patients", "get_batch_result_context"),
    character()
  )
  registry <- Discussion_intent_registry[[mode]]
  unique(c(base_tools, vapply(if (is.null(registry)) list() else registry, `[[`, character(1), "tool_name")))
}

Agent_active_tools_text <- function(mode) {
  tools <- Agent_mode_tools(mode)
  if (is.null(tools) || length(tools) == 0) return("No active tools.")
  paste(tools, collapse = ", ")
}

Agent_context_from_mode <- function(mode, single_result = NULL, batch_summary = NULL) {
  if (identical(mode, "single") && !is.null(single_result)) {
    return(format_Advisor_context(single_result))
  }
  if (identical(mode, "batch") && !is.null(batch_summary)) {
    return(format_Advisor_batch_context(batch_summary))
  }
  "No active RA-LMM result is currently available for discussion."
}

Discussion_finalize_route <- function(route) {
  if (is.null(route$instruction) && !is.null(route$question_rule_id)) {
    route$instruction <- Advisor_question_rules(route$question_rule_id)
  }
  route
}

Discussion_match_route <- function(mode, question) {
  registry <- Discussion_intent_registry[[mode]]
  if (is.null(registry) || length(registry) == 0) return(NULL)

  for (route in registry) {
    if (grepl(route$pattern, question, perl = TRUE)) {
      return(Discussion_finalize_route(route))
    }
  }

  NULL
}

Discussion_detect_intent <- function(mode, user_question) {
  question <- tolower(trimws(user_question))
  if (!nzchar(question)) {
    return(Discussion_finalize_route(Discussion_default_routes$none))
  }

  matched_route <- Discussion_match_route(mode, question)
  if (!is.null(matched_route)) {
    return(matched_route)
  }

  default_route <- Discussion_default_routes[[mode]]
  if (!is.null(default_route)) {
    return(Discussion_finalize_route(default_route))
  }

  Discussion_finalize_route(Discussion_default_routes$none)
}

format_Advisor_context <- function(result) {
  driver_text <- Advisor_prompt_drivers_text(result$key_drivers)

  sprintf(
    "Current RA-LMM Risk Advisor result:
Logistic regression predicted probability of low muscle mass: %.1f%%.
Risk category: %s.
Risk cutoffs: Low: P < %.4f; Intermediate: %.4f <= P < %.4f; High: P >= %.4f.
SHAP key drivers (feature and direction only): %s.",
    result$probability * 100,
    result$risk_level,
    result$thresholds[1],
    result$thresholds[1],
    result$thresholds[2],
    result$thresholds[2],
    driver_text
  )
}

format_Advisor_batch_context <- function(summary_stats) {
  sprintf(
    "Current RA-LMM Risk Advisor batch result:
Total patients: %d.
Mean predicted probability of low muscle mass: %.1f%%.
Risk distribution: High=%d (%.1f%%); Intermediate=%d (%.1f%%); Low=%d (%.1f%%).
Risk cutoffs: Low: P < %.4f; Intermediate: %.4f <= P < %.4f; High: P >= %.4f.",
    summary_stats$total_patients,
    summary_stats$mean_probability * 100,
    summary_stats$high_risk,
    (summary_stats$high_risk / summary_stats$total_patients) * 100,
    summary_stats$intermediate_risk,
    (summary_stats$intermediate_risk / summary_stats$total_patients) * 100,
    summary_stats$low_risk,
    (summary_stats$low_risk / summary_stats$total_patients) * 100,
    summary_stats$thresholds[1],
    summary_stats$thresholds[1],
    summary_stats$thresholds[2],
    summary_stats$thresholds[2]
  )
}

generate_Advisor_followup <- function(question_id, question, result, api_key, cache_env = NULL) {
  cache_key <- NULL
  start_time <- Sys.time()
  tryCatch({
    context <- format_Advisor_context(result)
    task_type <- Advisor_task_type(question_id)
    cache_key <- Advisor_cache_key("single_followup", context, question_id, question, task_type)
    cached_response <- Advisor_cache_get(cache_env, cache_key)

    if (!is.null(cached_response)) {
      log_info(
        "Advisor cache hit: question_id={question_id}, task_type={task_type}, risk_level={result$risk_level}, probability={result$probability}"
      )
      return(format_Advisor_output(cached_response))
    }

    log_info(
      "Advisor request started: question_id={question_id}, task_type={task_type}, risk_level={result$risk_level}, probability={result$probability}"
    )
    result_obj <- Advisor_request_with_retry(
      context = context,
      question_id = question_id,
      question = question,
      task_type = task_type,
      api_key = api_key,
      log_prefix = "RA-LMM Risk Advisor"
    )
    elapsed_ms <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000)
    log_info("Advisor request completed: question_id={question_id}, task_type={task_type}, latency_ms={elapsed_ms}")
    if (!isTRUE(result_obj$ok)) {
      log_warn("RA-LMM Risk Advisor response replaced by deterministic fallback")
      fallback_text <- format_Advisor_output(Advisor_fallback_answer(question_id, result = result))
      Advisor_cache_set(cache_env, cache_key, fallback_text)
      return(fallback_text)
    }
    answer_text <- format_Advisor_output(result_obj$answer)
    Advisor_cache_set(cache_env, cache_key, answer_text)
    answer_text
  }, error = function(e) {
    elapsed_ms <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000)
    log_error("RA-LMM Risk Advisor follow-up failed: question_id={question_id}, error={e$message}")
    log_warn("Advisor request finished with error: question_id={question_id}, latency_ms={elapsed_ms}")
    fallback_text <- format_Advisor_output(Advisor_fallback_answer(question_id, result = result))
    Advisor_cache_set(cache_env, cache_key, fallback_text)
    return(fallback_text)
  })
}

generate_Advisor_batch_followup <- function(question_id, question, summary_stats, api_key, cache_env = NULL) {
  cache_key <- NULL
  start_time <- Sys.time()
  tryCatch({
    context <- format_Advisor_batch_context(summary_stats)
    task_type <- Advisor_task_type(question_id)
    cache_key <- Advisor_cache_key("batch_followup", context, question_id, question, task_type)
    cached_response <- Advisor_cache_get(cache_env, cache_key)

    if (!is.null(cached_response)) {
      log_info(
        "Advisor batch cache hit: question_id={question_id}, task_type={task_type}, total_patients={summary_stats$total_patients}, mean_probability={summary_stats$mean_probability}"
      )
      return(format_Advisor_output(cached_response))
    }

    log_info(
      "Advisor batch request started: question_id={question_id}, task_type={task_type}, total_patients={summary_stats$total_patients}, mean_probability={summary_stats$mean_probability}"
    )
    result_obj <- Advisor_request_with_retry(
      context = context,
      question_id = question_id,
      question = question,
      task_type = task_type,
      api_key = api_key,
      log_prefix = "RA-LMM Risk Advisor batch"
    )
    elapsed_ms <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000)
    log_info("Advisor batch request completed: question_id={question_id}, task_type={task_type}, latency_ms={elapsed_ms}")
    if (!isTRUE(result_obj$ok)) {
      log_warn("RA-LMM Risk Advisor batch response replaced by deterministic fallback")
      fallback_text <- format_Advisor_output(Advisor_fallback_answer(question_id, summary_stats = summary_stats))
      Advisor_cache_set(cache_env, cache_key, fallback_text)
      return(fallback_text)
    }
    answer_text <- format_Advisor_output(result_obj$answer)
    Advisor_cache_set(cache_env, cache_key, answer_text)
    answer_text
  }, error = function(e) {
    elapsed_ms <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000)
    log_error("RA-LMM Risk Advisor batch follow-up failed: question_id={question_id}, error={e$message}")
    log_warn("Advisor batch request finished with error: question_id={question_id}, latency_ms={elapsed_ms}")
    fallback_text <- format_Advisor_output(Advisor_fallback_answer(question_id, summary_stats = summary_stats))
    Advisor_cache_set(cache_env, cache_key, fallback_text)
    return(fallback_text)
  })
}

resolve_discussion_mode <- function(single_result = NULL, batch_summary = NULL, preferred_mode = NULL) {
  if (!is.null(preferred_mode) && identical(preferred_mode, "single") && !is.null(single_result)) return("single")
  if (!is.null(preferred_mode) && identical(preferred_mode, "batch") && !is.null(batch_summary)) return("batch")
  if (!is.null(single_result)) return("single")
  if (!is.null(batch_summary)) return("batch")
  "none"
}

format_discussion_history_text <- function(history, max_messages = 8) {
  if (length(history) == 0) return("No prior discussion turns.")
  history_subset <- tail(history, max_messages)
  paste(vapply(history_subset, function(msg) {
    sprintf("%s: %s", toupper(msg$role), msg$content)
  }, character(1)), collapse = "\n")
}

Discussion_system_rules <- "
You are the RA-LMM Clinical Discussion Agent.
Support clinician-facing discussion of the supplied RA-LMM logistic regression result.

### Scope
1. Follow the supplied task and stay within the active RA-LMM result.
2. In single mode, use only the supplied probability, risk category, fixed cutoffs, and SHAP feature names and directions.
3. In batch mode, use only the supplied cohort size, mean probability, risk distribution, and fixed cutoffs.
4. Redirect out-of-scope requests to the active RA-LMM result.

### Language
Reply in the language of the supplied task.

### Output Discipline
1. Preserve the supplied numbers, risk categories, cutoffs, SHAP rankings, and directions.
2. Frame risk categories as classifications under the current model cutoffs.
3. Describe SHAP as model-derived contribution to a higher or lower predicted probability.
4. Do not describe SHAP contributions as biological mechanisms, causal effects, or clinical determinants.
5. Describe model probabilities as estimates and keep the wording descriptive.

### Safety Boundaries
1. Do not provide diagnosis, treatment, medication, dosing, testing, or management recommendations.
2. Do not infer causality, pathophysiology, progression, modifiability, or benefit from intervention.
3. Do not recommend what should be done next in clinical care or propose actions based on batch results.
4. Use the exact term \"low muscle mass\".
5. Use the explicit numeric cutoffs 0.0438 and 0.1442 when relevant and avoid generic labels such as threshold 1 or threshold 2.
6. Do not infer or discuss patient identity or reconstruct raw clinical input values.

### Style
Use 2 to 5 concise, clinically clear sentences specific to the active result and supplied task.

### Output Format
Return valid JSON only. The JSON object must contain exactly one string field named \"answer\".
Do not include Markdown, self-talk, chain-of-thought, or text outside the JSON object.
"

Discussion_answer_check <- function(answer) {
  text <- trimws(answer)
  if (!nzchar(text)) {
    return(list(ok = FALSE, reason = "empty answer"))
  }
  if (nchar(text) < 30) {
    return(list(ok = FALSE, reason = "answer too short"))
  }
  list(ok = TRUE, reason = "ok")
}

Discussion_build_prompt <- function(mode, context, history_text, user_question, retry_reason = NULL) {
  route <- Discussion_detect_intent(mode, user_question)
  retry_block <- ""
  if (!is.null(retry_reason)) {
    retry_block <- paste(
      "Retry instruction:",
      sprintf("Your previous answer could not be used because %s.", retry_reason),
      "Rewrite it in a cleaner and more natural discussion style while staying within scope.",
      sep = "\n"
    )
  }

  paste(
    "Active discussion mode:", mode,
    "\nTask:", route$task_type,
    "\nRule:", route$instruction,
    "\n\nResult:\n", context,
    if (nzchar(retry_block)) paste0("\n\n", retry_block) else "",
    "\n\nReturn concise JSON with only the answer field."
  )
}

Discussion_request_once <- function(mode, context, history_text, user_question, api_key, retry_reason = NULL) {
  if (!requireNamespace("ellmer", quietly = TRUE)) stop("ellmer is not installed")

  chat <- ellmer::chat_deepseek(
    credentials = function() api_key,
    model = "deepseek-v4-pro",
    system_prompt = Discussion_system_rules,
    api_args = list(
      temperature = 0.35,
      response_format = list(type = "json_object")
    )
  )

  raw <- chat$chat(
    Discussion_build_prompt(
      mode = mode,
      context = context,
      history_text = history_text,
      user_question = user_question,
      retry_reason = retry_reason
    ),
    echo = "none"
  )

  answer <- parse_Advisor_answer(raw)
  if (is.null(answer)) {
    return(list(ok = FALSE, reason = "non-schema output", answer = NULL))
  }

  check <- Discussion_answer_check(answer)
  list(ok = check$ok, reason = check$reason, answer = answer)
}

Discussion_request_with_retry <- function(mode, context, history_text, user_question, api_key, log_prefix) {
  log_info("{log_prefix} attempt 1 started: mode={mode}")
  first_attempt <- Discussion_request_once(
    mode = mode,
    context = context,
    history_text = history_text,
    user_question = user_question,
    api_key = api_key
  )

  if (isTRUE(first_attempt$ok)) {
    return(first_attempt)
  }

  log_warn("{log_prefix} first pass unusable: {first_attempt$reason}")
  log_info("{log_prefix} attempt 2 started: mode={mode}, retry_reason={first_attempt$reason}")

  second_attempt <- Discussion_request_once(
    mode = mode,
    context = context,
    history_text = history_text,
    user_question = user_question,
    api_key = api_key,
    retry_reason = first_attempt$reason
  )

  if (isTRUE(second_attempt$ok)) {
    log_info("{log_prefix} recovered after soft retry")
    return(second_attempt)
  }

  log_warn("{log_prefix} fallback triggered after retry: {second_attempt$reason}")
  list(ok = FALSE, reason = second_attempt$reason, answer = NULL)
}

Discussion_fallback_answer <- function(mode, user_question, result = NULL, summary_stats = NULL) {
  route <- Discussion_detect_intent(mode, user_question)
  if (identical(mode, "single") && !is.null(result)) {
    return(sprintf(
      "I can discuss this active single-patient result using the current probability (%.1f%%), risk category (%s), the numeric cutoffs 0.0438 and 0.1442, and the displayed SHAP drivers (%s). The current question was routed to %s. Please rephrase within that result-centered scope if you want a more specific interpretation.",
      result$probability * 100,
      result$risk_level,
      safe_Advisor_drivers_text(result),
      route$task_type
    ))
  }

  if (identical(mode, "batch") && !is.null(summary_stats)) {
    distribution <- sprintf(
      "High=%d (%.1f%%), Intermediate=%d (%.1f%%), Low=%d (%.1f%%)",
      summary_stats$high_risk, (summary_stats$high_risk / summary_stats$total_patients) * 100,
      summary_stats$intermediate_risk, (summary_stats$intermediate_risk / summary_stats$total_patients) * 100,
      summary_stats$low_risk, (summary_stats$low_risk / summary_stats$total_patients) * 100
    )
    return(sprintf(
      "I can discuss this active batch result using the current cohort summary: total patients=%d, mean probability=%.1f%%, and risk distribution=%s. The current question was routed to %s. Please rephrase the question so I can stay centered on this RA-LMM cohort result.",
      summary_stats$total_patients,
      summary_stats$mean_probability * 100,
      distribution,
      route$task_type
    ))
  }

  "Run a single or batch prediction to start a result-centered clinical discussion."
}

ui <- navbarPage(
  title = div(
    tags$img(src = "https://cdn-icons-png.flaticon.com/512/2919/2919600.png",
             height = 30, style = "margin-right: 10px;"),
    "RA-LMM Risk Advisor"
  ),
  theme = shinytheme("flatly"),
  header = tags$head(
    tags$title("RA-LMM Risk Advisor"),
    tags$style(HTML("
      /* UI Custom Styles */
      .risk-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 6px solid; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
      .low-risk { background-color: #eafaf1; border-left-color: #2ecc71; color: #27ae60; }
      .intermediate-risk { background-color: #fef9e7; border-left-color: #f1c40f; color: #d4ac0d; }
      .high-risk { background-color: #fdf2e9; border-left-color: #e67e22; color: #d35400; }
      .metric-card { background: #fff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 6px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
      .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
      .metric-label { font-size: 11px; text-transform: uppercase; color: #7f8c8d; margin-top: 5px; }
      .batch-summary-box { background: #fdfefe; border: 1px solid #d6dbdf; border-radius: 6px; padding: 15px; text-align: center; margin-bottom: 15px; }
      .summary-value { font-size: 22px; font-weight: bold; }
      .summary-label { font-size: 12px; color: #7f8c8d; }
      .risk-badge { padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; color: #fff; display: inline-block; }
      .badge-low { background-color: #2ecc71; }
      .badge-intermediate { background-color: #f1c40f; }
      .badge-high { background-color: #e67e22; }
      .driver-impact-up { color: #e74c3c; font-weight: bold; }
      .driver-impact-down { color: #2ecc71; font-weight: bold; }
      .discussion-shell { background: #ffffff; border: 1px solid #dfe6e9; border-radius: 6px; padding: 18px; }
      .discussion-history { min-height: 360px; max-height: 520px; overflow-y: auto; background: #f8f9fa; border: 1px solid #dfe6e9; border-radius: 6px; padding: 16px; }
      .discussion-empty { color: #7f8c8d; line-height: 1.7; padding: 36px 12px; text-align: center; }
      .discussion-message { margin-bottom: 14px; }
      .discussion-message:last-child { margin-bottom: 0; }
      .discussion-bubble { display: inline-block; max-width: 88%; padding: 12px 14px; border-radius: 6px; line-height: 1.65; }
      .discussion-message.user { text-align: right; }
      .discussion-message.user .discussion-bubble { background: #eaf2f8; color: #1f3a56; }
      .discussion-message.assistant .discussion-bubble { background: #ffffff; color: #2c3e50; border: 1px solid #dfe6e9; }
      .discussion-meta { font-size: 11px; color: #7f8c8d; margin-bottom: 4px; }
      .discussion-context-pill { display: inline-block; padding: 5px 10px; border-radius: 999px; font-size: 12px; font-weight: bold; color: #fff; }
      .discussion-context-single { background: #2980b9; }
      .discussion-context-batch { background: #27ae60; }
      .discussion-context-none { background: #7f8c8d; }
      .discussion-summary-row { margin-bottom: 10px; }
      .discussion-summary-label { font-weight: bold; color: #2c3e50; }
      .discussion-note { color: #5f6b7a; line-height: 1.6; }
    "))
  ),

  tabPanel(
    "Single Patient Assessment",
    icon = icon("user-md"),
    fluidRow(
      column(
        3,
        wellPanel(
          h4(icon("sliders-h"), "Clinical and Hematological Indicators", style = "color: #2c3e50;"),
          hr(style = "border-color: #bdc3c7;"),
          numericInputIcon("ANC", label = tags$span(icon("vial"), "ANC (10^9/L)"), value = 3.5, min = 0, max = 50, step = 0.1, icon = icon("temperature-low")),
          numericInputIcon("ALT", label = tags$span(icon("vial"), "ALT (U/L)"), value = 25, min = 0, max = 500, step = 1, icon = icon("chart-line")),
          numericInputIcon("Cr", label = tags$span(icon("vial"), "Cr (μmol/L)"), value = 22, min = 0, max = 500, step = 1, icon = icon("heartbeat")),
          numericInputIcon("Age", label = tags$span(icon("birthday-cake"), "Age (years)"), value = 65, min = 18, max = 120, step = 1, icon = icon("user-clock")),
          numericInputIcon("BMI", label = tags$span(icon("weight"), "BMI (kg/m^2)"), value = 22.5, min = 10, max = 50, step = 0.1, icon = icon("balance-scale")),
          pickerInput("Sex", label = tags$span(icon("venus-mars"), "Sex"), choices = list("Male" = "1", "Female" = "2"), selected = "1"),
          br(),
          actionBttn("go_one", label = "Assess Risk", icon = icon("brain"), style = "gradient", color = "primary", block = TRUE, size = "lg"),
          hr()
        )
      ),
      column(
        9,
        uiOutput("prediction_result_ui"),

        conditionalPanel(
          condition = "output.show_drivers == true",
          hr(),
          h4(icon("robot"), "RA-LMM Interpretation Agent", style="color: #2980b9;"),
          div(
            class = "well",
            style = "background-color: #eaf2f8; border-left: 5px solid #2980b9; font-size: 15px; line-height: 1.6;",
            uiOutput("ai_agent_explanation_ui")
          ),
          br(),
          h4(icon("comments"), "Ask", style="color: #2980b9;"),
          div(
            class = "well",
            style = "background-color: #f8f9fa; border-left: 5px solid #7f8c8d;",
            div(class = "btn-group-vertical", style = "width: 100%;",
                actionButton("ask_single_risk", "Explain this patient's risk level", icon = icon("circle-question"), class = "btn btn-default"),
                actionButton("ask_single_drivers", "Explain the SHAP key drivers", icon = icon("chart-line"), class = "btn btn-default"),
                actionButton("ask_single_thresholds", "Explain the risk cutoffs", icon = icon("sliders-h"), class = "btn btn-default"),
                actionButton("ask_single_monitoring", "Identify follow-up interpretation focus", icon = icon("search"), class = "btn btn-default")
            ),
            br(), br(),
            uiOutput("Advisor_followup_ui")
          ),
          br(),
          h4(icon("chart-bar"), "Key Drivers of Prediction"),
          div(
            class = "well key-drivers-table",
            DTOutput("key_drivers_table")
          ),
          br(),
          h5(icon("info-circle"), "Interpretation Guide"),
          div(
            class = "well",
            style = "background-color: #f8f9fa;",
            tags$ul(
              tags$li(tags$span(class = "driver-impact-up", "\u2191"), " indicates the feature increases low muscle mass risk"),
              tags$li(tags$span(class = "driver-impact-down", "\u2193"), " indicates the feature decreases low muscle mass risk"),
              tags$li("Driver 1: Most influential factor affecting prediction"),
              tags$li("Driver 2: Second most influential factor"),
              tags$li("Driver 3: Third most influential factor")
            )
          )
        )
      )
    )
  ),

  tabPanel(
    "Batch Assessment",
    icon = icon("users"),
    fluidRow(
      column(12, h3(icon("file-upload"), "Batch Patient Assessment"), p("Upload a CSV file containing multiple patients' clinical data for batch prediction."), hr())
    ),
    fluidRow(
      column(
        4,
        wellPanel(
          h4(icon("upload"), "Data Upload", style = "color: #2c3e50;"), hr(),
          div(
            class = "file-upload-box", style="background-color: #f8f9fa; border: 2px dashed #bdc3c7; padding: 15px; border-radius: 8px; text-align: center;",
            h5(icon("cloud-upload-alt"), "Upload CSV File"),
            tags$ul(style="text-align: left; font-size: 13px;",
                    tags$li("ANC (10^9/L)"), tags$li("ALT (U/L)"), tags$li("Cr (μmol/L)"),
                    tags$li("Sex (1=Male, 2=Female)"), tags$li("Age (years)"), tags$li("BMI (kg/m^2)")
            ),
            fileInput("batch_file", label = NULL, accept = c(".csv", ".txt"), buttonLabel = "Browse...", placeholder = "No file selected"),
            downloadLink("download_template", "Download template: batch_template.csv")
          ),
          hr(),
          h5(icon("cog"), "Processing Options"),
          awesomeCheckbox("include_drivers", "Include Key Drivers Analysis", value = TRUE, status = "primary"),
          awesomeCheckbox("include_descriptive", "Include Descriptive Statistics", value = TRUE, status = "primary"),
          br(),
          actionBttn("go_batch", label = "Start Batch Prediction", icon = icon("play-circle"), style = "gradient", color = "success", block = TRUE, size = "lg"),
          br(),
          conditionalPanel(condition = "output.batch_result_available == true", downloadBttn("download_results", "Download Results", style = "gradient", color = "primary", block = TRUE))
        )
      ),
      column(
        8,
        uiOutput("batch_processing_ui"),
        conditionalPanel(
          condition = "output.batch_result_available == true",
          h4(icon("chart-pie"), "Batch Summary"),
          uiOutput("batch_summary_ui"),
          br(),
          h4(icon("robot"), "DeepSeek Interpretation Agent", style = "color: #27ae60;"),
          div(
            class = "well",
            style = "background-color: #ebf5fb; border-left: 5px solid #27ae60; font-size: 15px; line-height: 1.6;",
            uiOutput("batch_ai_summary_ui")
          ),
          br(),
          h4(icon("comments"), "Ask", style = "color: #27ae60;"),
          div(
            class = "well",
            style = "background-color: #f8f9fa; border-left: 5px solid #7f8c8d;",
            div(class = "btn-group-vertical", style = "width: 100%;",
                actionButton("ask_batch_distribution", "Explain the cohort risk distribution", icon = icon("chart-pie"), class = "btn btn-default"),
                actionButton("ask_batch_probability", "Explain the mean probability", icon = icon("percent"), class = "btn btn-default"),
                actionButton("ask_batch_monitoring", "Identify cohort follow-up focus", icon = icon("search"), class = "btn btn-default"),
                actionButton("ask_batch_limitations", "Explain limitations of this batch result", icon = icon("triangle-exclamation"), class = "btn btn-default")
            ),
            br(), br(),
            uiOutput("Advisor_batch_followup_ui")
          ),
          hr(),
          h4(icon("table"), "Detailed Predictions"),
          div(class = "well", style = "padding: 20px;", DTOutput("batch_results_table")),
          br(),
          conditionalPanel(
            condition = "input.include_descriptive == true",
            h4(icon("chart-bar"), "Descriptive Statistics"),
            uiOutput("descriptive_stats_ui")
          )
        )
      )
    )
  ),

  tabPanel(
    "Clinical Discussion",
    icon = icon("comments"),
    fluidRow(
      column(12, h3(icon("stethoscope"), "Clinical Discussion"), p("Use the active RA-LMM result as the anchor for a more flexible clinician-facing AI discussion."), hr())
    ),
    fluidRow(
      column(
        4,
        wellPanel(
          h4(icon("crosshairs"), "Discussion Context", style = "color: #2c3e50;"),
          uiOutput("discussion_context_ui"),
          hr(),
          h5(icon("lightbulb"), "Suggested Prompts"),
          uiOutput("discussion_quick_prompts_ui"),
          br(),
          actionButton("discussion_reset", "Reset Discussion", icon = icon("redo"), class = "btn btn-default")
        )
      ),
      column(
        8,
        div(
          class = "discussion-shell",
          h4(icon("comments"), "Discussion Window", style = "color: #2c3e50; margin-top: 0;"),
          div(class = "discussion-history", uiOutput("discussion_chat_ui"))
        ),
        br(),
        uiOutput("discussion_input_panel_ui"),
        br(),
        div(
          class = "well",
          style = "background-color: #f8f9fa; border-left: 5px solid #7f8c8d;",
          div(class = "discussion-note", tags$strong("Scope: "), "Discussion is centered on the active RA-LMM result and displayed variables."),
          div(class = "discussion-note", tags$strong("Safety: "), "This discussion does not provide diagnosis, treatment, medication, or testing advice.")
        )
      )
    )
  ),

  tabPanel(
    "About",
    icon = icon("info-circle"),
    fluidRow(
      column(
        8, offset = 2,
        div(
          class = "well", style = "padding: 30px;",
          h2(icon("hospital"), "RA-LMM Risk Advisor", style = "color: #2c3e50; text-align: center;"), hr(),

          p(style = "font-size: 16px; line-height: 1.6;",
            "RA-LMM Risk Advisor is a machine learning-based tool designed specifically for patients with rheumatoid arthritis. It assesses the risk of low muscle mass by analyzing routinely available blood-based and clinical indicators. To further enhance interpretability, the system integrates the DeepSeek Large Language Model (LLM) as an interpretation agent that produces structured follow-up explanations based on model predictions and SHAP key driving factors."),

          br(),
          h4(icon("cogs"), "Technical Specifications"),
          tags$ul(
            style = "font-size: 15px; line-height: 1.8;",
            tags$li(strong("Machine Learning Model:"), " Logistic regression with SHAP interpretability"),
            tags$li(strong("AI Interpretation Agent:"), " Powered by DeepSeek LLM for structured predictive summarization and follow-up explanations"),
            tags$li(strong("Features:"), " ANC, ALT, Cr, Sex, Age, BMI")
          ),

          br(),
          h4(icon("shield-alt"), "Clinical Validation"),
          p(style = "font-size: 15px; line-height: 1.6;",
            "The model has been validated on independent test datasets and demonstrates good performance across diverse patient populations."),

          br(),
          h4(icon("handshake"), "Intended Use"),
          p(style = "font-size: 15px; line-height: 1.6;",
            "This tool is intended for clinical research and screening purposes. It should not replace clinical judgment or comprehensive assessment.")
        )
      )
    )
  )
)

server <- function(input, output, session) {

  values <- reactiveValues(
    api_key = NULL,
    single_result = NULL, key_drivers = NULL, batch_results = NULL, batch_summary = NULL, ai_text = NULL, batch_ai_text = NULL,
    Advisor_followup_text = NULL,
    Advisor_batch_followup_text = NULL,
    agent_cache = new.env(parent = emptyenv()),
    single_agent_busy = FALSE,
    batch_agent_busy = FALSE,
    discussion_mode = "none",
    discussion_history = list(),
    discussion_context = NULL,
    discussion_busy = FALSE,
    discussion_last_question = NULL,
    discussion_last_answer = NULL,
    discussion_turn_count = 0
  )

  reset_clinical_discussion <- function(clear_cache = FALSE) {
    values$discussion_history <- list()
    values$discussion_context <- NULL
    values$discussion_busy <- FALSE
    values$discussion_last_question <- NULL
    values$discussion_last_answer <- NULL
    values$discussion_turn_count <- 0
    if (isTRUE(clear_cache)) {
      values$agent_cache <- new.env(parent = emptyenv())
    }
  }

  append_discussion_message <- function(role, content) {
    values$discussion_history <- c(
      values$discussion_history,
      list(list(
        role = role,
        content = content,
        timestamp = format(Sys.time(), "%H:%M")
      ))
    )
    values$discussion_turn_count <- values$discussion_turn_count + 1
  }

  current_discussion_mode <- reactive({
    resolve_discussion_mode(
      single_result = values$single_result,
      batch_summary = values$batch_summary,
      preferred_mode = values$discussion_mode
    )
  })

  discussion_context_text <- reactive({
    Agent_context_from_mode(
      mode = current_discussion_mode(),
      single_result = values$single_result,
      batch_summary = values$batch_summary
    )
  })

  discussion_quick_question <- function(mode, prompt_type) {
    Agent_get_discussion_prompt(mode, prompt_type)
  }

  generate_clinical_discussion_reply <- function(user_question, mode) {
    if (identical(mode, "none")) {
      return(Discussion_fallback_answer(mode, user_question))
    }

    context <- discussion_context_text()
    route <- Discussion_detect_intent(mode, user_question)
    history_text <- format_discussion_history_text(values$discussion_history)
    values$discussion_context <- context
    cache_key <- Advisor_cache_key(
      "clinical_discussion",
      context,
      route$intent_id,
      user_question,
      history_text
    )
    cached_response <- Advisor_cache_get(values$agent_cache, cache_key)
    if (!is.null(cached_response)) {
      log_info("Clinical discussion cache hit: mode={mode}, intent_id={route$intent_id}, turn_count={values$discussion_turn_count}")
      return(cached_response)
    }

    start_time <- Sys.time()
    log_info("Clinical discussion request started: mode={mode}, intent_id={route$intent_id}, turn_count={values$discussion_turn_count}")
    result_obj <- tryCatch({
      Discussion_request_with_retry(
        mode = mode,
        context = context,
        history_text = history_text,
        user_question = user_question,
        api_key = values$api_key,
        log_prefix = "Clinical discussion"
      )
    }, error = function(e) {
      log_error("Clinical discussion request failed before completion: mode={mode}, error={e$message}")
      list(ok = FALSE, reason = e$message, answer = NULL)
    })

    elapsed_ms <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000)
    log_info("Clinical discussion request completed: mode={mode}, intent_id={route$intent_id}, latency_ms={elapsed_ms}")

    if (isTRUE(result_obj$ok) && !is.null(result_obj$answer)) {
      Advisor_cache_set(values$agent_cache, cache_key, result_obj$answer)
      return(result_obj$answer)
    }

    fallback_text <- if (identical(mode, "single")) {
      Discussion_fallback_answer(mode, user_question, result = values$single_result)
    } else {
      Discussion_fallback_answer(mode, user_question, summary_stats = values$batch_summary)
    }
    Advisor_cache_set(values$agent_cache, cache_key, fallback_text)
    fallback_text
  }

  submit_clinical_discussion <- function(user_question) {
    question_text <- trimws(user_question)
    mode <- current_discussion_mode()

    if (!nzchar(question_text)) {
      showNotification("Please enter a discussion question.", type = "warning", duration = 4)
      return(invisible(NULL))
    }

    if (identical(mode, "none")) {
      showNotification("Run a single or batch prediction to start discussion.", type = "warning", duration = 4)
      return(invisible(NULL))
    }

    if (isTRUE(values$discussion_busy)) {
      showNotification("The clinical discussion agent is already responding.", type = "warning", duration = 4)
      return(invisible(NULL))
    }

    values$discussion_busy <- TRUE
    append_discussion_message("user", question_text)
    values$discussion_last_question <- question_text

    showModal(modalDialog(
      title = "Clinical Discussion",
      "Generating a discussion response grounded in the active RA-LMM result...",
      footer = NULL,
      easyClose = FALSE
    ))

    on.exit({
      values$discussion_busy <- FALSE
      removeModal()
    }, add = TRUE)

    reply_text <- generate_clinical_discussion_reply(question_text, mode)
    append_discussion_message("assistant", reply_text)
    values$discussion_last_answer <- reply_text
    updateTextAreaInput(session, "discussion_question", value = "")
  }

  env_api_key <- trimws(Sys.getenv("DEEPSEEK_API_KEY", unset = ""))
  if (nzchar(env_api_key)) {
    values$api_key <- env_api_key
    showNotification("DeepSeek API key loaded from environment.", type = "message", duration = 5)
  } else {
    showModal(modalDialog(
      title = div(icon("key"), " Welcome to RA-LMM Interpretation Agent: Configure API"),
      p("To use the DeepSeek interpretation agent, please provide your DeepSeek API Key. Your key is not stored permanently and is strictly used during this active session."),
      passwordInput("user_api_key", "DeepSeek API Key:", placeholder = "sk-..."),
      footer = tagList(
        actionButton("submit_api_key", "Save & Start App", class = "btn-primary", icon = icon("check"))
      ),
      easyClose = FALSE,
      backdrop = "static"
    ))
  }

  observeEvent(input$submit_api_key, {
    if (nchar(trimws(input$user_api_key)) < 5) {
      showNotification("Please enter a valid API key.", type = "error")
    } else {
      values$api_key <- trimws(input$user_api_key)
      removeModal()
      showNotification("API Key configured successfully. You may now use the app.", type = "message", duration = 5)
    }
  })

  model_components <- reactive({
    required_files <- c("logreg_model.rds", "feature_cols.rds", "target_levels.rds", "background_data.rds")
    missing_files <- required_files[!file.exists(file.path(MODEL_DIR, required_files))]
    if(length(missing_files) > 0) stop(paste("Missing required files:", paste(missing_files, collapse = ", ")))

    list(
      learner = readRDS(file.path(MODEL_DIR, "logreg_model.rds")),
      feature_cols = readRDS(file.path(MODEL_DIR, "feature_cols.rds")),
      target_levels = readRDS(file.path(MODEL_DIR, "target_levels.rds")),
      background_data = readRDS(file.path(MODEL_DIR, "background_data.rds"))
    )
  })

  predict_single_patient <- function(params) {
    tryCatch({
      sex_processed <- ifelse(tolower(as.character(params$Sex)) %in% c("female", "f", "woman", "2"), "2", "1")
      sex_factor <- factor(sex_processed, levels = c("1", "2"))

      input_data <- data.table(
        ANC = as.numeric(params$ANC), ALT = as.numeric(params$ALT), Cr = as.numeric(params$Cr),
        Sex = sex_factor, Age = as.numeric(params$Age), BMI = as.numeric(params$BMI),
        Muscle = factor(NA, levels = model_components()$target_levels)
      )

      task <- TaskClassif$new(id = "single_prediction", backend = input_data, target = "Muscle", positive = "Low")
      pred <- model_components()$learner$predict(task)
      prob_pos <- as.numeric(pred$prob[, "Low"])

      thresholds <- RISK_THRESHOLDS
      risk_level <- if(prob_pos < thresholds[1]) "Low" else if(prob_pos < thresholds[2]) "Intermediate" else "High"

      X_new_for_shap <- input_data[, model_components()$feature_cols, with = FALSE]
      key_drivers_df <- compute_key_drivers(
        learner = model_components()$learner, X_new = X_new_for_shap,
        bg_X = model_components()$background_data, feature_cols = model_components()$feature_cols,
        target_levels = model_components()$target_levels
      )

      list(probability = round(prob_pos, 4), risk_level = risk_level, thresholds = thresholds, features = params, key_drivers = key_drivers_df)
    }, error = function(e) stop(sprintf("Prediction error: %s", e$message)))
  }

  predict_batch_patients <- function(data, include_drivers = TRUE) {
    tryCatch({
      required_cols <- c("ANC","ALT","Cr","Sex","Age","BMI")
      missing_cols <- setdiff(required_cols, colnames(data))
      if(length(missing_cols) > 0) stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))

      n_patients <- nrow(data)
      data_processed <- copy(data)
      if(is.character(data_processed$Sex)) {
        data_processed$Sex <- ifelse(tolower(data_processed$Sex) %in% c("female", "f", "woman", "2"), "2", "1")
      }
      data_processed$Sex <- factor(data_processed$Sex, levels = c("1", "2"),labels = c("1", "2"))

      numeric_cols <- c("ANC","ALT","Cr","Age","BMI")
      for(col in numeric_cols) data_processed[[col]] <- as.numeric(data_processed[[col]])
      data_processed$Muscle <- factor(NA, levels = model_components()$target_levels)

      progress <- shiny::Progress$new()
      progress$set(message = "Processing batch predictions", value = 0)
      on.exit(progress$close())

      results <- data.table(
        PatientID = if("PatientID" %in% colnames(data)) data$PatientID else paste0("Patient_", 1:n_patients),
        ANC = data_processed$ANC, ALT = data_processed$ALT, Cr = data_processed$Cr,
        Sex = data_processed$Sex, Age = data_processed$Age, BMI = data_processed$BMI,
        Probability = numeric(n_patients), RiskLevel = character(n_patients), stringsAsFactors = FALSE
      )
      if(include_drivers) {
        results$Driver1 <- character(n_patients); results$Driver2 <- character(n_patients); results$Driver3 <- character(n_patients)
      }

      thresholds <- RISK_THRESHOLDS
      task_batch <- TaskClassif$new(id = "batch_prediction", backend = data_processed, target = "Muscle", positive = "Low")
      task_batch$select(model_components()$feature_cols)
      preds_batch <- model_components()$learner$predict(task_batch)
      probs <- as.numeric(preds_batch$prob[, "Low"])

      progress$set(message = "Calculating predictions", value = 0.3)
      results$Probability <- round(probs, 4)
      results$RiskLevel <- ifelse(probs < thresholds[1], "Low", ifelse(probs < thresholds[2], "Intermediate", "High"))

      if(include_drivers) {
        progress$set(message = "Calculating key drivers", value = 0.6)
        X_new_for_shap <- data_processed[, model_components()$feature_cols, with = FALSE]
        key_drivers_df <- compute_key_drivers(
          learner = model_components()$learner, X_new = X_new_for_shap,
          bg_X = model_components()$background_data, feature_cols = model_components()$feature_cols,
          target_levels = model_components()$target_levels
        )
        if(!is.null(key_drivers_df)) {
          results$Driver1 <- key_drivers_df$driver1
          results$Driver2 <- key_drivers_df$driver2
          results$Driver3 <- key_drivers_df$driver3
        }
      }

      progress$set(message = "Finalizing results", value = 0.9)
      summary_stats <- list(
        total_patients = n_patients, low_risk = sum(results$RiskLevel == "Low"),
        intermediate_risk = sum(results$RiskLevel == "Intermediate"), high_risk = sum(results$RiskLevel == "High"),
        mean_probability = mean(results$Probability), median_probability = median(results$Probability),
        sd_probability = sd(results$Probability), thresholds = thresholds
      )

      descriptive_stats <- if(n_patients > 0) {
        list(
          ANC = summary(data_processed$ANC), ALT = summary(data_processed$ALT), Cr = summary(data_processed$Cr),
          Sex_dist = table(data_processed$Sex), Age = summary(data_processed$Age), BMI = summary(data_processed$BMI)
        )
      } else { NULL }

      progress$set(value = 1)
      list(results = results, summary = summary_stats, descriptive = descriptive_stats)

    }, error = function(e) stop(sprintf("Batch prediction error: %s", e$message)))
  }

  observeEvent(input$go_one, {
    showModal(modalDialog(title = "Processing...", "Analyzing clinical parameters and consulting DeepSeek...", footer = NULL, easyClose = FALSE))

    result <- tryCatch({
      predict_single_patient(list(ANC = input$ANC, ALT = input$ALT, Cr = input$Cr, Sex = input$Sex, Age = input$Age, BMI = input$BMI))
    }, error = function(e) list(error = e$message))

    if(!is.null(result$error)) {
      removeModal()
      showNotification(paste("Error:", result$error), type = "error", duration = 5)
    } else {
      values$agent_cache <- new.env(parent = emptyenv())
      values$discussion_mode <- "single"
      reset_clinical_discussion()
      values$single_result <- result
      values$key_drivers <- result$key_drivers
      values$Advisor_followup_text <- NULL

      ai_response <- generate_ai_interpretation(result$risk_level, result$probability, result$key_drivers, values$api_key, cache_env = values$agent_cache)
      values$ai_text <- ai_response
      removeModal()
    }
  })

  observeEvent(input$go_batch, {
    req(input$batch_file)
    showModal(modalDialog(
      title = "Processing Batch Data...",
      div(class = "text-center", h4("Analyzing multiple patient records"), p("This may take a moment depending on the number of patients."), br(), div(class = "spinner-border text-primary", role = "status", span(class = "sr-only", "Loading..."))),
      footer = NULL, easyClose = FALSE, size = "m"
    ))

    tryCatch({
      uploaded_data <- fread(input$batch_file$datapath, stringsAsFactors = FALSE)
      batch_result <- predict_batch_patients(data = uploaded_data, include_drivers = input$include_drivers)

      values$agent_cache <- new.env(parent = emptyenv())
      values$discussion_mode <- "batch"
      reset_clinical_discussion()
      values$batch_results <- batch_result$results
      values$batch_summary <- batch_result$summary
      values$batch_descriptive <- batch_result$descriptive
      values$Advisor_batch_followup_text <- NULL

      # Use dynamic user API Key
      batch_ai_response <- generate_batch_ai_interpretation(batch_result$summary, values$api_key, cache_env = values$agent_cache)
      values$batch_ai_text <- batch_ai_response

      removeModal()
      showNotification("Batch prediction completed successfully!", type = "message", duration = 5)
    }, error = function(e) {
      removeModal()
      showNotification(paste("Error:", e$message), type = "error", duration = 10)
    })
  })

  ask_single_agent <- function(question_id, question) {
    if (isTRUE(values$single_agent_busy)) {
      showNotification("The interpretation agent is already working on this patient result.", type = "warning", duration = 4)
      return(invisible(NULL))
    }
    values$single_agent_busy <- TRUE
    showModal(modalDialog(title = "RA-LMM Risk Advisor", "Generating an interpretation response based on the current RA-LMM Risk Advisor result...", footer = NULL, easyClose = FALSE))
    on.exit({
      values$single_agent_busy <- FALSE
      removeModal()
    }, add = TRUE)
    response <- generate_Advisor_followup(question_id, question, values$single_result, values$api_key, cache_env = values$agent_cache)
    values$Advisor_followup_text <- response
  }

  observeEvent(input$ask_single_risk, {
    req(values$single_result)
    ask_single_agent("single_risk_explanation", Agent_get_followup_prompt("single", "single_risk_explanation"))
  })

  observeEvent(input$ask_single_drivers, {
    req(values$single_result)
    ask_single_agent("single_shap_drivers", Agent_get_followup_prompt("single", "single_shap_drivers"))
  })

  observeEvent(input$ask_single_thresholds, {
    req(values$single_result)
    ask_single_agent("single_thresholds", Agent_get_followup_prompt("single", "single_thresholds"))
  })

  observeEvent(input$ask_single_monitoring, {
    req(values$single_result)
    ask_single_agent("single_monitoring_focus", Agent_get_followup_prompt("single", "single_monitoring_focus"))
  })

  ask_batch_agent <- function(question_id, question) {
    if (isTRUE(values$batch_agent_busy)) {
      showNotification("The interpretation agent is already working on this batch result.", type = "warning", duration = 4)
      return(invisible(NULL))
    }
    values$batch_agent_busy <- TRUE
    showModal(modalDialog(title = "RA-LMM Risk Advisor Batch", "Generating an interpretation response based on the current batch result...", footer = NULL, easyClose = FALSE))
    on.exit({
      values$batch_agent_busy <- FALSE
      removeModal()
    }, add = TRUE)
    response <- generate_Advisor_batch_followup(question_id, question, values$batch_summary, values$api_key, cache_env = values$agent_cache)
    values$Advisor_batch_followup_text <- response
  }

  observeEvent(input$ask_batch_distribution, {
    req(values$batch_summary)
    ask_batch_agent("batch_distribution", Agent_get_followup_prompt("batch", "batch_distribution"))
  })

  observeEvent(input$ask_batch_probability, {
    req(values$batch_summary)
    ask_batch_agent("batch_mean_probability", Agent_get_followup_prompt("batch", "batch_mean_probability"))
  })

  observeEvent(input$ask_batch_monitoring, {
    req(values$batch_summary)
    ask_batch_agent("batch_monitoring_focus", Agent_get_followup_prompt("batch", "batch_monitoring_focus"))
  })

  observeEvent(input$ask_batch_limitations, {
    req(values$batch_summary)
    ask_batch_agent("batch_limitations", Agent_get_followup_prompt("batch", "batch_limitations"))
  })

  observeEvent(input$discussion_send, {
    submit_clinical_discussion(input$discussion_question)
  })

  observeEvent(input$discussion_quick_explain, {
    submit_clinical_discussion(discussion_quick_question(current_discussion_mode(), "explain"))
  })

  observeEvent(input$discussion_quick_summary, {
    submit_clinical_discussion(discussion_quick_question(current_discussion_mode(), "summarize"))
  })

  observeEvent(input$discussion_quick_limitations, {
    submit_clinical_discussion(discussion_quick_question(current_discussion_mode(), "limitations"))
  })

  observeEvent(input$discussion_quick_focus, {
    submit_clinical_discussion(discussion_quick_question(current_discussion_mode(), "focus"))
  })

  observeEvent(input$discussion_reset, {
    reset_clinical_discussion()
    showNotification("Discussion history cleared. Active prediction results were kept.", type = "message", duration = 4)
  })

  output$download_template <- downloadHandler(
    filename = function() { "batch_template.csv" },
    content = function(file) {
      template_data <- data.table(
        PatientID = c("P001", "P002", "P003"), ANC = c(3.5, 4.0, 2.8), ALT = c(25, 30, 40),
        Cr = c(22, 24, 26), Sex = c("1", "2", "1"), Age = c(65, 70, 75), BMI = c(22.5, 24.0, 21.0)
      )
      fwrite(template_data, file)
    }
  )

  output$download_results <- downloadHandler(
    filename = function() { paste0("batch_predictions_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv") },
    content = function(file) { req(values$batch_results); fwrite(values$batch_results, file) }
  )

  output$key_drivers_table <- renderDT({
    req(values$key_drivers)
    driver_descriptions <- data.table(
      Rank = c("Driver 1", "Driver 2", "Driver 3"),
      Description = c(values$key_drivers$driver1[1], values$key_drivers$driver2[1], values$key_drivers$driver3[1])
    )
    driver_descriptions <- driver_descriptions[!is.na(Description)]
    if(nrow(driver_descriptions) == 0) return(NULL)

    styled_descriptions <- sapply(driver_descriptions$Description, function(desc) {
      if(grepl("\u2191", desc, fixed = TRUE)) { gsub("\u2191", '<span class="driver-impact-up">\u2191</span>', desc, fixed = TRUE) }
      else if(grepl("\u2193", desc, fixed = TRUE)) { gsub("\u2193", '<span class="driver-impact-down">\u2193</span>', desc, fixed = TRUE) }
      else { desc }
    })
    driver_descriptions$Description <- styled_descriptions

    datatable(driver_descriptions, options = list(paging = FALSE, searching = FALSE, info = FALSE, ordering = FALSE, dom = 't'), class = 'cell-border stripe', rownames = FALSE, colnames = c("Rank", "Key Driver (Feature & Impact Direction)"), escape = FALSE) %>%
      formatStyle('Rank', fontWeight = 'bold', backgroundColor = '#f8f9fa')
  })

  output$batch_processing_ui <- renderUI({
    if(is.null(input$batch_file)) {
      div(class = "text-center", style = "padding: 50px;", icon("users", "fa-5x", style = "color: #ecf0f1; margin-bottom: 20px;"), h4("Upload a CSV file to begin batch assessment"), p("Use the upload area on the left to select your patient data file"))
    }
  })

  output$batch_summary_ui <- renderUI({
    req(values$batch_summary)
    summary <- values$batch_summary
    tagList(
      fluidRow(
        column(3, div(class = "batch-summary-box", div(class = "summary-value", summary$total_patients), div(class = "summary-label", "Total Patients"))),
        column(3, div(class = "batch-summary-box", div(class = "summary-value", span(class = "risk-badge badge-low", summary$low_risk)), div(class = "summary-label", "Low Risk"))),
        column(3, div(class = "batch-summary-box", div(class = "summary-value", span(class = "risk-badge badge-intermediate", summary$intermediate_risk)), div(class = "summary-label", "Intermediate Risk"))),
        column(3, div(class = "batch-summary-box", div(class = "summary-value", span(class = "risk-badge badge-high", summary$high_risk)), div(class = "summary-label", "High Risk")))
      ),
      fluidRow(
        column(4, div(class = "batch-summary-box", div(class = "summary-value", paste0(round(summary$mean_probability * 100, 1), "%")), div(class = "summary-label", "Mean Probability"))),
        column(4, div(class = "batch-summary-box", div(class = "summary-value", paste0(round(summary$median_probability * 100, 1), "%")), div(class = "summary-label", "Median Probability"))),
        column(4, div(class = "batch-summary-box", div(class = "summary-value", paste0(round(summary$sd_probability * 100, 2), "%")), div(class = "summary-label", "Std Deviation")))
      )
    )
  })

  output$batch_results_table <- renderDT({
    req(values$batch_results)
    results_display <- copy(values$batch_results)
    results_display$Probability <- sprintf("%.4f", results_display$Probability)
    results_display$Sex <- ifelse(results_display$Sex == "1", "Male", "Female")
    results_display$RiskLevel <- sapply(results_display$RiskLevel, function(risk) {
      badge_class <- switch(risk, "Low" = "badge-low", "Intermediate" = "badge-intermediate", "High" = "badge-high")
      paste0('<span class="risk-badge ', badge_class, '">', risk, '</span>')
    })

    datatable(results_display, options = list(pageLength = 10, scrollX = TRUE, dom = 'lfrtip', columnDefs = list(list(className = 'dt-center', targets = '_all'))), class = 'cell-border stripe hover', rownames = FALSE, escape = FALSE, selection = 'none') %>%
      formatStyle('RiskLevel', fontWeight = 'bold') %>% formatRound(columns = c('ANC','ALT','Cr','Age','BMI'), digits = 1)
  })

  output$descriptive_stats_ui <- renderUI({
    req(values$batch_descriptive)
    fluidRow(column(6, h5("Continuous Variables Summary"), verbatimTextOutput("cont_summary")), column(6, h5("Categorical Variables Distribution"), verbatimTextOutput("cat_summary")))
  })

  output$cont_summary <- renderPrint({
    req(values$batch_descriptive)
    cat("ANC (10^9/L):\n"); print(values$batch_descriptive$ANC)
    cat("\nALT (U/L):\n"); print(values$batch_descriptive$ALT)
    cat("\nCr (μmol/L):\n"); print(values$batch_descriptive$Cr)
    cat("\nAge (years):\n"); print(values$batch_descriptive$Age)
    cat("\nBMI (kg/m^2):\n"); print(values$batch_descriptive$BMI)
  })

  output$cat_summary <- renderPrint({
    req(values$batch_descriptive)
    cat("Sex Distribution:\n")
    sex_labels <- c("Male", "Female")
    sex_counts <- values$batch_descriptive$Sex_dist
    names(sex_counts) <- sex_labels[as.numeric(names(sex_counts))]
    print(sex_counts)
    cat("\nRisk Level Distribution:\n")
    if(!is.null(values$batch_results)) print(table(values$batch_results$RiskLevel))
  })

  output$batch_result_available <- reactive({ !is.null(values$batch_results) })
  outputOptions(output, "batch_result_available", suspendWhenHidden = FALSE)

  output$ai_agent_explanation_ui <- renderUI({
    req(values$ai_text)
    tagList(
      div(HTML(gsub("\n", "<br/>", values$ai_text))),
      format_Advisor_notes_ui()
    )
  })

  output$Advisor_followup_ui <- renderUI({
    req(values$Advisor_followup_text)
    div(
      style = "background-color: #ffffff; border: 1px solid #dfe6e9; border-radius: 6px; padding: 12px; line-height: 1.6;",
      tagList(
        div(HTML(gsub("\n", "<br/>", values$Advisor_followup_text))),
        format_Advisor_notes_ui()
      )
    )
  })

  output$Advisor_batch_followup_ui <- renderUI({
    req(values$Advisor_batch_followup_text)
    div(
      style = "background-color: #ffffff; border: 1px solid #dfe6e9; border-radius: 6px; padding: 12px; line-height: 1.6;",
      tagList(
        div(HTML(gsub("\n", "<br/>", values$Advisor_batch_followup_text))),
        format_Advisor_notes_ui()
      )
    )
  })

  output$batch_ai_summary_ui <- renderUI({
    req(values$batch_ai_text)
    tagList(
      div(HTML(gsub("\n", "<br/>", values$batch_ai_text))),
      format_Advisor_notes_ui()
    )
  })

  output$discussion_context_ui <- renderUI({
    mode <- current_discussion_mode()

    if (identical(mode, "single") && !is.null(values$single_result)) {
      result <- values$single_result
      tagList(
        div(class = "discussion-summary-row",
            span(class = "discussion-context-pill discussion-context-single", "Single Result Active")),
        br(), br(),
        div(class = "discussion-summary-row", span(class = "discussion-summary-label", "Probability: "), paste0(round(result$probability * 100, 1), "%")),
        div(class = "discussion-summary-row", span(class = "discussion-summary-label", "Risk level: "), result$risk_level),
        div(class = "discussion-summary-row", span(class = "discussion-summary-label", "Risk cutoffs: "), sprintf("0.0438; 0.1442")),
        div(class = "discussion-summary-row", span(class = "discussion-summary-label", "Top SHAP drivers: "), safe_Advisor_drivers_text(result))
      )
    } else if (identical(mode, "batch") && !is.null(values$batch_summary)) {
      summary <- values$batch_summary
      distribution <- sprintf(
        "High=%d (%.1f%%), Intermediate=%d (%.1f%%), Low=%d (%.1f%%)",
        summary$high_risk, (summary$high_risk / summary$total_patients) * 100,
        summary$intermediate_risk, (summary$intermediate_risk / summary$total_patients) * 100,
        summary$low_risk, (summary$low_risk / summary$total_patients) * 100
      )
      tagList(
        div(class = "discussion-summary-row",
            span(class = "discussion-context-pill discussion-context-batch", "Batch Result Active")),
        br(), br(),
        div(class = "discussion-summary-row", span(class = "discussion-summary-label", "Total patients: "), summary$total_patients),
        div(class = "discussion-summary-row", span(class = "discussion-summary-label", "Mean probability: "), paste0(round(summary$mean_probability * 100, 1), "%")),
        div(class = "discussion-summary-row", span(class = "discussion-summary-label", "Risk distribution: "), distribution),
        div(class = "discussion-summary-row", span(class = "discussion-summary-label", "Risk cutoffs: "), sprintf("0.0438; 0.1442"))
      )
    } else {
      tagList(
        div(class = "discussion-summary-row",
            span(class = "discussion-context-pill discussion-context-none", "No Active Result")),
        br(), br(),
        p(class = "discussion-note", "Run a single or batch prediction to start a result-centered discussion.")
      )
    }
  })

  output$discussion_quick_prompts_ui <- renderUI({
    mode <- current_discussion_mode()
    if (identical(mode, "none")) {
      return(p(class = "discussion-note", "Quick prompts will become available after a single or batch prediction is active."))
    }

    div(class = "btn-group-vertical", style = "width: 100%;",
        actionButton("discussion_quick_explain", "Explain this result", icon = icon("circle-question"), class = "btn btn-default"),
        actionButton("discussion_quick_summary", "Summarize the active result", icon = icon("file-alt"), class = "btn btn-default"),
        actionButton("discussion_quick_limitations", "Explain limitations", icon = icon("triangle-exclamation"), class = "btn btn-default"),
        actionButton("discussion_quick_focus", if (identical(mode, "single")) "Discuss key drivers" else "Discuss this cohort", icon = icon("search"), class = "btn btn-default")
    )
  })

  output$discussion_chat_ui <- renderUI({
    history <- values$discussion_history
    if (length(history) == 0) {
      return(div(
        class = "discussion-empty",
        h4("Start a clinical discussion"),
        p("Ask about the active RA-LMM result, request a summary, explore displayed drivers, or discuss limitations and interpretation boundaries.")
      ))
    }

    tagList(lapply(history, function(msg) {
      div(
        class = paste("discussion-message", msg$role),
        div(class = "discussion-meta", paste(if (identical(msg$role, "user")) "Clinician" else "Discussion Agent", msg$timestamp)),
        div(class = "discussion-bubble", HTML(gsub("\n", "<br/>", msg$content)))
      )
    }))
  })

  output$discussion_input_panel_ui <- renderUI({
    mode <- current_discussion_mode()
    input_box <- textAreaInput(
      "discussion_question",
      label = NULL,
      placeholder = "Ask about the current RA-LMM result, interpretation, or related discussion points...",
      rows = 4,
      width = "100%"
    )

    send_button <- actionButton("discussion_send", "Send", icon = icon("paper-plane"), class = "btn-primary")

    if (identical(mode, "none")) {
      input_box <- tagAppendAttributes(input_box, disabled = "disabled")
      send_button <- tagAppendAttributes(send_button, disabled = "disabled")
    }

    tagList(
      if (identical(mode, "none")) {
        p(class = "discussion-note", "Run a single or batch prediction to start discussion.")
      },
      input_box,
      div(style = "display: flex; justify-content: flex-end; margin-top: 8px;", send_button)
    )
  })

  output$prediction_result_ui <- renderUI({
    if(is.null(values$single_result)) {
      return(div(class = "text-center", style = "padding: 50px;", icon("stethoscope", "fa-5x", style = "color: #ecf0f1; margin-bottom: 20px;"), h4("Enter patient parameters to begin assessment"), p("Fill in the clinical values on the left and click 'Assess Risk'")))
    }

    result <- values$single_result
    risk_class <- switch(result$risk_level, "Low" = "low-risk", "Intermediate" = "intermediate-risk", "High" = "high-risk")
    risk_icon <- switch(result$risk_level, "Low" = icon("smile", class = "fa-3x", style = "color: #28a745;"), "Intermediate" = icon("meh", class = "fa-3x", style = "color: #ffc107;"), "High" = icon("frown", class = "fa-3x", style = "color: #dc3545;"))

    tagList(
      div(
        class = paste("risk-box", risk_class),
        fluidRow(
          column(2, class = "text-center", risk_icon),
          column(10, h3(paste(result$risk_level, "Risk"), style = "margin-top: 0;"), p(strong("Probability: "), paste0(round(result$probability * 100, 1), "%")))
        ),
        hr(),
        fluidRow(
          column(4, div(class = "metric-card", div(class = "metric-value", paste0(round(result$probability * 100, 1), "%")), div(class = "metric-label", "Low Muscle Mass Probability"))),
          column(4, div(class = "metric-card", div(class = "metric-value", round(result$thresholds[1], 4)), div(class = "metric-label", "Low Risk Threshold"))),
          column(4, div(class = "metric-card", div(class = "metric-value", round(result$thresholds[2], 4)), div(class = "metric-label", "High Risk Threshold")))
        ),
        hr(),

        h4(icon("chart-pie"), "Risk Interpretation"),
        tags$ul(
          tags$li(strong("Low Risk (P < ", round(result$thresholds[1], 4), "):"), " Low model-derived probability of low muscle mass."),
          tags$li(strong("Intermediate Risk (", round(result$thresholds[1], 4), " <= P < ", round(result$thresholds[2], 4), "):"), " Intermediate model-derived probability of low muscle mass."),
          tags$li(strong("High Risk (P >= ", round(result$thresholds[2], 4), "):"), " High model-derived probability of low muscle mass.")
        ),
        hr(),
        h4(icon("notes-medical"), "Input Parameters"),
        fluidRow(
          column(2, strong("ANC:"), br(), result$features$ANC, " 10^9/L"),
          column(2, strong("ALT:"), br(), result$features$ALT, " U/L"),
          column(2, strong("Cr:"), br(), result$features$Cr, " μmol/L"),
          column(2, strong("Age:"), br(), result$features$Age, " years"),
          column(2, strong("BMI:"), br(), result$features$BMI, " kg/m^2"),
          column(2, strong("Sex:"), br(), ifelse(result$features$Sex == "1", "Male", "Female"))
        )
      )
    )
  })

  output$show_drivers <- reactive({
    !is.null(values$single_result) && !is.null(values$key_drivers) && any(!is.na(values$key_drivers$driver1))
  })
  outputOptions(output, "show_drivers", suspendWhenHidden = FALSE)
}

shinyApp(ui = ui, server = server)



