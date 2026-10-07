# ==============================================================================
# Fake News Detection Using R - Plumber REST API
# ==============================================================================

suppressPackageStartupMessages({
  library(plumber)
  library(jsonlite)
  library(text2vec)
  library(glmnet)
  library(Matrix)
})

# Locate model artifacts directory
base_dir <- getwd()
models_dir <- file.path(base_dir, "Back End", "models")
if (!dir.exists(models_dir)) {
  models_dir <- file.path(base_dir, "models")
}

model_file       <- file.path(models_dir, "model.rds")
vectorizer_file  <- file.path(models_dir, "vectorizer.rds")
tfidf_file       <- file.path(models_dir, "tfidf_model.rds")
stopwords_file   <- file.path(models_dir, "stopwords.rds")
metrics_file     <- file.path(models_dir, "metrics.json")

# Load artifacts into memory
cat("Loading trained model artifacts...\n")
model_loaded <- FALSE
cv_fit <- NULL
vectorizer <- NULL
tfidf_model <- NULL
english_stopwords <- NULL
metrics_data <- list()

if (file.exists(model_file) && file.exists(vectorizer_file) && file.exists(tfidf_file)) {
  cv_fit            <- readRDS(model_file)
  vectorizer        <- readRDS(vectorizer_file)
  tfidf_model       <- readRDS(tfidf_file)
  if (file.exists(stopwords_file)) {
    english_stopwords <- readRDS(stopwords_file)
  }
  if (file.exists(metrics_file)) {
    metrics_data    <- fromJSON(metrics_file)
  }
  model_loaded <- TRUE
  cat("Model and vectorizers successfully loaded into Plumber API.\n")
} else {
  warning("Model artifacts not yet found. Please run train_model.R first.")
}

# Preprocessing function for inference
clean_single_text <- function(txt) {
  if (is.null(txt) || length(txt) == 0 || is.na(txt)) return("")
  # Lowercase
  txt <- tolower(txt)
  # Remove URLs
  txt <- gsub("https?://\\S+|www\\.\\S+", " ", txt)
  # Remove publisher wire artifacts (e.g., 'reuters')
  txt <- gsub("\\b(reuters)\\b", " ", txt)
  # Remove punctuation, numbers, and special characters
  txt <- gsub("[^a-z\\s]", " ", txt)
  # Collapse whitespace
  txt <- gsub("\\s+", " ", txt)
  return(trimws(txt))
}

tok_fun <- function(x) {
  tokens <- word_tokenizer(x)
  lapply(tokens, function(tok) {
    if (!is.null(english_stopwords)) {
      tok[!tok %in% english_stopwords & nchar(tok) >= 3]
    } else {
      tok[nchar(tok) >= 3]
    }
  })
}

#* Enable CORS for cross-origin requests from frontend dashboard
#* @filter cors
function(req, res) {
  res$setHeader("Access-Control-Allow-Origin", "*")
  res$setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
  res$setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept")
  
  if (req$REQUEST_METHOD == "OPTIONS") {
    res$status <- 200
    return(list(status = "preflight ok"))
  }
  
  plumber::forward()
}

#* Health check endpoint
#* @get /health
#* @serializer unboxedJSON
function(res) {
  list(
    status = "healthy",
    service = "Fake News Detection API",
    engine = "R Plumber + glmnet",
    model_loaded = model_loaded,
    timestamp = as.character(Sys.time())
  )
}

#* Model evaluation metrics endpoint
#* @get /metrics
#* @serializer unboxedJSON
function(res) {
  if (!model_loaded && file.exists(metrics_file)) {
    metrics_data <- fromJSON(metrics_file)
  }
  if (length(metrics_data) == 0) {
    res$status <- 404
    return(list(error = "Metrics data not found. Please ensure train_model.R has completed."))
  }
  return(metrics_data)
}

#* Predict news authenticity endpoint
#* @post /predict
#* @serializer unboxedJSON
function(req, res, text = "") {
  tryCatch({
    # Check if text was provided via JSON body
    input_text <- text
    if ((is.null(input_text) || nchar(trimws(input_text)) == 0) && !is.null(req$postBody)) {
      parsed <- tryCatch(fromJSON(req$postBody), error = function(e) NULL)
      if (is.list(parsed) && !is.null(parsed$text)) {
        input_text <- parsed$text
      } else if (is.character(parsed) && length(parsed) == 1) {
        input_text <- parsed
      }
    }
    
    if (is.null(input_text) || nchar(trimws(input_text)) == 0) {
      res$status <- 400
      return(list(error = "Please provide valid 'text' in the request body or parameters."))
    }
    
    if (!model_loaded) {
      # Try reloading if trained recently
      if (file.exists(model_file) && file.exists(vectorizer_file) && file.exists(tfidf_file)) {
        cv_fit            <<- readRDS(model_file)
        vectorizer        <<- readRDS(vectorizer_file)
        tfidf_model       <<- readRDS(tfidf_file)
        if (file.exists(stopwords_file)) english_stopwords <<- readRDS(stopwords_file)
        if (file.exists(metrics_file)) metrics_data <<- fromJSON(metrics_file)
        model_loaded      <<- TRUE
      } else {
        res$status <- 503
        return(list(error = "Model is not loaded. Train the model first using train_model.R."))
      }
    }
    
    # 1. Clean incoming news text
    cleaned <- clean_single_text(input_text)
    if (nchar(cleaned) == 0) {
      res$status <- 400
      return(list(error = "Text contained only punctuation, numbers, or empty characters."))
    }
    
    # 2. Extract TF-IDF features
    it_infer <- itoken(cleaned, tokenizer = tok_fun, progressbar = FALSE)
    dtm_infer <- create_dtm(it_infer, vectorizer)
    dtm_infer_tfidf <- transform(dtm_infer, tfidf_model)
    
    # 3. Model inference (predict probability of class 1 = FAKE)
    fake_prob <- as.numeric(predict(cv_fit, newx = dtm_infer_tfidf, s = "lambda.1se", type = "response"))
    fake_prob <- min(max(fake_prob, 0.0001), 0.9999)
    real_prob <- 1 - fake_prob
    
    label <- ifelse(fake_prob >= 0.5, "FAKE", "REAL")
    confidence <- ifelse(label == "FAKE", fake_prob, real_prob)
    
    list(
      status = "success",
      prediction = label,
      confidence = round(confidence * 100, 2),
      confidence_ratio = round(confidence, 4),
      fake_probability = round(fake_prob * 100, 2),
      real_probability = round(real_prob * 100, 2),
      text_length_chars = nchar(input_text),
      cleaned_word_count = length(unlist(tok_fun(cleaned))),
      timestamp = as.character(Sys.time())
    )
  }, error = function(err) {
    res$status <- 500
    list(
      status = "error",
      message = as.character(err$message)
    )
  })
}
