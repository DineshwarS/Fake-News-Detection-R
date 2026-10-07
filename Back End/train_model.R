# ==============================================================================
# Fake News Detection Using R - Training Pipeline
# Model: Regularized Logistic Regression (glmnet) with TF-IDF (text2vec)
# ==============================================================================

suppressPackageStartupMessages({
  library(text2vec)
  library(glmnet)
  library(jsonlite)
  library(Matrix)
})

cat(">>> Step 1: Loading Datasets...\n")

# Locate directories
base_dir <- getwd()
dataset_dir <- file.path(base_dir, "Dataset")
if (!dir.exists(dataset_dir)) {
  dataset_dir <- file.path(base_dir, "dataset")
}

fake_path <- file.path(dataset_dir, "Fake.csv")
true_path <- file.path(dataset_dir, "True.csv")

if (!file.exists(fake_path) || !file.exists(true_path)) {
  stop(sprintf("Dataset files not found in: %s", dataset_dir))
}

fake_df <- read.csv(fake_path, stringsAsFactors = FALSE)
true_df <- read.csv(true_path, stringsAsFactors = FALSE)

cat(sprintf("Loaded Fake news records: %d\n", nrow(fake_df)))
cat(sprintf("Loaded True news records: %d\n", nrow(true_df)))

fake_df$label <- "FAKE"
true_df$label <- "REAL"

# Combine datasets
combined_df <- rbind(fake_df, true_df)
rm(fake_df, true_df)
gc()

cat(">>> Step 2: Cleaning & Preprocessing Dataset...\n")

# Use title + text
combined_df$text_full <- paste(combined_df$title, combined_df$text, sep = " ")

# Remove rows with empty text
combined_df <- combined_df[!is.na(combined_df$text_full) & nzchar(trimws(combined_df$text_full)), ]

# Remove duplicates based on text content
combined_df <- combined_df[!duplicated(combined_df$text_full), ]
cat(sprintf("Total unique valid articles: %d (Fake: %d, Real: %d)\n",
            nrow(combined_df),
            sum(combined_df$label == "FAKE"),
            sum(combined_df$label == "REAL")))

# Standard English stopwords
english_stopwords <- c(
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
  "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
  "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
  "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
  "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
  "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
  "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
  "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
  "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
  "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
  "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
  "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
  "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
  "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
  "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
  "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
  "yourself", "yourselves"
)

# Preprocessing function
clean_text_vec <- function(txts) {
  # Lowercase
  txts <- tolower(txts)
  # Remove URLs
  txts <- gsub("https?://\\S+|www\\.\\S+", " ", txts)
  # Strip publisher wire artifacts (e.g., 'reuters') to prevent trivial leakage
  txts <- gsub("\\b(reuters)\\b", " ", txts)
  # Remove punctuation, numbers, and special characters
  txts <- gsub("[^a-z\\s]", " ", txts)
  # Collapse whitespace
  txts <- gsub("\\s+", " ", txts)
  txts <- trimws(txts)
  return(txts)
}

cat("Preprocessing text corpus...\n")
combined_df$clean_text <- clean_text_vec(combined_df$text_full)

# Remove any empty after cleaning
valid_idx <- nzchar(combined_df$clean_text)
combined_df <- combined_df[valid_idx, ]

cat(">>> Step 3: Train / Test Split (80% Train, 20% Test)...\n")
set.seed(42)
n_total <- nrow(combined_df)
train_indices <- sample.int(n_total, size = floor(0.80 * n_total))

train_df <- combined_df[train_indices, ]
test_df  <- combined_df[-train_indices, ]

cat(sprintf("Training samples: %d | Testing samples: %d\n", nrow(train_df), nrow(test_df)))

cat(">>> Step 4: TF-IDF Feature Extraction (text2vec)...\n")
# Tokenizer with stopword filter
tok_fun <- function(x) {
  tokens <- word_tokenizer(x)
  lapply(tokens, function(tok) tok[!tok %in% english_stopwords & nchar(tok) >= 3])
}

it_train <- itoken(train_df$clean_text, tokenizer = tok_fun, progressbar = FALSE)
vocab <- create_vocabulary(it_train)
cat(sprintf("Initial vocabulary size: %d terms\n", nrow(vocab)))

# Prune vocabulary strictly on training set (avoid data leakage)
pruned_vocab <- prune_vocabulary(
  vocab,
  term_count_min = 5,
  doc_proportion_max = 0.5,
  vocab_term_max = 10000
)
cat(sprintf("Pruned vocabulary size: %d terms\n", nrow(pruned_vocab)))

vectorizer <- vocab_vectorizer(pruned_vocab)
dtm_train <- create_dtm(it_train, vectorizer)

tfidf_model <- TfIdf$new()
dtm_train_tfidf <- fit_transform(dtm_train, tfidf_model)
cat(sprintf("Train DTM matrix dimensions: %d rows x %d features\n", nrow(dtm_train_tfidf), ncol(dtm_train_tfidf)))

cat(">>> Step 5: Training Regularized Logistic Regression (glmnet)...\n")
# Binary target: 1 = FAKE, 0 = REAL
y_train <- ifelse(train_df$label == "FAKE", 1, 0)

# Train elastic net logistic regression
cv_fit <- cv.glmnet(
  x = dtm_train_tfidf,
  y = y_train,
  family = "binomial",
  alpha = 0.5,
  nfolds = 5,
  type.measure = "deviance"
)
cat(sprintf("Optimal lambda.1se: %.6f\n", cv_fit$lambda.1se))

cat(">>> Step 6: Evaluating Model on Unseen Test Set...\n")
it_test <- itoken(test_df$clean_text, tokenizer = tok_fun, progressbar = FALSE)
dtm_test <- create_dtm(it_test, vectorizer)
dtm_test_tfidf <- transform(dtm_test, tfidf_model)

pred_probs <- as.numeric(predict(cv_fit, newx = dtm_test_tfidf, s = "lambda.1se", type = "response"))
pred_labels <- ifelse(pred_probs >= 0.5, "FAKE", "REAL")
actual_labels <- test_df$label

# Evaluation metrics calculation
# Positive class = FAKE, Negative class = REAL
tp <- sum(pred_labels == "FAKE" & actual_labels == "FAKE")
fp <- sum(pred_labels == "FAKE" & actual_labels == "REAL")
tn <- sum(pred_labels == "REAL" & actual_labels == "REAL")
fn <- sum(pred_labels == "REAL" & actual_labels == "FAKE")

accuracy <- (tp + tn) / (tp + tn + fp + fn)
precision <- if ((tp + fp) > 0) tp / (tp + fp) else 0
recall <- if ((tp + fn) > 0) tp / (tp + fn) else 0
f1_score <- if ((precision + recall) > 0) 2 * (precision * recall) / (precision + recall) else 0

cat("--------------------------------------------------\n")
cat("TEST EVALUATION RESULTS:\n")
cat(sprintf("Accuracy : %.4f (%.2f%%)\n", accuracy, accuracy * 100))
cat(sprintf("Precision: %.4f (%.2f%%)\n", precision, precision * 100))
cat(sprintf("Recall   : %.4f (%.2f%%)\n", recall, recall * 100))
cat(sprintf("F1-Score : %.4f (%.2f%%)\n", f1_score, f1_score * 100))
cat(sprintf("Confusion Matrix:\n  TP (Fake as Fake): %d\n  FP (Real as Fake): %d\n  TN (Real as Real): %d\n  FN (Fake as Real): %d\n",
            tp, fp, tn, fn))
cat("--------------------------------------------------\n")

cat(">>> Step 7: Saving Model Artifacts & Metrics...\n")
models_dir <- file.path(base_dir, "Back End", "models")
if (!dir.exists(models_dir)) {
  dir.create(models_dir, recursive = TRUE)
}

saveRDS(cv_fit, file = file.path(models_dir, "model.rds"))
saveRDS(vectorizer, file = file.path(models_dir, "vectorizer.rds"))
saveRDS(tfidf_model, file = file.path(models_dir, "tfidf_model.rds"))
saveRDS(english_stopwords, file = file.path(models_dir, "stopwords.rds"))

metrics_obj <- list(
  model_name = "Logistic Regression (ElasticNet via glmnet)",
  feature_extractor = "TF-IDF (10,000 terms)",
  dataset_summary = list(
    total_samples = n_total,
    train_samples = nrow(train_df),
    test_samples = nrow(test_df),
    fake_count = sum(combined_df$label == "FAKE"),
    real_count = sum(combined_df$label == "REAL")
  ),
  accuracy = round(accuracy * 100, 2),
  precision = round(precision * 100, 2),
  recall = round(recall * 100, 2),
  f1_score = round(f1_score * 100, 2),
  confusion_matrix = list(
    true_positive = tp,
    false_positive = fp,
    true_negative = tn,
    false_negative = fn,
    total_tested = nrow(test_df)
  )
)

writeLines(toJSON(metrics_obj, pretty = TRUE, auto_unbox = TRUE),
           file.path(models_dir, "metrics.json"))

cat(">>> Model training pipeline completed successfully!\n")
