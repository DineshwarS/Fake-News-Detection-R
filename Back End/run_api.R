# ==============================================================================
# Entry point to launch the Plumber REST API Server
# Host: 127.0.0.1, Port: 8000
# ==============================================================================

suppressPackageStartupMessages({
  library(plumber)
})

base_dir <- getwd()
plumber_file <- file.path(base_dir, "Back End", "plumber.R")

if (!file.exists(plumber_file)) {
  plumber_file <- file.path(base_dir, "plumber.R")
}

if (!file.exists(plumber_file)) {
  stop(sprintf("Cannot find plumber.R file at: %s", plumber_file))
}

cat("============================================================\n")
cat("Starting Fake News Detection Plumber REST API Server...\n")
cat(sprintf("Plumber definition: %s\n", plumber_file))
cat("API URL: http://127.0.0.1:8000\n")
cat("Swagger UI: http://127.0.0.1:8000/__docs__/\n")
cat("Press Ctrl+C to terminate the server.\n")
cat("============================================================\n")

pr <- plumb(plumber_file)
pr$run(host = "127.0.0.1", port = 8000, docs = TRUE)
