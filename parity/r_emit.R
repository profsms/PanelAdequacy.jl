library(panelcert)

float_fields <- c("rho", "Vn", "lambda_n", "n_eff",
                  "kappa_greedy", "kappa_designed")
integer_fields <- c("n", "N", "T", "d_K", "C", "effective_C")
all_fields <- c("n", "N", "T", "d_K", "rho", "Vn", "lambda_n", "n_eff",
                "kappa_greedy", "kappa_designed", "C", "effective_C",
                "verdict")

json_string <- function(x) {
  x <- gsub("\\\\", "\\\\\\\\", as.character(x))
  x <- gsub('"', '\\\\"', x, fixed = TRUE)
  paste0('"', x, '"')
}

method_for <- function(name) if (startsWith(name, "sparse")) "sparse" else "structured"

record <- function(data, method) {
  controls <- if ("control" %in% names(data)) data$control else NULL
  report <- cycle_report(data$y, data$x, data$unit, data$time,
                         method = method, controls = controls, interval = FALSE)
  greedy <- cycle_contrasts(data$x, data$unit, data$time, method = "greedy",
                            controls = controls)
  designed <- cycle_contrasts(data$x, data$unit, data$time, method = method,
                              controls = controls)
  d <- report$design; s <- report$statistic
  list(record = list(n = d$n, N = d$N, T = d$T, d_K = d$d_K, rho = d$rho,
                     Vn = designed$V_n, lambda_n = s$lambda_n,
                     n_eff = s$n_eff, kappa_greedy = greedy$kappa,
                     kappa_designed = designed$kappa, C = s$C,
                     effective_C = s$effective_C, verdict = report$verdict),
       system = designed)
}

packing_signature <- function(cs) {
  entries <- lapply(seq_along(cs$rows), function(i) {
    ord <- order(cs$rows[[i]])
    rows <- cs$rows[[i]][ord]
    signs <- cs$signs[[i]][ord]
    orientation <- if (signs[1L] < 0) -1 else 1
    list(first = rows[1L], signature = paste0(
      rows, ifelse(signs * orientation > 0, ":+", ":-"), collapse = ","))
  })
  ord <- order(vapply(entries, `[[`, integer(1), "first"))
  vapply(entries[ord], `[[`, character(1), "signature")
}

write_records <- function(path, records) {
  names_ <- sort(names(records))
  lines <- vapply(names_, function(name) {
    r <- records[[name]]
    fields <- vapply(all_fields, function(field) {
      value <- r[[field]]
      encoded <- if (field %in% float_fields) sprintf("%.17g", value) else
        if (field %in% integer_fields) sprintf("%d", as.integer(value)) else
          json_string(value)
      paste0(json_string(field), ": ", encoded)
    }, character(1))
    paste0("  ", json_string(name), ": {", paste(fields, collapse = ", "), "}")
  }, character(1))
  writeLines(c("{", paste(lines, collapse = ",\n"), "}"), path, useBytes = TRUE)
}

write_packing <- function(path, signatures) {
  names_ <- sort(names(signatures))
  lines <- vapply(names_, function(name) {
    values <- paste(vapply(signatures[[name]], json_string, character(1)), collapse = ", ")
    paste0("  ", json_string(name), ": [", values, "]")
  }, character(1))
  writeLines(c("{", paste(lines, collapse = ",\n"), "}"), path, useBytes = TRUE)
}

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[1L]) else "parity/r_emit.R"
root <- normalizePath(dirname(script_path), mustWork = TRUE)
out <- if (length(args) >= 1L) normalizePath(args[1L], mustWork = FALSE) else file.path(root, "r.json")
packing_out <- if (length(args) >= 2L) normalizePath(args[2L], mustWork = FALSE) else file.path(root, "r_packing.json")
records <- list(); signatures <- list()
for (path in sort(list.files(file.path(root, "designs"), "\\.csv$", full.names = TRUE))) {
  name <- tools::file_path_sans_ext(basename(path))
  data <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  result <- record(data, method_for(name))
  records[[name]] <- result$record
  signatures[[name]] <- packing_signature(result$system)
}
write_records(out, records)
write_packing(packing_out, signatures)
cat("wrote", out, "and", packing_out, "\n")
