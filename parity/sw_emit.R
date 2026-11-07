# Stepped-wedge parity emitter (R). Evaluates every case of sw_reference.csv and
# writes case,quantity,value (requires panelcert >= 0.9.0 installed or load_all).
#   Rscript parity/sw_emit.R tests/testthat/fixtures/sw_reference.csv parity/sw_r.csv
args <- commandArgs(trailingOnly = TRUE)
if (!requireNamespace("panelcert", quietly = TRUE) ||
    utils::packageVersion("panelcert") < "0.9.0") {
  devtools::load_all(Sys.getenv("PANELCERT_SRC", "."), quiet = TRUE)
} else library(panelcert)
ref <- utils::read.csv(args[1], stringsAsFactors = FALSE)
num <- function(x) if (is.na(x) || identical(x, "")) NULL else as.numeric(x)
out <- list(); cache <- list()
for (i in seq_len(nrow(ref))) {
  r <- ref[i, ]
  if (!(r$kind %in% c("balanced", "integer", "nested"))) next
  if (r$quantity == "phi_prime") {
    v <- sw_phi_nested(num(r$icc), num(r$cac), num(r$m))
  } else {
    if (is.null(cache[[r$case]])) {
      d <- as.numeric(strsplit(r$design, ";", fixed = TRUE)[[1]])
      a <- list(estimator = r$estimator)
      if (r$estimator == "GLS") {
        if (!is.null(num(r$phi))) a$phi <- num(r$phi)
        if (!is.null(num(r$icc))) { a$icc <- num(r$icc); a$cac <- num(r$cac); a$m <- num(r$m) }
      }
      cache[[r$case]] <- list(W = do.call(sw_exposure_weights, c(list(design = d), a)),
                              ec = do.call(sw_eti_cost, c(list(design = d), a)))
    }
    W <- cache[[r$case]]$W; ec <- cache[[r$case]]$ec
    v <- switch(r$quantity, Gamma = sw_gamma(W), kappa = sw_radius(W),
                it_ratio = ec$it_ratio, eti_cost = ec$eti_cost, var_it = ec$var_it,
                var_eti = ec$var_eti, unname(W[as.integer(sub("W_", "", r$quantity))]))
  }
  out[[length(out) + 1]] <- data.frame(case = r$case, quantity = r$quantity,
                                       value = sprintf("%.17g", v))
}
utils::write.csv(do.call(rbind, out), args[2], row.names = FALSE, quote = FALSE)
