#!/usr/bin/env Rscript
# ═══════════════════════════════════════════════════════════════════════════
# viz_results.R — Extreme-tail Q-Q plot: Fed-cSPA vs Asymptotic baseline
#
# Usage:
#   Rscript viz/viz_results.R <input.parquet> <output.pdf>
#   Rscript viz/viz_results.R results/fedcspa_results.parquet viz/output/qq_extreme_tail.pdf
#
# Requirements:
#   R >= 4.3, ggplot2, ggrastr, ggsci, arrow (for Parquet), extrafont
#
# Design:
#   - Scatter points are rasterised via ggrastr::rasterise() to prevent
#     PDF bloat from millions of points, while axes/labels/legend remain
#     pure vector elements.
#   - Font: Helvetica (Nature Publishing Group standard).
#   - Colours: NPG palette via ggsci::scale_color_npg().
#   - X-axis: expected -log10(p) under uniform null.
#   - Y-axis: observed -log10(p).
#   - Two layers: Fed-cSPA (primary) and Asymptotic (baseline).
#   - Identity line (y = x) as a dashed reference.
#   - λ_GC (genomic inflation factor) annotated on-plot.
# ═══════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrastr)
  library(ggsci)
})

# ── Parse CLI args ────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript viz_results.R <input.parquet|csv> <output.pdf>\n")
  quit(status = 1)
}
input_path  <- args[1]
output_path <- args[2]

# ── Load data ─────────────────────────────────────────────────────────────
if (grepl("\\.parquet$", input_path)) {
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Package 'arrow' is required to read Parquet files. Install via: install.packages('arrow')")
  }
  df <- as.data.frame(arrow::read_parquet(input_path))
} else {
  df <- read.csv(input_path)
}

cat(sprintf("Loaded %d variant-pair results from %s\n", nrow(df), input_path))

# ── Compute expected p-values (uniform quantiles) ─────────────────────────
n <- nrow(df)

# Fed-cSPA: sort observed p-values
obs_spa  <- sort(df$pvalue_fedcspa)
obs_asym <- sort(df$pvalue_asymptotic)
expected <- (seq_len(n) - 0.5) / n  # Haase (1969) plotting positions

# -log10 transform
plot_df <- data.frame(
  expected = rep(-log10(expected), 2),
  observed = c(-log10(pmax(obs_spa, 1e-300)), -log10(pmax(obs_asym, 1e-300))),
  method   = factor(rep(c("Fed-cSPA", "Asymptotic"), each = n),
                     levels = c("Fed-cSPA", "Asymptotic"))
)

# ── Genomic inflation factor λ_GC ────────────────────────────────────────
# λ = median(χ²_obs) / 0.4549  (median of χ²₁ = 0.4549)
chi2_spa  <- qchisq(1 - obs_spa,  df = 1)
chi2_asym <- qchisq(1 - obs_asym, df = 1)
lambda_spa  <- median(chi2_spa,  na.rm = TRUE) / qchisq(0.5, df = 1)
lambda_asym <- median(chi2_asym, na.rm = TRUE) / qchisq(0.5, df = 1)

# ── Configure Helvetica font ─────────────────────────────────────────────
# Try to load Helvetica; fall back to sans if unavailable
tryCatch({
  if (requireNamespace("extrafont", quietly = TRUE)) {
    extrafont::loadfonts(device = "pdf", quiet = TRUE)
  }
}, error = function(e) NULL)

base_family <- ifelse(
  "Helvetica" %in% names(grDevices::pdfFonts()),
  "Helvetica",
  "sans"
)

# ── Build Q-Q plot ────────────────────────────────────────────────────────
max_val <- max(plot_df$expected, plot_df$observed, na.rm = TRUE) * 1.05

p <- ggplot(plot_df, aes(x = expected, y = observed, colour = method)) +

  # ── Identity reference line (pure vector) ──
  geom_abline(intercept = 0, slope = 1, linetype = "dashed",
              colour = "grey40", linewidth = 0.5) +

  # ── Rasterised scatter points (prevents PDF bloat) ──
  rasterise(
    geom_point(alpha = 0.6, size = 1.0, shape = 16),
    dpi = 300
  ) +

  # ── NPG colour palette ──
  scale_color_npg(
    name   = "Method",
    labels = c(
      bquote("Fed-cSPA" ~ (lambda[GC] == .(sprintf("%.3f", lambda_spa)))),
      bquote("Asymptotic" ~ (lambda[GC] == .(sprintf("%.3f", lambda_asym))))
    )
  ) +

  # ── Axis labels + limits ──
  scale_x_continuous(
    name   = expression(Expected ~ -log[10](italic(p))),
    limits = c(0, max_val),
    expand = c(0.02, 0)
  ) +
  scale_y_continuous(
    name   = expression(Observed ~ -log[10](italic(p))),
    limits = c(0, max_val),
    expand = c(0.02, 0)
  ) +

  # ── Title ──
  ggtitle(
    label    = "Extreme-tail Q-Q Plot",
    subtitle = "Fed-cSPA vs Asymptotic Baseline · MetaRareEpi"
  ) +

  # ── λ_GC annotation ──
  annotate(
    "text",
    x = max_val * 0.05, y = max_val * 0.95,
    label = sprintf("Fed-cSPA λ[GC] == %.3f", lambda_spa),
    hjust = 0, vjust = 1, size = 3.2, colour = "grey30",
    family = base_family, parse = TRUE
  ) +
  annotate(
    "text",
    x = max_val * 0.05, y = max_val * 0.88,
    label = sprintf("Asymptotic λ[GC] == %.3f", lambda_asym),
    hjust = 0, vjust = 1, size = 3.2, colour = "grey30",
    family = base_family, parse = TRUE
  ) +

  # ── Theme: publication-grade ──
  theme_bw(base_size = 11, base_family = base_family) +
  theme(
    plot.title       = element_text(face = "bold", size = 13, hjust = 0),
    plot.subtitle    = element_text(size = 9, colour = "grey40", hjust = 0),
    legend.position  = c(0.75, 0.20),
    legend.background = element_rect(fill = alpha("white", 0.85),
                                     colour = NA),
    legend.key.size  = unit(0.4, "cm"),
    legend.text      = element_text(size = 8),
    legend.title     = element_text(size = 9, face = "bold"),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(colour = "black", linewidth = 0.6),
    axis.ticks       = element_line(linewidth = 0.4),
    aspect.ratio     = 1,
  )

# ── Save PDF ──────────────────────────────────────────────────────────────
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

ggsave(
  filename = output_path,
  plot     = p,
  width    = 5.5,
  height   = 5.5,
  units    = "in",
  device   = cairo_pdf  # ensures Helvetica embedding
)

cat(sprintf("✓ Q-Q plot saved to %s\n", output_path))
cat(sprintf("  Fed-cSPA λ_GC = %.4f\n", lambda_spa))
cat(sprintf("  Asymptotic λ_GC = %.4f\n", lambda_asym))
cat(sprintf("  N variant pairs = %d\n", n))
