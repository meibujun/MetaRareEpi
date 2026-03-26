#!/usr/bin/env Rscript
# ==============================================================================
# VISUALIZATION 1: Extreme Tail Q-Q Calibration
# Blueprint: Nature Genetics (R - ggplot2 + ggrastr)
#
# Rasterizes overplotted scatter points to keep PDF < 2MB while retaining 
# extreme detail, leaving text and axes as pure vectors. Uses ggsci NPG palette.
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrastr)
  library(ggsci)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript viz_qq_nature.R <input.parquet> <output.pdf>\n")
  quit(status = 1)
}
input_path  <- args[1]
output_path <- args[2]

if (grepl("\\.parquet$", input_path)) {
  df <- as.data.frame(arrow::read_parquet(input_path))
} else {
  df <- read.csv(input_path)
}

n <- nrow(df)
obs_spa  <- sort(df$pvalue_fedcspa)
obs_asym <- sort(df$pvalue_asymptotic)
expected <- (seq_len(n) - 0.5) / n

df_plot <- data.frame(
  Expected  = -log10(expected),
  P_Asymp   = -log10(pmax(obs_asym, 1e-300)),
  P_FedCSPA = -log10(pmax(obs_spa, 1e-300))
)

# Optional: if a naive column exists, plot it. For now just Asymptotic + Fed-cSPA.
p <- ggplot(df_plot, aes(x = Expected)) +
    geom_abline(slope = 1, intercept = 0, color = "#DC0000", linetype = "dashed", linewidth = 1.0) +
    
    # Rasterize layers to keep PDF small while retaining extreme detail
    rasterise(geom_point(aes(y = P_Asymp, color="Asymptotic SOTA (Tail Failure)"), shape=17, alpha=0.8, size=2.0), dpi=600) +
    rasterise(geom_point(aes(y = P_FedCSPA, color="MetaRareEpi (Exact Fed-cSPA)"), shape=16, size=2.5, alpha=0.9), dpi=600) +
    
    scale_color_npg() +
    # Ensure axes start exactly at 0 to match Nature style
    scale_x_continuous(expand = c(0,0)) +
    scale_y_continuous(expand = c(0,0)) +
    labs(x = expression(Expected ~ -log[10](italic(P))), 
         y = expression(Observed ~ -log[10](italic(P)))) +
    
    theme_classic(base_size = 14, base_family = "Helvetica") +
    theme(legend.position = c(0.35, 0.85), 
          legend.background = element_blank(),
          legend.title = element_blank(),
          axis.line = element_line(linewidth = 0.8),
          axis.text = element_text(color="black"))

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

# Width 89mm is standard single-column width for Nature Portfolio journals
ggsave(output_path, p, width = 89, height = 89, units = "mm", device=cairo_pdf)

cat(sprintf("✓ Nature Genetics Q-Q plot saved: %s\n", output_path))
