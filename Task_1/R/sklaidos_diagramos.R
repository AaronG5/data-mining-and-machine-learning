library(ggplot2)

FILE <- "../A27_be_virsutiniu_isskirciu.csv"
KOR_DIR <- "../kor_rez"
OUT_DIR <- file.path(KOR_DIR, "sklaidos_diagramos")
PAIRS <- list(c("Sum_of_Luminosity", "LogOfAreas"), c("LogOfAreas", "Pixels_Areas"))
LOG_SCALE <- c("Sum_of_Luminosity", "Pixels_Areas")

df <- read.csv(FILE)
df$class <- trimws(df$class)
rho_visi <- read.csv(file.path(KOR_DIR, "koreliacija_spearman_visi.csv"), row.names = 1, check.names = FALSE)
p_visi <- read.csv(file.path(KOR_DIR, "koreliacija_p_visi.csv"), row.names = 1, check.names = FALSE)

dir.create(OUT_DIR, showWarnings = FALSE)

for (pora in PAIRS) {
  rho <- rho_visi[pora[1], pora[2]]
  p <- p_visi[pora[1], pora[2]]
  p_reiksme <- if (p < 0.001) "p < 0.001" else sprintf("p = %.3f", p)
  scatterplot <- ggplot(df) +
    aes(x = .data[[pora[1]]], y = .data[[pora[2]]], color = class) +
    geom_point(alpha = 0.5, size = 2.5) +
    geom_smooth(method = lm, se = FALSE, color = scales::alpha("red", 0.5)) +
    scale_color_manual(values = c("Bumps" = "steelblue", "Other_Faults" = "darkorange")) +
    labs(subtitle = sprintf("Spirmeno koreliacijos koeficientas \u03c1 = %.2f, %s", rho, p_reiksme))
  if (pora[1] %in% LOG_SCALE) {
    scatterplot <- scatterplot +
      scale_x_log10(name = paste0(pora[1], " (logaritminė skalė)"), labels = scales::label_comma())
  }
  if (pora[2] %in% LOG_SCALE) {
    scatterplot <- scatterplot +
      scale_y_log10(name = paste0(pora[2], " (logaritminė skalė)"), labels = scales::label_comma())
  }
  ggsave(file.path(OUT_DIR, paste0(pora[1], "_", pora[2], ".png")), scatterplot, width = 7, height = 5)
}
