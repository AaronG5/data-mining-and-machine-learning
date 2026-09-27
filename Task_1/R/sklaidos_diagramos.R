library(ggplot2)

FILE <- "../A27_be_virsutiniu_isskirciu.csv"
OUT_DIR <- "../kor_rez/sklaidos_diagramos"
PAIRS <- list(c("Sum_of_Luminosity", "LogOfAreas"))
LOG_X <- c("Sum_of_Luminosity")

df <- read.csv(FILE)
df$class <- trimws(df$class)

dir.create(OUT_DIR, showWarnings = FALSE)

for (pora in PAIRS) {
  rho <- cor(df[[pora[1]]], df[[pora[2]]], method = "spearman")
  scatterplot <- ggplot(df) +
    aes(x = .data[[pora[1]]], y = .data[[pora[2]]], color = class) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = lm, se = FALSE, color = "red") +
    scale_color_manual(values = c("Bumps" = "steelblue", "Other_Faults" = "darkorange")) +
    labs(subtitle = sprintf("Spirmeno koreliacijos koeficientas \u03c1 = %.2f", rho))
  if (pora[1] %in% LOG_X) {
    scatterplot <- scatterplot +
      scale_x_log10(name = paste0(pora[1], " (logaritminė skalė)"), labels = scales::label_comma())
  }
  ggsave(file.path(OUT_DIR, paste0(pora[1], "_", pora[2], ".png")), scatterplot, width = 7, height = 5)
}
